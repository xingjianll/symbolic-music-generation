"""
Generate music using a trained 4D continuous music generation model.

Usage:
    python generate_music.py --checkpoint path/to/checkpoint.ckpt --output output.mid --length 100
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import symusic
from src.model.model import MidiQwenNew
from src.train_model_new import MidiDataset4DStreaming, MidiDataset4D, custom_collate_fn
from src.utils import CONTEXT_SIZE


def load_model_from_checkpoint(checkpoint_path: str, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create dummy tokenizer (same as training)
    class DummyTokenizer:
        pad_token_id = 0
        def __getitem__(self, key):
            return 0

    dummy_tokenizer = DummyTokenizer()
    dummy_dataloader = []  # Not used for inference

    try:
        # Try normal loading first
        model = MidiQwenNew.load_from_checkpoint(
            checkpoint_path,
            tokenizer=dummy_tokenizer,
            dataloader=dummy_dataloader,
            map_location=device
        )
    except Exception as e:
        print(f"Normal loading failed: {e}")
        print("Trying manual state dict loading...")

        # Manual loading as fallback
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create fresh model
        model = MidiQwenNew(dummy_tokenizer, dummy_dataloader)

        # Load only the model weights, ignore other hyperparameters
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    model.to(device)

    return model


import torch

def decode_sinusoidal_to_4d(
        sinusoidal_output: torch.Tensor,
        theta: float = 10000.0,
) -> torch.Tensor:
    """
    Decode sinusoidal encodings back to 4D positions by reversing the sinusoidal encoding.

    Assumes the encoding was of the form (per scalar dim):
        for each frequency f in freq_bands:
            cos = cos(pos_d * f)
            sin = sin(pos_d * f)
    and that the final layout is:
        [cos(f0_dim0), sin(f0_dim0), cos(f1_dim0), sin(f1_dim0), ...,
         cos(f0_dim1), sin(f0_dim1), cos(f1_dim1), sin(f1_dim1), ...,
         ... for 4 dims total ...]

    Args:
        sinusoidal_output: (batch, seq, head_dim) tensor with sinusoidal encodings
                           in [cos, sin, cos, sin, ...] order
        theta: base used in frequency computation (must match the encoder)

    Returns:
        positions_4d: (batch, seq, 4) tensor of [time, duration, pitch, velocity]
    """
    batch_size, seq_len, head_dim = sinusoidal_output.shape
    device = sinusoidal_output.device

    # We assume 4 scalar dimensions, each using the same number of frequencies.
    # Each frequency contributes 2 channels (cos, sin).
    # head_dim = 4 * num_freq * 2  =>  num_freq = head_dim // 8
    if head_dim % 8 != 0:
        raise ValueError(
            f"head_dim must be divisible by 8 (got {head_dim}). "
            "Expected 4 dimensions, each with N frequencies and 2 channels (cos, sin)."
        )

    num_freq = head_dim // 8  # per scalar dimension

    # Split into cos and sin:
    # sinusoidal_output[..., 0] = cos of pair 0
    # sinusoidal_output[..., 1] = sin of pair 0
    # sinusoidal_output[..., 2] = cos of pair 1, etc.
    cos_values = sinusoidal_output[:, :, 0::2]  # (batch, seq, head_dim//2)
    sin_values = sinusoidal_output[:, :, 1::2]  # (batch, seq, head_dim//2)

    # Extract wrapped phases: phase_ij = atan2(sin_ij, cos_ij) = (freq_j * x_d) mod 2π
    phases = torch.atan2(sin_values, cos_values)  # (batch, seq, head_dim//2)

    # Reshape to group by dimension and frequency:
    # phases_per_dim[b, t, d, k] = phase for dim d, frequency index k
    phases_per_dim = phases.view(batch_size, seq_len, 4, num_freq)  # 4 dims

    # Recreate the frequency bands as in the encoder.
    #
    # Typical transformer-style:
    #   freq_bands[k] = 1 / (theta ** (2k / head_dim))
    #
    # Here head_dim//2 channels correspond to all (cos,sin) pairs,
    # and we used num_freq = head_dim//8 per dimension,
    # so the exponent index step is 2, giving length num_freq.
    freq_indices = torch.arange(0, num_freq * 2, 2, device=device, dtype=torch.float32)
    freq_bands = 1.0 / (theta ** (freq_indices / head_dim))  # (num_freq,)

    # Prepare output
    positions_4d = torch.zeros(batch_size, seq_len, 4, device=device, dtype=torch.float32)

    # Least-squares denominator (scalar)
    # x ≈ argmin_x Σ_i (freq_i * x - phase_i_unwrapped)^2
    #    => x = Σ_i freq_i * phase_i_unwrapped / Σ_i freq_i^2
    den = (freq_bands * freq_bands).sum()  # scalar

    # Broadcastable frequency tensor for the numerator
    omega = freq_bands.view(1, 1, num_freq)  # (1, 1, num_freq)

    for d in range(4):
        # phases for this dimension: (batch, seq, num_freq)
        dim_phases = phases_per_dim[:, :, d, :]

        # Unwrap freq * x across the frequency axis to remove 2π jumps
        try:
            thetas_unwrapped = torch.unwrap(dim_phases, dim=-1)
        except AttributeError:
            # Manual unwrap fallback for older PyTorch
            diff = torch.diff(dim_phases, dim=-1)
            diff_wrapped = torch.where(diff > torch.pi, diff - 2 * torch.pi, diff)
            diff_wrapped = torch.where(diff_wrapped < -torch.pi, diff_wrapped + 2 * torch.pi, diff_wrapped)
            thetas_unwrapped = torch.zeros_like(dim_phases)
            thetas_unwrapped[:, :, 0] = dim_phases[:, :, 0]
            thetas_unwrapped[:, :, 1:] = dim_phases[:, :, 0:1] + diff_wrapped.cumsum(dim=-1)

        # Numerator: Σ_i freq_i * (freq_i * x) ≈ Σ_i freq_i * phase_i_unwrapped
        num = (omega * thetas_unwrapped).sum(dim=-1)  # (batch, seq)

        positions_4d[:, :, d] = num / den  # (batch, seq)

    return positions_4d



def clamp_4d_positions(positions_4d: torch.Tensor) -> torch.Tensor:
    """Clamp 4D positions to reasonable musical ranges."""
    # Clamp to reasonable ranges
    positions_4d[:, :, 0] = torch.clamp(positions_4d[:, :, 0], 0, 300)      # time: 0-300 seconds
    positions_4d[:, :, 1] = torch.clamp(positions_4d[:, :, 1], 0, 10)       # duration: 0-10 seconds
    positions_4d[:, :, 2] = torch.clamp(positions_4d[:, :, 2], 0, 127)      # pitch: 0-127
    positions_4d[:, :, 3] = torch.clamp(positions_4d[:, :, 3], 0, 127)      # velocity: 0-127

    return positions_4d


def positions_to_midi(positions_4d: torch.Tensor, output_path: str):
    """Convert 4D positions back to MIDI file."""
    positions = positions_4d[0].cpu().numpy()  # Take first batch item

    # Create new score
    score = symusic.Score()
    track = symusic.Track(program=0, is_drum=False, name="Generated Music")

    current_time = 0.0
    for i, pos in enumerate(positions):
        start_time, duration, pitch, velocity = pos

        # Skip special tokens
        if pitch <= 2:  # BOS, EOS, PAD tokens
            continue

        # Ensure reasonable values
        if duration <= 0 or velocity <= 0 or pitch < 21 or pitch > 108:
            continue

        # Create note
        note = symusic.Note(
            start=int(start_time * 480),  # Convert to ticks (480 ticks per quarter note)
            end=int((start_time + duration) * 480),
            pitch=int(pitch),
            velocity=int(velocity)
        )
        track.notes.append(note)

    score.tracks.append(track)
    score.dump_midi(output_path)
    print(f"Generated MIDI saved to {output_path}")


def load_midi_prompt(data_dir, num_notes: int = 20):
    """Load the first MIDI file and extract the first num_notes as prompt."""
    # Get first MIDI file
    all_files = list(data_dir.glob("**/*.mid"))
    if not all_files:
        raise ValueError(f"No MIDI files found in {data_dir}")

    first_file = all_files[0]
    print(f"Loading prompt from: {first_file.name}")

    # Load and process the MIDI file (same as in training)
    from src.utils import merge_score_tracks
    score = symusic.Score.from_file(str(first_file))
    merge_score_tracks(score)
    score = score.to("second")

    if not score.tracks or len(score.tracks[0].notes) == 0:
        raise ValueError("No notes found in MIDI file")

    track = score.tracks[0]
    all_notes = list(track.notes)
    all_notes.sort(key=lambda x: x.start)

    # Take first num_notes
    prompt_notes = all_notes[:min(num_notes, len(all_notes))]
    print(f"Using {len(prompt_notes)} notes as prompt")

    # Convert to 4D positions
    positions = []
    for note in prompt_notes:
        positions.append([
            float(note.start),
            float(note.duration),
            float(note.pitch),
            float(note.velocity)
        ])

    return torch.tensor(positions, dtype=torch.float32)


def generate_music(model, prompt_length: int = 20, total_length: int = 200, device='cuda'):
    """Generate music using the trained model with MIDI file prompt."""
    print(f"Generating {total_length} tokens...")

    # Load prompt from first MIDI file
    from pathlib import Path
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = project_dir / "data" / "aria-midi-v1-unique-ext" / "data"

    prompt_positions = load_midi_prompt(data_dir, num_notes=prompt_length)

    # Start with BOS token followed by prompt
    bos_token = torch.tensor([[0.0, 0.0, 0, 0]], dtype=torch.float32)
    prompt_with_bos = torch.cat([bos_token, prompt_positions], dim=0)

    # Convert to batch format and move to device
    current_positions = prompt_with_bos.unsqueeze(0).to(device)  # (1, prompt_length+1, 4)

    print(f"Prompt shape: {current_positions.shape}")
    print(f"First few prompt positions:\n{current_positions[0, :5]}")

    with torch.no_grad():
        for step in range(total_length):
            if step % 20 == 0:
                print(f"Step {step}/{total_length}")

            # Prepare input (pad to context size if needed)
            seq_len = current_positions.shape[1]
            if seq_len >= CONTEXT_SIZE:
                # Take the last CONTEXT_SIZE-1 tokens to leave room for new generation
                current_positions = current_positions[:, -(CONTEXT_SIZE-1):]
                seq_len = current_positions.shape[1]

            # Pad to context size
            padded_positions = torch.zeros(1, CONTEXT_SIZE, 4, device=device)
            padded_positions[:, :seq_len] = current_positions

            # Create input_ids (all zeros for vocab_size=1)
            input_ids = torch.zeros(1, CONTEXT_SIZE, dtype=torch.long, device=device)

            # Create attention mask
            attention_mask = torch.zeros(1, CONTEXT_SIZE, dtype=torch.long, device=device)
            attention_mask[:, :seq_len] = 1

            # Forward pass
            # print(f"Input context last 3 positions: {padded_positions[0, seq_len-3:seq_len]}")
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_tensors=padded_positions
            )

            # Get the prediction for the next token (last non-padded position)
            next_sinusoidal = outputs.logits[0, seq_len-1:seq_len]  # (1, head_dim) - sinusoidal encoding

            # Print position (time) sinusoidal values (1st dimension)
            # head_dim=128, split into 4 groups of 32 for each dimension
            # Position/time is the 1st group: indices 0-31
            position_sinusoids = next_sinusoidal[0, 0:32]  # 32 values for position/time
            print(f"Position sinusoids (first 8 pairs): {position_sinusoids[:16].cpu().numpy()}")

            # Extract phases using atan2 for position dimension
            cos_values = position_sinusoids[0::2]  # cos values at even indices
            sin_values = position_sinusoids[1::2]  # sin values at odd indices
            phases = torch.atan2(sin_values, cos_values)  # 16 phase values
            print(f"Position phases after atan2 (first 8): {phases[:8].cpu().numpy()}")

            # Remove frequency component to recover position values
            # Get the frequency bands used for encoding
            theta = 10000
            head_dim = 128
            freq_indices = torch.arange(0, head_dim // 4, 2, device=next_sinusoidal.device, dtype=torch.float32)
            freq_bands = 1.0 / (theta ** (freq_indices / head_dim))  # 16 frequency bands

            # Divide phases by frequencies to get position estimates
            position_estimates = phases / freq_bands
            print(f"Position estimates after removing freq (first 8): {position_estimates[:8].cpu().numpy()}")

            # Simple decoding: just use the lowest frequency (most reliable) for each dimension
            # Split sinusoidal output into 4 dimensions (32 values each)
            dim_sinusoids = next_sinusoidal[0].view(4, 32)  # (4, 32)

            # For each dimension, extract the last cos/sin pair (lowest frequency)
            positions_4d = torch.zeros(1, 1, 4, device=next_sinusoidal.device)
            for d in range(4):
                cos_val = dim_sinusoids[d, -2]  # Last cos value (second to last in the 32)
                sin_val = dim_sinusoids[d, -1]  # Last sin value (last in the 32)
                phase = torch.atan2(sin_val, cos_val)
                # Lowest frequency is the last one: freq_bands[-1]
                lowest_freq = freq_bands[-1]  # ~0.129
                positions_4d[0, 0, d] = phase / lowest_freq

            # Comment out complex decoding
            # next_position = decode_sinusoidal_to_4d(next_sinusoidal.unsqueeze(0))  # (1, 1, 4)

            next_position = clamp_4d_positions(positions_4d)

            # Print the 4D values with annotations
            values = next_position[0, 0].cpu().numpy()
            print(f"Step {step}: time={values[0]:.2f}s, duration={values[1]:.2f}s, pitch={values[2]:.0f}, velocity={values[3]:.0f}")

            # Append to sequence
            current_positions = torch.cat([current_positions, next_position], dim=1)

    return current_positions


def main():
    parser = argparse.ArgumentParser(description='Generate music using trained 4D model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_music.mid', help='Output MIDI file path')
    parser.add_argument('--length', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--prompt-notes', type=int, default=20, help='Number of notes to use as prompt')

    args = parser.parse_args()

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    # Generate music
    generated_positions = generate_music(model, total_length=args.length, device=args.device)

    # Convert to MIDI
    positions_to_midi(generated_positions, args.output)

    print(f"Music generation complete! Check {args.output}")


if __name__ == "__main__":
    main()