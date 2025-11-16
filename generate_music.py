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
import numpy as np

from mido import Message, MidiFile, MidiTrack, MetaMessage
import mido
def positions_to_midi(notes, output_path="generated.mid", ticks_per_beat=480):
    """
    Convert generated note vectors into a MIDI file.

    notes: array-like of shape (N, 4)
           columns = [start_time, duration, pitch, velocity]

    Output:
        generated.mid
    """

    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    # Add tempo (120 BPM)
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

    # Convert continuous times to ticks
    def to_ticks(time_in_units):
        return int(time_in_units * ticks_per_beat)  # scale factor if needed

    # Sort by start time just to be safe
    notes_sorted = sorted(notes, key=lambda x: x[0])

    last_tick = 0

    for (start, dur, pitch, velocity) in notes_sorted:
        pitch = int(pitch)
        velocity = int(velocity)

        start_tick = to_ticks(start)
        duration_tick = to_ticks(dur)

        # delta time = time since last event
        delta = max(0, start_tick - last_tick)

        # Note on
        track.append(Message('note_on', note=pitch, velocity=velocity, time=delta))

        # Note off
        track.append(Message('note_off', note=pitch, velocity=0, time=duration_tick))

        last_tick = start_tick + duration_tick

    midi.save(output_path)
    print(f"Saved MIDI to {output_path}")

def f(v, c):
    return v//c*c


def process_last_logits(pairs, device='cuda'):
    # angle of each pair (1,16)
    angles = torch.atan2(pairs[..., 1], pairs[..., 0])

    # ----------------------------
    # Convert angles → positions
    # ----------------------------
    angles = (angles / torch.pi).clamp(0, 1)
    pos0 = angles[0] * 0.25
    pos1 = angles[1] * 1
    pos2 = angles[2] * 8
    pos3 = angles[3] * 64

    pos4 = angles[4] * 0.25
    pos5 = angles[5] * 1
    pos6 = angles[6] * 8
    pos7 = angles[7] * 64

    pos8 = angles[8] * 4
    pos9 = angles[9] * 16
    pos10 = angles[10] * 64
    pos11 = angles[11] * 128

    pos12 = angles[12] * 4
    pos13 = angles[13] * 16
    pos14 = angles[14] * 64
    pos15 = angles[15] * 128

    feature0 = f(pos3, 8) + f(pos2, 1) + f(pos1, 0.25) + pos0
    feature1 = f(pos7, 8) + f(pos6, 1) + f(pos5, 0.25) + pos4
    feature2 = f(pos11, 64) + f(pos10, 16) + f(pos9, 4) + pos8
    feature3 = f(pos15, 64) + f(pos14, 16) + f(pos13, 4) + pos12
    next_pos = torch.stack([feature0, feature1, feature2, feature3]).to(device)
    next_pos = next_pos.unsqueeze(0).unsqueeze(0)
    return next_pos

from src.model.modeling import _compute_encoding
@torch.no_grad()
def generate_music(model, batch, total_length: int = 200, device='cuda'):
    """
    Autoregressive generation for a 4-dim position model that uses only:
        input_ids
        attention_mask
        position_tensors
    """

    # BOS = [0,0,0,0]
    # generated = torch.zeros(1, 1, 4, device=device)
    batch['input_ids'] = batch['input_ids'].to(device)
    batch['attention_mask'] = batch['attention_mask'].to(device)
    batch['position_tensors'] = batch['position_tensors'].to(device)
    batch['labels'] = batch['labels'].to(device)

    print(batch['position_tensors'].shape)
    print(batch['labels'].shape)
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch['input_ids'][0:1, :],
            attention_mask=batch['attention_mask'][0:1, :],
            position_tensors=batch['position_tensors'][0:1, :,:],
        )
    att = model.model.model.layers[0].self_attn._attn_out
    hid1 = model.model.model.layers[0].self_attn._debug_first_hidden
    q = model.model.model.layers[0].self_attn._q
    k = model.model.model.layers[0].self_attn._k
    v = model.model.model.layers[0].self_attn._v
    aw = model.model.model.layers[0].self_attn._test_attn_w

    with torch.no_grad():
        L = 1
        # --- RUN MODEL ---
        out = model(
            input_ids=batch['input_ids'][0:1, 0:L],
            attention_mask=batch['attention_mask'][0:1, 0:L],
            position_tensors=batch['position_tensors'][0:1, 0:L,:],
        )

    att2 = model.model.model.layers[0].self_attn._attn_out
    hid2 = model.model.model.layers[0].self_attn._debug_first_hidden
    q2 = model.model.model.layers[0].self_attn._q
    k2 = model.model.model.layers[0].self_attn._k
    v2 = model.model.model.layers[0].self_attn._v
    aw2 = model.model.model.layers[0].self_attn._test_attn_w

    q_diff = (q - q2).abs().max().item()
    k_diff = (k - k2).abs().max().item()
    v_diff = (v - v2).abs().max().item()

    print("\n===== attention weights =====")
    print(aw)
    print(aw2)

    print("\n===== Q/K/V DIFFS =====")
    print(f"max |Q_full - Q_masked| = {q_diff}")
    print(f"max |K_full - K_masked| = {k_diff}")
    print(f"max |V_full - V_masked| = {v_diff}")

    diff = (att2 - att).abs().max()
    print("max diff attn head 0 batch 0 seq 0:", diff.item())

    diff = (hid1 - hid2).abs().max()
    print("max diff hidden batch 0 seq 0:", diff.item())

    print(process_last_logits(out.logits[0,0,:,:]))
    print(batch['labels'][0,0,:])

    return None


def main():
    parser = argparse.ArgumentParser(description='Generate music using trained 4D model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_music.mid', help='Output MIDI file path')
    parser.add_argument('--length', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)

    # Generate music
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = project_dir / "data" / "aria-midi-v1-unique-ext" / "data"

    # Get all MIDI files
    all_files = list(sorted(data_dir.glob("**/*.mid")))
    from sklearn.model_selection import train_test_split
    train_files, val_files = train_test_split(all_files, test_size=0.05, random_state=42)
    print(train_files[:1])

    from torch.utils.data import DataLoader, Dataset
    from src.train_model_new import MidiDataset4D, custom_collate_fn
    from src.utils import CONTEXT_SIZE
    val_dataset = MidiDataset4D(train_files[:1], max_seq_len=CONTEXT_SIZE)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=8
    )
    val_iter = iter(val_loader)
    batch = next(val_iter)

    generated_positions = generate_music(model, batch, total_length=args.length, device=args.device)

    # Convert to MIDI
    # positions_to_midi(generated_positions, args.output)

    print(f"Music generation complete! Check {args.output}")


if __name__ == "__main__":
    main()