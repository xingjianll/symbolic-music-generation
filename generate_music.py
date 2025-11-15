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
    position_tensors = batch['position_tensors']
    labels = batch['labels']
    labels = labels.to(device)
    original = position_tensors.clone()
    position_tensors = position_tensors[:,:1,:].to(device)
    print(position_tensors)

    for step in range(total_length):

        # attention mask = ones because every pos is valid
        attn_mask = torch.ones(position_tensors.shape[:-1], device=device)  # (1, seq)
        input_ids = torch.zeros(position_tensors.shape[:-1], dtype=torch.long, device=device)

        # run model
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_tensors=position_tensors
        )

        # model returns logits: (1, seq, 4, 2)
        pairs = out.logits[:, -1, :, :]
        t = _compute_encoding(labels)
        print("--------")
        print(labels[0, 1, :])
        print(t[0, 1, :, :])
        print(out.logits[0, -1, :, :])
        print((out.logits[0, -1, :, :] * t[0, 1, :, :]).sum(dim=-1))
        print("--------")

        # angle of each pair (1,16)
        angles = torch.atan2(pairs[..., 1], pairs[..., 0])

        # ----------------------------
        # Convert angles → positions
        # ----------------------------
        angles = (angles / torch.pi).clamp(0, 1)
        pos0 = angles[0, 0] * 0.25
        pos1 = angles[0, 1] * 1
        pos2 = angles[0, 2] * 8
        pos3 = angles[0, 3] * 64

        pos4 = angles[0, 4] * 0.25
        pos5 = angles[0, 5] * 1
        pos6 = angles[0, 6] * 8
        pos7 = angles[0, 7] * 64

        pos8 = angles[0, 8] * 4
        pos9 = angles[0, 9] * 16
        pos10 = angles[0, 10] * 64
        pos11 = angles[0, 11] * 128

        pos12 = angles[0, 12] * 4
        pos13 = angles[0, 13] * 16
        pos14 = angles[0, 14] * 64
        pos15 = angles[0, 15] * 128

        feature0 = f(pos3, 8) + f(pos2, 1) + f(pos1, 0.25) + pos0
        feature1 = f(pos7, 8) + f(pos6, 1) + f(pos5, 0.25) + pos4
        feature2 = f(pos11, 64) + f(pos10, 16) + f(pos9, 4) + pos8
        feature3 = f(pos15, 64) + f(pos14, 16) + f(pos13, 4) + pos12
        next_pos = torch.stack([position_tensors[0, -1, 0]+feature0, feature1, feature2, feature3]).to(device)
        next_pos = next_pos.unsqueeze(0).unsqueeze(0)
        print(next_pos)

        # append to the sequence
        position_tensors = torch.cat([position_tensors, next_pos], dim=1)
        print(original[0, 2, :])
        print(labels[0, 1, :])

        if step % 20 == 0:
            print(f"Generated {step}/{total_length}")

    # drop the BOS token
    notes = position_tensors[:, 1:, :].squeeze(0).cpu().numpy()
    return notes


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
    all_files = list(data_dir.glob("**/*.mid"))
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
    print(batch['position_tensors'])

    generated_positions = generate_music(model, batch, total_length=args.length, device=args.device)

    # Convert to MIDI
    positions_to_midi(generated_positions, args.output)

    print(f"Music generation complete! Check {args.output}")


if __name__ == "__main__":
    main()