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

import numpy as np
import mido
from mido import Message, MidiFile, MidiTrack, bpm2tempo

def notes_to_midi(notes_array, output_path, tempo=120):
    """
    Convert a numpy array of [start_sec, duration_sec, pitch, velocity]
    into a MIDI file using mido (supports overlapping same-pitch notes).
    """

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    track.append(mido.MetaMessage('set_tempo', tempo=bpm2tempo(tempo)))

    # Convert seconds to ticks
    ticks_per_beat = mid.ticks_per_beat
    sec_to_ticks = (tempo / 60.0) * ticks_per_beat

    events = []

    for start, duration, pitch, velocity in notes_array:

        start_ticks = int(start * sec_to_ticks)
        end_ticks = int((start + duration) * sec_to_ticks)

        pitch = int(np.clip(pitch, 0, 127))
        velocity = int(np.clip(velocity, 0, 127))

        # Add separate ON and OFF events
        events.append((start_ticks,  Message('note_on',  note=pitch, velocity=velocity)))
        events.append((end_ticks,    Message('note_off', note=pitch, velocity=0)))

    # Sort all events by time
    events.sort(key=lambda x: x[0])

    # Insert with delta times
    last_time = 0
    for abs_time, msg in events:
        delta = abs_time - last_time
        msg.time = delta
        last_time = abs_time
        track.append(msg)

    mid.save(output_path)
    print(f"MIDI saved to {output_path}")




def f(v, c):
    return v//c*c

def f2(v, c):
    return (v / c).round() * c

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

    feature0 = f2(pos3, 8) + f2(pos2, 1) + f2(pos1, 0.25) + pos0
    feature1 = f2(pos7, 8) + f2(pos6, 1) + f2(pos5, 0.25) + pos4
    feature2 = f2(pos11, 64) + f2(pos10, 16) + f2(pos9, 4) + pos8
    feature3 = f2(pos15, 64) + f2(pos14, 16) + f2(pos13, 4) + pos12
    next_pos = torch.stack([feature0, feature1, feature2.round(), feature3.round()]).to(device)
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

    generated = batch['position_tensors'][0:1, 0:1,:]
    for i in range(total_length):
        with torch.no_grad():
            out = model(
                input_ids=None,
                position_tensors=generated,
            )
            next_pos = process_last_logits(out.logits[0,i,:,:])
            time = generated[0, -1, 0]
            next_pos[0, 0, 0] += time
            print(next_pos[0, 0, :])
            print(batch['position_tensors'][0, i+1, :])
            print("----------")
            generated = torch.cat([generated, next_pos], dim=1)

    return generated[0]


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
    data_dir = project_dir / "data" / "aria-midi-v1-deduped-ext" / "data"

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
    notes_to_midi(generated_positions.cpu().numpy(), args.output)

    print(f"Music generation complete! Check {args.output}")


if __name__ == "__main__":
    main()