from pathlib import Path
import os
from typing import List, Dict, Any, Tuple

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from symusic.core import ScoreSecond
from torch.utils.data import DataLoader, Dataset
import lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import symusic
from sklearn.model_selection import train_test_split

from src.utils import CONTEXT_SIZE, merge_score_tracks, handle_tempos, handle_key_sigs, handle_time_sigs
from src.model.model import MidiQwenNew

EPOCHS = 24
BATCH_SIZE = 32
MAX_SEQ_LEN = CONTEXT_SIZE


def _create_position_tensors(notes, score: ScoreSecond) -> torch.Tensor:
    """Create 4D position tensors [start_time, duration, pitch, velocity]."""
    positions = []

    # BOS token: start_time=0, duration=0, pitch=0, velocity=0
    positions.append([0.0, 0.0, 0, 0])

    # Process each note
    for note in notes:
        start_time = float(note.start)
        end_time = float(note.end)
        duration = end_time - start_time
        pitch = int(note.pitch)
        velocity = int(note.velocity)

        positions.append([start_time, duration, pitch, velocity])

    # EOS token: start_time = last note end time, duration=0, pitch=1, velocity=0
    if notes:
        last_end_time = float(notes[-1].end)
    else:
        last_end_time = 0.0
    positions.append([last_end_time, 0.0, 1, 0])

    return torch.tensor(positions, dtype=torch.float32)


def create_rope_targets(position_tensors: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    """
    Create RoPE-rotated targets from 4D position tensors.

    For base vector [1, 0, 1, 0, ...], RoPE rotation gives [cos(f1), sin(f1), cos(f2), sin(f2), ...]

    Args:
        position_tensors: (seq_len, 4) tensor of [start_time, duration, pitch, velocity]
        head_dim: dimension of attention heads

    Returns:
        torch.Tensor: (seq_len, head_dim) rotated representations
    """
    seq_len = position_tensors.shape[0]
    device = position_tensors.device
    theta = 10000

    # Create frequency bands (same calculation as rot_pos_emb)
    freq_bands = 1.0 / (theta ** (torch.arange(0, head_dim // 4, 2, device=device).float() / head_dim))

    frequencies = []
    for d in range(4):  # For each of the 4 position dimensions
        # Get positions for this dimension: (seq_len, 1)
        pos_d = position_tensors[:, d:d + 1]

        # Apply frequency bands to this position dimension
        dim_frequencies = pos_d * freq_bands  # (seq_len, head_dim//8)

        # Duplicate each frequency to create pairs (matching RoPE paper)
        dim_frequencies = dim_frequencies.repeat_interleave(2, dim=-1)  # (seq_len, head_dim//4)

        frequencies.append(dim_frequencies)

    # Concatenate frequencies from all 4 dimensions
    all_freqs = torch.cat(frequencies, dim=-1)  # (seq_len, head_dim)

    # For base vector [1, 0, 1, 0, ...], RoPE gives [cos, sin, cos, sin, ...]
    # So we just need to interleave cos and sin values
    cos_vals = all_freqs.cos()  # (seq_len, head_dim)
    sin_vals = all_freqs.sin()  # (seq_len, head_dim)

    # Create the rotated representation: [cos(f1), sin(f1), cos(f2), sin(f2), ...]
    rotated = torch.zeros_like(all_freqs)
    rotated[:, 0::2] = cos_vals[:, 0::2]  # Even indices get cos
    rotated[:, 1::2] = sin_vals[:, 1::2]  # Odd indices get sin

    return rotated


class MidiDataset4D(Dataset):
    """Dataset that concatenates all MIDI files and chunks for pretraining."""

    def __init__(self, files: List[Path], max_seq_len: int = CONTEXT_SIZE):
        self.files = files
        self.max_seq_len = max_seq_len
        self.chunks = []

        # Load all files and create one big concatenated sequence
        print("Loading and concatenating all MIDI files...")
        all_position_tensors = self._load_and_concatenate_files()

        # Chunk the concatenated sequence
        print(f"Chunking into sequences of length {max_seq_len}...")
        self._create_chunks(all_position_tensors)

        print(f"Created {len(self.chunks)} chunks for training")

    def _load_and_concatenate_files(self):
        """Load all MIDI files and concatenate into one big sequence."""
        all_tensors = []

        for file_path in self.files:
            try:
                # Load MIDI file using symusic
                score = symusic.Score.from_file(str(file_path))

                # Use preprocessing functions to clean up the score (in tick format)
                merge_score_tracks(score)

                # Convert to seconds after preprocessing
                score = score.to("second")

                # Extract notes from the merged track
                if not score.tracks or len(score.tracks[0].notes) == 0:
                    continue

                track = score.tracks[0]
                all_notes = list(track.notes)

                # Skip very short pieces
                if len(all_notes) < 5:
                    continue

                # Sort notes by start time
                all_notes.sort(key=lambda x: x.start)

                # Create 4D position tensors [start_time, duration, pitch, velocity]
                position_tensors = _create_position_tensors(all_notes, score)

                # Add this piece to the concatenated sequence
                all_tensors.append(position_tensors)

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                continue

        # Concatenate all pieces
        if all_tensors:
            concatenated = torch.cat(all_tensors, dim=0)
            print(f"Concatenated {len(all_tensors)} files into {concatenated.shape[0]} total vectors")
            return concatenated
        else:
            return torch.tensor([], dtype=torch.float32).reshape(0, 4)

    def _create_chunks(self, all_position_tensors):
        """Split concatenated sequence into fixed-size chunks."""
        total_len = all_position_tensors.shape[0]

        for i in range(0, total_len, self.max_seq_len):
            print(total_len // self.max_seq_len)
            print(i // self.max_seq_len)
            end_idx = min(i + self.max_seq_len, total_len)
            chunk = all_position_tensors[i:end_idx]
            original_len = chunk.shape[0]

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)

            # Pad chunk to max_seq_len if it's the last chunk and shorter
            if chunk.shape[0] < self.max_seq_len:
                pad_len = self.max_seq_len - chunk.shape[0]
                last_time = chunk[-1, 0].item()
                pad_tensor = torch.tensor([last_time, 0.0, 2, 0]).repeat(pad_len, 1)
                chunk = torch.cat([chunk, pad_tensor], dim=0)

                # Mask the padded positions
                attention_mask[original_len:] = 0

            # Create input_ids (all zeros since vocab_size = 1)
            input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)

            # Create RoPE-rotated targets instead of raw 4D vectors
            rope_targets = create_rope_targets(chunk)  # (seq_len, head_dim=128)

            # Create labels (next RoPE target prediction)
            labels = rope_targets[1:].clone()  # Next rope target prediction
            # Pad labels to same length as chunk
            last_target = rope_targets[-1:].clone()
            labels = torch.cat([labels, last_target], dim=0)

            # Set padded label positions to -100 (ignore in loss)
            if original_len < self.max_seq_len:
                labels[original_len:] = -100  # -1 because labels are shifted

            self.chunks.append({
                'input_ids': input_ids,
                'position_tensors': chunk,
                'labels': labels,
                'attention_mask': attention_mask
            })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def custom_collate_fn(batch):
    """Custom collate function for chunked 4D positional data."""
    input_ids_batch = torch.stack([item['input_ids'] for item in batch])
    position_tensors_batch = torch.stack([item['position_tensors'] for item in batch])
    labels_batch = torch.stack([item['labels'] for item in batch])
    attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])

    return {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch,
        'labels': labels_batch,
        'position_tensors': position_tensors_batch
    }


def main():
    # Setup paths
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data" / "aria-midi-v1-unique-ext" / "data"

    # Get all MIDI files
    all_files = list(data_dir.glob("**/*.mid"))[:10000]
    print(f"Found {len(all_files)} MIDI files")

    # Split into train/val
    train_files, val_files = train_test_split(all_files, test_size=0.05, random_state=42)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create datasets (no tokenizer needed)
    print("Creating train dataset...")
    train_dataset = MidiDataset4D(train_files, max_seq_len=MAX_SEQ_LEN)  # Start with subset

    print("Creating val dataset...")
    val_dataset = MidiDataset4D(val_files, max_seq_len=MAX_SEQ_LEN)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=0
    )

    # Setup logging and checkpoints
    wandb_logger = WandbLogger(project="symbolic-music-4d", log_model=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="qwen-4d-{epoch:02d}-{val_loss:.4f}",
        monitor='val_loss',
        save_top_k=4,
        save_last=True,
    )

    # Create dummy tokenizer object for MidiQwenNew compatibility
    class DummyTokenizer:
        pad_token_id = 0

        def __getitem__(self, key):
            return 0

    dummy_tokenizer = DummyTokenizer()

    # Create model
    model = MidiQwenNew(dummy_tokenizer, train_loader, lr=3e-4, warmup_steps=1000)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=4,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        val_check_interval=0.25,  # Validate 4 times per epoch
        precision="bf16-mixed",
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()