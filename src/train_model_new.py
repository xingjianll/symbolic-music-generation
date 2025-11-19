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


class MidiDataset4D(Dataset):
    """Dataset that concatenates all MIDI files and chunks for pretraining."""

    def __init__(self, files: List[Path], max_seq_len: int = CONTEXT_SIZE):
        self.files = files
        self.max_seq_len = max_seq_len
        self.chunks = []

        # Load all files and create one big concatenated sequence
        print("Loading and concatenating all MIDI files...")
        all_position_tensors = self._load_and_concatenate_files(self.files)

        # Chunk the concatenated sequence
        print(f"Chunking into sequences of length {max_seq_len}...")
        self._create_chunks(all_position_tensors)

        print(f"Created {len(self.chunks)} chunks for training")

    def _load_and_concatenate_files(self, file_list: List[Path]):
        """Load all MIDI files and concatenate into one big sequence."""
        all_tensors = []

        for file_path in file_list:
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

            # Create labels (next 4D positions with delta time as first feature)
            labels = chunk[1:].clone()  # Next position prediction
            last_position = chunk[-1:].clone()
            labels = torch.cat([labels, last_position], dim=0)

            # Convert first feature from absolute time to delta time
            labels[0:, 0] = labels[0:, 0] - chunk[0:, 0]  # Subsequent labels: delta from previous position

            # Set padded label positions to -100 (ignore in loss)
            if original_len < self.max_seq_len:
                labels[original_len:] = -100

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



class MidiDataset4DStreaming(MidiDataset4D):
    """Streaming dataset that loads first 100 files, then fetches 2 files per batch."""

    def __init__(self, files: List[Path], max_seq_len: int = CONTEXT_SIZE):
        self.files = files
        self.max_seq_len = max_seq_len
        self.file_idx = 0
        self.current_sequence = torch.tensor([], dtype=torch.float32).reshape(0, 4)
        self.total_chunks_served = 0

        # Load first 100 files
        print("Loading first 100 files into memory...")
        initial_files = files[:100]
        self.current_sequence = self._load_and_concatenate_files(initial_files)
        self.file_idx = 100

        # Calculate total estimated chunks for __len__
        self.estimated_total_chunks = self._estimate_total_chunks()

        print(f"Loaded {len(initial_files)} files, estimated {self.estimated_total_chunks} total chunks")

    def _estimate_total_chunks(self) -> int:
        """Estimate total chunks across all files."""
        # Sample first 50 files to estimate average chunk count
        sample_files = self.files[:50]
        total_notes = 0
        valid_files = 0

        for file_path in sample_files:
            try:
                score = symusic.Score.from_file(str(file_path))
                merge_score_tracks(score)
                if score.tracks and len(score.tracks[0].notes) > 5:
                    total_notes += len(score.tracks[0].notes)
                    valid_files += 1
            except:
                continue

        if valid_files == 0:
            return 1000  # Fallback estimate

        avg_notes_per_file = total_notes / valid_files
        avg_chunks_per_file = (avg_notes_per_file + 2) / self.max_seq_len  # +2 for BOS/EOS
        total_estimated = int(len(self.files) * avg_chunks_per_file)
        return max(total_estimated, 1)

    def _fetch_next_files(self) -> torch.Tensor:
        """Fetch next 2 files and add to current sequence."""
        if self.file_idx >= len(self.files):
            return torch.tensor([], dtype=torch.float32).reshape(0, 4)

        end_idx = min(self.file_idx + 2, len(self.files))
        next_files = self.files[self.file_idx:end_idx]
        self.file_idx = end_idx

        return self._load_and_concatenate_files(next_files)

    def __len__(self):
        return self.estimated_total_chunks

    def __getitem__(self, idx):
        # Check if we need more data
        while self.current_sequence.shape[0] < self.max_seq_len:
            new_data = self._fetch_next_files()
            if new_data.shape[0] == 0:
                break  # No more files
            self.current_sequence = torch.cat([self.current_sequence, new_data], dim=0)

        # Extract chunk
        if self.current_sequence.shape[0] == 0:
            # No data available, return dummy chunk
            chunk = torch.zeros(self.max_seq_len, 4)
            original_len = 0
        else:
            chunk_end = min(self.max_seq_len, self.current_sequence.shape[0])
            chunk = self.current_sequence[:chunk_end]
            original_len = chunk.shape[0]

            # Remove served chunk from sequence
            self.current_sequence = self.current_sequence[chunk_end:]
            self.total_chunks_served += 1

        # Create attention mask
        attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)

        # Only pad if this is truly the last chunk of the epoch
        is_last_chunk = (self.file_idx >= len(self.files) and
                         self.current_sequence.shape[0] < self.max_seq_len)

        if chunk.shape[0] < self.max_seq_len:
            if is_last_chunk:
                # Pad only at very end of epoch
                pad_len = self.max_seq_len - chunk.shape[0]
                last_time = chunk[-1, 0].item() if chunk.shape[0] > 0 else 0.0
                pad_tensor = torch.tensor([last_time, 0.0, 2, 0]).repeat(pad_len, 1)
                chunk = torch.cat([chunk, pad_tensor], dim=0)
                attention_mask[original_len:] = 0
            else:
                # Don't pad, just return shorter chunk and adjust mask
                temp_chunk = torch.zeros(self.max_seq_len, 4)
                temp_chunk[:chunk.shape[0]] = chunk
                chunk = temp_chunk
                attention_mask[original_len:] = 0

        # Create input_ids and labels
        input_ids = torch.zeros(self.max_seq_len, dtype=torch.long)

        # Labels are next 4D positions directly
        labels = chunk[1:].clone()  # Next position prediction
        last_position = chunk[-1:].clone()
        labels = torch.cat([labels, last_position], dim=0)
        labels[0:, 0] = labels[0:, 0] - chunk[0:, 0]

        if original_len < self.max_seq_len:
            labels[original_len:] = -100

        return {
            'input_ids': input_ids,
            'position_tensors': chunk,
            'labels': labels,
            'attention_mask': attention_mask
        }

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
    data_dir = project_dir / "data" / "aria-midi-v1-deduped-ext" / "data"

    # Get all MIDI files
    all_files = list(sorted(data_dir.glob("**/*.mid")))
    print(f"Found {len(all_files)} MIDI files")

    # Split into train/val
    train_files, val_files = train_test_split(all_files, test_size=0.02, random_state=42)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create datasets (no tokenizer needed)
    print("Creating train dataset...")
    train_dataset = MidiDataset4DStreaming(train_files, max_seq_len=MAX_SEQ_LEN)

    print("Creating val dataset...")
    val_dataset = MidiDataset4DStreaming(val_files, max_seq_len=MAX_SEQ_LEN)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=8
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=8
    )
    print("123")

    # Setup logging and checkpoints
    wandb_logger = WandbLogger(project="symbolic-music-4d", log_model=True)
    print("here")

    steps_per_epoch = len(train_loader)
    steps_per_half_epoch = steps_per_epoch // 2
    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="qwen-4d-{epoch:02d}-{step:05d}-{val_loss:.4f}",
        monitor='val_loss',
        save_top_k=5,
        save_last=True,
        mode='min',
        every_n_train_steps=steps_per_half_epoch,
    )

    # Create dummy tokenizer object for MidiQwenNew compatibility
    class DummyTokenizer:
        pad_token_id = 0

        def __getitem__(self, key):
            return 0

    print("here10")
    dummy_tokenizer = DummyTokenizer()
    print("here20")

    # Create model
    model = MidiQwenNew(dummy_tokenizer, train_loader, lr=3e-4, warmup_steps=500)
    print("here30")
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=5.0,
        log_every_n_steps=4,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        val_check_interval=0.1,
        precision="bf16-mixed",
        num_sanity_val_steps=0,
    )
    print("here40")
    print(train_files[:1])

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()