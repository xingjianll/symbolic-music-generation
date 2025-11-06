from pathlib import Path
import os
from typing import List, Dict, Any, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

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
BATCH_SIZE = 8  # Adjust based on GPU memory
MAX_SEQ_LEN = CONTEXT_SIZE


def _create_position_tensors(notes, score: ScoreSecond) -> torch.Tensor:
    """Create 4D position tensors [start_time, duration, pitch, velocity] efficiently."""
    num_notes = len(notes)
    
    if num_notes == 0:
        # Only BOS and EOS tokens
        return torch.tensor([[0.0, 0.0, 0, 0], [0.0, 0.0, 1, 0]], dtype=torch.float32)
    
    # Preallocate tensor: BOS + notes + EOS
    positions = torch.zeros(num_notes + 2, 4, dtype=torch.float32)
    
    # BOS token at index 0: [0, 0, 0, 0] (already zeros)
    
    # Vectorized note processing
    start_times = torch.tensor([note.start for note in notes], dtype=torch.float32)
    end_times = torch.tensor([note.end for note in notes], dtype=torch.float32)
    durations = end_times - start_times
    pitches = torch.tensor([note.pitch for note in notes], dtype=torch.float32)
    velocities = torch.tensor([note.velocity for note in notes], dtype=torch.float32)
    
    # Fill note positions (indices 1 to num_notes)
    positions[1:num_notes+1, 0] = start_times
    positions[1:num_notes+1, 1] = durations  
    positions[1:num_notes+1, 2] = pitches
    positions[1:num_notes+1, 3] = velocities
    
    # EOS token at last index
    last_end_time = end_times[-1] if num_notes > 0 else 0.0
    positions[-1] = torch.tensor([last_end_time, 0.0, 1, 0], dtype=torch.float32)
    
    return positions


def create_rope_targets(position_tensors: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    """
    Create RoPE-rotated targets from 4D position tensors, vectorized like rot_pos_emb.
    
    Args:
        position_tensors: (seq_len, 4) tensor of [start_time, duration, pitch, velocity]
        head_dim: dimension of attention heads
        
    Returns:
        torch.Tensor: (seq_len, head_dim) rotated representations
    """
    from src.model.modeling import apply_rotary_pos_emb
    
    seq_len = position_tensors.shape[0]
    device = position_tensors.device
    theta = 10000
    
    # Create base vector: [1, 0, 1, 0, ...] pattern for all positions
    base_pattern = torch.tensor([1.0, 0.0], device=device)
    base_vector = base_pattern.repeat(head_dim // 2).unsqueeze(0).repeat(seq_len, 1)  # (seq_len, head_dim)
    
    # Create frequency bands (same calculation as rot_pos_emb)
    freq_bands = 1.0 / (theta ** (torch.arange(0, head_dim // 4, 2, device=device).float() / head_dim))
    
    frequencies = []
    for d in range(4):  # For each of the 4 position dimensions
        # Get positions for this dimension: (seq_len, 1)
        pos_d = position_tensors[:, d:d+1]
        
        # Apply frequency bands to this position dimension
        dim_frequencies = pos_d * freq_bands  # (seq_len, head_dim//8)
        
        # Duplicate each frequency to create pairs (matching RoPE paper)
        dim_frequencies = dim_frequencies.repeat_interleave(2, dim=-1)  # (seq_len, head_dim//4)
        
        frequencies.append(dim_frequencies)
    
    # Concatenate frequencies from all 4 dimensions
    all_freqs = torch.cat(frequencies, dim=-1)  # (seq_len, head_dim)
    
    # Convert frequencies to cos/sin (same as in Qwen3Model forward)
    cos_vals = all_freqs.cos()  # (seq_len, head_dim)
    sin_vals = all_freqs.sin()  # (seq_len, head_dim)
    
    # Reshape for apply_rotary_pos_emb: (1, 1, seq_len, head_dim)
    q = base_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    k = q.clone()  # dummy key
    
    # Apply RoPE rotation using existing function
    rotated_q, _ = apply_rotary_pos_emb(q, k, cos_vals, sin_vals, unsqueeze_dim=1)
    
    # Extract the rotated vectors: (1, 1, seq_len, head_dim) -> (seq_len, head_dim)
    return rotated_q.squeeze(0).squeeze(0)


def process_single_file(file_path: Path) -> np.ndarray:
    """Process a single MIDI file and return position tensors as numpy array. For multiprocessing."""
    try:
        # Load MIDI file using symusic
        score = symusic.Score.from_file(str(file_path))

        # Use preprocessing functions to clean up the score (in tick format)
        merge_score_tracks(score)
        
        # Convert to seconds after preprocessing
        score = score.to("second")
        
        # Extract notes from the merged track
        if not score.tracks or len(score.tracks[0].notes) == 0:
            return None
            
        track = score.tracks[0]
        all_notes = list(track.notes)
        
        # Skip very short pieces
        if len(all_notes) < 5:
            return None
        
        # Sort notes by start time
        all_notes.sort(key=lambda x: x.start)
        
        # Create 4D position tensors [start_time, duration, pitch, velocity]
        position_tensors = _create_position_tensors(all_notes, score)
        
        # Convert to numpy for safe multiprocessing serialization
        return position_tensors.numpy()
        
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None


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
        """Load all MIDI files and concatenate into one big sequence using parallel processing."""
        # Limit processes to avoid overwhelming the system
        max_processes = min(cpu_count(), 16)  # Cap at 16 processes
        print(f"Processing {len(self.files)} files using {max_processes} processes...")
        
        # Use multiprocessing to process files in parallel
        with Pool(processes=max_processes) as pool:
            results = pool.map(process_single_file, self.files)
        
        # Filter out None results (failed files) and convert back to tensors
        all_arrays = [array for array in results if array is not None]
        
        # Convert numpy arrays back to tensors and concatenate
        if all_arrays:
            all_tensors = [torch.from_numpy(array) for array in all_arrays]
            concatenated = torch.cat(all_tensors, dim=0)
            print(f"Successfully processed {len(all_tensors)}/{len(self.files)} files")
            print(f"Concatenated into {concatenated.shape[0]} total vectors")
            return concatenated
        else:
            print("No files were successfully processed!")
            return torch.tensor([], dtype=torch.float32).reshape(0, 4)
    
    def _create_chunks(self, all_position_tensors):
        """Split concatenated sequence into fixed-size chunks."""
        total_len = all_position_tensors.shape[0]
        
        # Use GPU if available for faster processing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_position_tensors = all_position_tensors.to(device)
        print(f"Using {device} for RoPE computation...")
        
        # Check if last chunk needs padding
        remainder = total_len % self.max_seq_len
        needs_padding = remainder != 0
        
        # Add padding to original tensor if needed
        if needs_padding:
            pad_len = self.max_seq_len - remainder
            last_time = all_position_tensors[-1, 0].item()
            pad_token = torch.tensor([last_time, 0.0, 2, 0], device=device)
            all_position_tensors = torch.cat([all_position_tensors, pad_token.repeat(pad_len, 1)])
        
        # Split into chunks and process in batches to avoid memory issues
        chunks = all_position_tensors.view(-1, self.max_seq_len, 4)
        num_chunks = chunks.shape[0]
        
        # Process RoPE targets in batches to avoid memory explosion
        batch_size = 20  # Process 20 chunks at a time
        all_rope_targets = []
        
        print(f"Processing {num_chunks} chunks in batches of {batch_size}...")
        for i in range(0, num_chunks, batch_size):
            print(i)
            end_idx = min(i + batch_size, num_chunks)
            chunk_batch = chunks[i:end_idx]  # (batch_size, seq_len, 4)
            
            # Flatten for RoPE processing: (batch_size * seq_len, 4)
            flattened = chunk_batch.view(-1, 4)
            rope_batch = create_rope_targets(flattened)  # (batch_size * seq_len, head_dim)
            
            # Reshape back to chunks: (batch_size, seq_len, head_dim)
            rope_chunked = rope_batch.view(end_idx - i, self.max_seq_len, -1)
            all_rope_targets.append(rope_chunked)
        
        # Concatenate all batches
        rope_targets_chunked = torch.cat(all_rope_targets, dim=0)  # (num_chunks, seq_len, head_dim)
        
        # Vectorized creation of labels, masks, input_ids
        labels_batch = torch.cat([rope_targets_chunked[:, 1:], rope_targets_chunked[:, -1:]], dim=1)
        attention_masks = torch.ones(num_chunks, self.max_seq_len, dtype=torch.long, device=device)
        input_ids_batch = torch.zeros(num_chunks, self.max_seq_len, dtype=torch.long, device=device)
        
        # Handle padding mask for last chunk only
        if needs_padding:
            attention_masks[-1, remainder:] = 0
            labels_batch[-1, remainder:] = -100
        
        # Move back to CPU for dataset storage (datasets are typically CPU-based)
        chunks = chunks.cpu()
        labels_batch = labels_batch.cpu()
        attention_masks = attention_masks.cpu()
        input_ids_batch = input_ids_batch.cpu()
        
        # Convert to list of dictionaries
        self.chunks = []
        for i in range(num_chunks):
            self.chunks.append({
                'input_ids': input_ids_batch[i],
                'position_tensors': chunks[i],
                'labels': labels_batch[i],
                'attention_mask': attention_masks[i]
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
    all_files = list(data_dir.glob("**/*.mid"))
    print(f"Found {len(all_files)} MIDI files")
    
    # Split into train/val
    train_files, val_files = train_test_split(all_files, test_size=0.1, random_state=42)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets (no tokenizer needed)
    print("Creating train dataset...")
    train_dataset = MidiDataset4D(train_files[:1000], max_seq_len=MAX_SEQ_LEN)  # Start with subset
    
    print("Creating val dataset...")
    val_dataset = MidiDataset4D(train_files[:100], max_seq_len=MAX_SEQ_LEN)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
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
        accumulate_grad_batches=4,  # Effective batch size = 8 * 4 = 32
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    print("Training complete!")


if __name__ == "__main__":
    main()