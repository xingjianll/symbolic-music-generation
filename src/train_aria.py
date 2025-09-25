from pathlib import Path
import os
from typing import List, Dict, Any

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import lightning as pl
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ariautils.midi import MidiDict
import numpy as np

from src.utils import CONTEXT_SIZE, merge_score_tracks
from src.model.model import MidiAria
import symusic

EPOCHS = 3

device = "cuda"
torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)



class MelodyHarmonizationDataset(Dataset):
    def __init__(self, melody_files: List[Path], harmony_files: List[Path], tokenizer, max_seq_len: int = 8192):
        self.melody_files = melody_files
        self.harmony_files = harmony_files
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences = []

        # Ensure we have matching pairs
        assert len(melody_files) == len(harmony_files), "Melody and harmony file counts must match"

        # Process all MIDI file pairs and tokenize them
        self._load_and_tokenize_pairs()

    def _load_and_tokenize_pairs(self):
        for melody_file, harmony_file in zip(self.melody_files, self.harmony_files):
            try:
                # Load MIDI files using symusic first, then merge tracks
                melody_score = symusic.Score.from_file(str(melody_file))
                harmony_score = symusic.Score.from_file(str(harmony_file))

                # Merge tracks using preprocessing function and set to piano
                merge_score_tracks(melody_score)
                merge_score_tracks(harmony_score)

                # Set all tracks to piano (program 0)
                for track in melody_score.tracks:
                    track.program = 0
                for track in harmony_score.tracks:
                    track.program = 0

                # Convert to MidiDict for tokenization
                # Save to temporary files first (symusic → MIDI → MidiDict)
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_melody, \
                     tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as temp_harmony:

                    melody_score.dump_midi(temp_melody.name)
                    harmony_score.dump_midi(temp_harmony.name)

                    # Load with MidiDict
                    melody_dict = MidiDict.from_midi(temp_melody.name)
                    harmony_dict = MidiDict.from_midi(temp_harmony.name)

                    # Clean up temp files
                    os.unlink(temp_melody.name)
                    os.unlink(temp_harmony.name)

                # Tokenize both - keep eos and dim tokens for training
                melody_tokens = self.tokenizer.tokenize(melody_dict, add_eos_token=True, add_dim_token=True)
                melody_token_ids = self.tokenizer._tokenizer.encode(melody_tokens)

                harmony_tokens = self.tokenizer.tokenize(harmony_dict, add_eos_token=True, add_dim_token=True)
                harmony_token_ids = self.tokenizer._tokenizer.encode(harmony_tokens)

                # Create combined sequence: melody + harmony (no separator needed)
                # The natural <E> <S> boundary between sequences provides the separation
                combined_sequence = melody_token_ids + harmony_token_ids

                # Split into chunks if too long
                if len(combined_sequence) > self.max_seq_len:
                    print(f"Sequence too long ({len(combined_sequence)} tokens), skipping {melody_file.name}")
                    continue

                if len(combined_sequence) > 100:  # Keep meaningful sequences
                    # Store the combined sequence and the melody length for masking
                    self.sequences.append({
                        'input_ids': combined_sequence,
                        'melody_length': len(melody_token_ids)  # No separator token
                    })

            except Exception as e:
                print(f"Failed to process pair {melody_file.name}: {e}")
                continue
        print(f"size of pairs: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch, pad_token_id):
    """Custom collate function for melody harmonization with loss masking"""
    input_ids = []
    melody_lengths = []

    for item in batch:
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        input_ids.append(ids)
        melody_lengths.append(item["melody_length"])

    # Pad sequences to the same length
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )

    # Create labels with masking for melody part (we don't want to train on predicting the prompt)
    labels = padded_input_ids.clone()

    # Mask the melody part + separator token (set to -100 so they're ignored in loss)
    for i, melody_length in enumerate(melody_lengths):
        labels[i, :melody_length] = -100

    return {
        "input_ids": padded_input_ids,
        "labels": labels
    }

if __name__ == "__main__":
    # tokenizer = get_tokenizer(version="v2")

    tokenizer = AutoTokenizer.from_pretrained(
        "loubb/aria-medium-base",
        trust_remote_code=True,
        add_eos_token=True,
        add_dim_token=False
    )
    tokenizer.preprocess_score = lambda x: x

    project_dir = Path(__file__).resolve().parents[1]


    # Load paired datasets - melody files and harmony files
    melody_train_files = sorted((project_dir / 'data' / 'wikifonia_midi_no_chord').glob("**/*.mid"))
    harmony_train_files = sorted((project_dir / 'data' / 'wikifonia_midi').glob("**/*.mid"))

    # Split into train/val (80/20 split)
    split_idx = int(len(melody_train_files) * 0.8)

    melody_train = melody_train_files[:split_idx]
    harmony_train = harmony_train_files[:split_idx]

    melody_val = melody_train_files[split_idx:]
    harmony_val = harmony_train_files[split_idx:]

    print(f"Training pairs: {len(melody_train)}, Validation pairs: {len(melody_val)}")

    # --- TRAIN DATASET ---
    train_dataset = MelodyHarmonizationDataset(
        melody_files=melody_train,
        harmony_files=harmony_train,
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE
    )

    # Create collate function with pad_token_id
    def train_collate_fn(batch):
        return collate_fn(batch, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=train_collate_fn,
        num_workers=10,
        shuffle=True
    )

    # --- VAL DATASET ---
    val_dataset = MelodyHarmonizationDataset(
        melody_files=melody_val,
        harmony_files=harmony_val,
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        collate_fn=train_collate_fn,
        num_workers=10
    )

    # === WANDB LOGGER ===
    wandb_logger = WandbLogger(project="symbolic-music-generation", log_model=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="aria-harmony-{epoch:02d}-{val_loss:.4f}",
        monitor='train_loss',
        every_n_epochs=1,
        save_top_k=8,
        save_last=True,
    )

    # === TRAIN ===
    model = MidiAria(tokenizer, train_loader)
    hf_model = AutoModelForCausalLM.from_pretrained(
        "loubb/aria-medium-base",
        trust_remote_code=True
    )

    model.load_state_dict(hf_model.state_dict(), strict=False)
    model.to_lora()

    # Enable gradient checkpointing to save memory
    # model.model.gradient_checkpointing_enable()

    model.to(device)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        val_check_interval=20,
    )

    trainer.fit(model, train_loader, val_loader)

    print("Done")