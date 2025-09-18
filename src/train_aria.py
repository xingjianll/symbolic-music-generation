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

from src.tokenizer import get_tokenizer
from src.utils import CONTEXT_SIZE
import src.utils
from src.model.model import MidiGPT2, MidiQwen, MidiAria

EPOCHS = 12

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)



class MidiDataset(Dataset):
    def __init__(self, midi_files: List[Path], tokenizer, max_seq_len: int = 8192):
        self.midi_files = midi_files
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sequences = []

        # Process all MIDI files and tokenize them
        self._load_and_tokenize()

    def _load_and_tokenize(self):
        for midi_file in self.midi_files:
            try:
                # Load MIDI file using Aria's MidiDict
                midi_dict = MidiDict.from_midi(str(midi_file))

                # Tokenize using Aria's approach
                tokens = self.tokenizer.tokenize(midi_dict, add_eos_token=False, add_dim_token=False)
                tokens = self.tokenizer._tokenizer.encode(tokens)

                # Split into chunks if longer than max_seq_len
                if len(tokens) > self.max_seq_len:
                    for i in range(0, len(tokens), self.max_seq_len):
                        chunk = tokens[i:i + self.max_seq_len]
                        if len(chunk) > 50:  # Only keep meaningful chunks
                            self.sequences.append(chunk)
                else:
                    if len(tokens) > 50:  # Only keep meaningful sequences
                        self.sequences.append(tokens)

            except Exception as e:
                print(f"Failed to process {midi_file}: {e}")
                continue

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {"input_ids": self.sequences[idx]}

def collate_fn(batch, pad_token_id):
    """Custom collate function for Aria tokenizer"""
    input_ids = []
    for item in batch:
        ids = torch.tensor(item["input_ids"], dtype=torch.long)
        input_ids.append(ids)

    # Pad sequences to the same length
    padded = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id
    )

    return {
        "input_ids": padded,
        "labels": padded.clone()
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


    train_files = list((project_dir / 'data' / 'single_track_combined_train').glob("**/*.mid"))
    val_files = list((project_dir / 'data' / 'single_track_combined_val').glob("**/*.mid"))

    # --- TRAIN DATASET ---
    train_dataset = MidiDataset(
        midi_files=train_files,
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE
    )

    # Create collate function with pad_token_id
    def train_collate_fn(batch):
        return collate_fn(batch, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        collate_fn=train_collate_fn,
        num_workers=0,
        shuffle=True
    )

    # --- VAL DATASET ---
    val_dataset = MidiDataset(
        midi_files=val_files,
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
        collate_fn=train_collate_fn,
        num_workers=0
    )

    # === WANDB LOGGER ===
    wandb_logger = WandbLogger(project="symbolic-music-generation", log_model=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="qwen-midi-{epoch:02d}-{val_loss:.4f}",
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
    model.to(device)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        val_check_interval=300,
    )

    trainer.fit(model, train_loader, val_loader)

    print("Done")