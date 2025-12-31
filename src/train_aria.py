import tempfile
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
from concurrent.futures import ProcessPoolExecutor, as_completed

EPOCHS = 6

device = "cuda"
torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(device)


# worker_globals.py (or same file, top-level)
_worker_tokenizer = None
_worker_max_seq_len = None

def worker_init(tokenizer_cfg, max_seq_len):
    global _worker_tokenizer, _worker_max_seq_len

    # Construct tokenizer ONCE per worker
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        "loubb/aria-medium-base",
        trust_remote_code=True,
        add_eos_token=True,
        add_dim_token=False
    )
    _worker_max_seq_len = max_seq_len

def process_pair(melody_file, harmony_file):
    global _worker_tokenizer, _worker_max_seq_len

    try:
        melody_score = symusic.Score.from_file(str(melody_file))
        harmony_score = symusic.Score.from_file(str(harmony_file))

        merge_score_tracks(melody_score)
        merge_score_tracks(harmony_score)

        for t in melody_score.tracks:
            t.program = 0
        for t in harmony_score.tracks:
            t.program = 0

        def first_n_pitches(score, n=10):
            if not score.tracks:
                return []
            notes = sorted(score.tracks[0].notes, key=lambda x: x.start)
            return [n.pitch for n in notes[:n]]

        if first_n_pitches(melody_score) == first_n_pitches(harmony_score):
            return None

        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as m1, \
                tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as m2:
            melody_score.dump_midi(m1.name)
            harmony_score.dump_midi(m2.name)

            melody_dict = MidiDict.from_midi(m1.name)
            harmony_dict = MidiDict.from_midi(m2.name)

        os.unlink(m1.name)
        os.unlink(m2.name)

        mel_ids = _worker_tokenizer._tokenizer.encode(
            _worker_tokenizer.tokenize(
                melody_dict, add_eos_token=True, add_dim_token=False
            )
        )

        har_ids = _worker_tokenizer._tokenizer.encode(
            _worker_tokenizer.tokenize(
                harmony_dict, add_eos_token=True, add_dim_token=False
            )
        )

        combined = mel_ids + har_ids
        if len(combined) > _worker_max_seq_len:
            return None

        if len(combined) < 20:
            return None

        return {
            "input_ids": combined,
            "melody_length": len(mel_ids),
        }

    except Exception as e:
        return ("error", melody_file.name, str(e))


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

    def _load_and_tokenize_pairs(self, num_workers=None):
        num_workers = num_workers or os.cpu_count()
        self.sequences = []

        tokenizer_cfg = {}

        with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=worker_init,
                initargs=(tokenizer_cfg, self.max_seq_len),
        ) as executor:

            futures = [
                executor.submit(process_pair, m, h)
                for m, h in zip(self.melody_files, self.harmony_files)
            ]
            count = 0
            for fut in as_completed(futures):
                print(count)
                count += 1
                result = fut.result()

                if result is None:
                    continue

                if isinstance(result, tuple) and result[0] == "error":
                    _, name, err = result
                    print(f"Failed to process {name}: {err}")
                    continue

                self.sequences.append(result)

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

    # 2. Mask PAD tokens
    labels[padded_input_ids == pad_token_id] = -100

    attention_mask = (padded_input_ids != pad_token_id).long()

    return {
        "input_ids": padded_input_ids,
        "labels": labels,
        # "attention_mask": attention_mask,
    }


class StyleDataset(Dataset):
    def __init__(self, midi_files: List[Path], tokenizer, max_seq_len: int = 8192, use_style_token: bool = True):
        self.midi_files = midi_files
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.use_style_token = use_style_token
        self.sequences = []

        self._load_and_tokenize()

    def _load_and_tokenize(self):
        for midi_file in self.midi_files:
            try:
                # Load and merge tracks
                score = symusic.Score.from_file(str(midi_file))
                merge_score_tracks(score)

                # Set all tracks to piano (program 0)
                for track in score.tracks:
                    track.program = 0

                # Dump to temp MIDI file
                with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                    score.dump_midi(tmp.name)
                    midi_dict = MidiDict.from_midi(tmp.name)
                os.unlink(tmp.name)

                # Tokenize with EOS + DIM tokens
                midi_tokens = self.tokenizer.tokenize(
                    midi_dict, add_eos_token=True, add_dim_token=True
                )
                midi_token_ids = self.tokenizer._tokenizer.encode(midi_tokens)

                # Optional style token
                if self.use_style_token:
                    style_token_id = self.tokenizer._tokenizer.encode("<style:chopin>")
                    midi_token_ids = style_token_id + midi_token_ids

                # ✅ Truncate instead of skipping
                if len(midi_token_ids) > self.max_seq_len:
                    midi_token_ids = midi_token_ids[:self.max_seq_len]

                # Store only meaningful sequences
                if len(midi_token_ids) > 50:
                    self.sequences.append({
                        "input_ids": midi_token_ids
                    })

            except Exception as e:
                print(f"Failed to process {midi_file.name}: {e}")
                continue

        print(f"Loaded {len(self.sequences)} Chopin sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def style_collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )

    # ✅ Standard LM objective: labels = input_ids shifted by 1
    labels = padded_input_ids.clone()

    return {
        "input_ids": padded_input_ids,
        "labels": labels
    }


def train_seq2seq():
    # tokenizer = get_tokenizer(version="v2")

    tokenizer = AutoTokenizer.from_pretrained(
        "loubb/aria-medium-base",
        trust_remote_code=True,
        add_eos_token=True,
        add_dim_token=True
    )
    tokenizer.preprocess_score = lambda x: x

    project_dir = Path(__file__).resolve().parents[1]

    # Load paired datasets - melody files and harmony files
    # melody_train_files = sorted((project_dir / 'data' / 'mel').glob("**/*.mid"))
    # harmony_train_files = sorted((project_dir / 'data' / 'merged').glob("**/*.mid"))

    # print("melody_train_files (first 10):")
    # for f in melody_train_files[:10]:
    #     print(f)
    #
    # print("\nharmony_train_files (first 10):")
    # for f in harmony_train_files[:10]:
    #     print(f)
    # # Split into train/val (95/5 split)
    # split_idx = int(len(melody_train_files) * 0.95)

    melody_train = sorted((project_dir / 'data' / 'new' /  'mel').glob("**/*.mid"))
    harmony_train = sorted((project_dir / 'data' / 'new' / 'merged').glob("**/*.mid"))

    melody_val = sorted((project_dir / 'data' / 'new' / 'mel_val').glob("**/*.mid"))
    harmony_val = sorted((project_dir / 'data' / 'new' / 'merged_val').glob("**/*.mid"))

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
    steps_per_epoch = len(train_loader)
    steps_per_half_epoch = steps_per_epoch // 10

    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="aria-harmony-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=6,
        save_last=True,
        save_weights_only=True,
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
        val_check_interval=300,
    )

    trainer.fit(model, train_loader, val_loader)

    print("Done")


def train_style():
    # tokenizer = get_tokenizer(version="v2")

    tokenizer = AutoTokenizer.from_pretrained(
        "loubb/aria-medium-base",
        trust_remote_code=True,
        add_eos_token=True,
        add_dim_token=False
    )
    tokenizer.preprocess_score = lambda x: x

    project_dir = Path(__file__).resolve().parents[1]

    # Chopin dataset
    chopin_files = sorted((project_dir / 'data' / 'chopin').glob("**/*.mid"))

    train_dataset = StyleDataset(
        midi_files=chopin_files[:int(0.8 * len(chopin_files))],
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE
    )
    val_dataset = StyleDataset(
        midi_files=chopin_files[int(0.8 * len(chopin_files)):],
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=lambda b: chopin_collate_fn(b, tokenizer.pad_token_id),
        num_workers=10,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        collate_fn=lambda b: chopin_collate_fn(b, tokenizer.pad_token_id),
        num_workers=10
    )

    # === WANDB LOGGER ===
    wandb_logger = WandbLogger(project="symbolic-music-generation", log_model=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="aria-style-{epoch:02d}-{val_loss:.4f}",
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

if __name__ == "__main__":
    train_seq2seq()