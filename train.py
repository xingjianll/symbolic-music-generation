from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from miditok import TokenizerConfig, REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import lightning as pl
from utils import CONTEXT_SIZE
import utils
from model import MidiGPT2, MidiQwen


EPOCHS = 32

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    midi_dir = project_dir.joinpath("data/midi")
    midi_paths = list(midi_dir.glob("**/*.mid"))
    dataset_chunks_dir = project_dir.joinpath("data/chunks")

    tokenizer = utils.get_tokenizer()

    all_midi = list(dataset_chunks_dir.glob("**/*.mid"))
    train_files, val_files = train_test_split(all_midi, test_size=0.1, random_state=42)

    # --- TRAIN DATASET ---
    train_dataset = DatasetMIDI(
        files_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=utils.CONTEXT_SIZE,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    train_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=train_collator, num_workers=19)

    # --- VAL DATASET ---
    val_dataset = DatasetMIDI(
        files_paths=val_files,
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    val_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=val_collator, num_workers=19)

    # === WANDB LOGGER ===
    wandb_logger = WandbLogger(project="symbolic-music-generation", log_model=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=project_dir / "checkpoints",
        filename="gpt2-midi-{epoch:02d}-{train_loss:.4f}",
        monitor='train_loss',
        every_n_epochs=1,
        save_top_k=2,
        save_last=True,
    )

    # === TRAIN ===
    model = MidiGPT2(tokenizer, train_loader)
    model.load_checkpoint_expanding_pos_emb("checkpoints/a.ckpt")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        accelerator="auto",
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

    print("Done")