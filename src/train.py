from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from miditok.pytorch_data import DatasetMIDI, DataCollator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import lightning as pl

from src.tokenizer import get_tokenizer
from src.utils import CONTEXT_SIZE
import src.utils
from src.model import MidiGPT2, MidiQwen

EPOCHS = 32

if __name__ == "__main__":
    tokenizer = get_tokenizer(version="v2")

    project_dir = Path(__file__).resolve().parents[1]


    train_files = list((project_dir / 'data' / 'chunks_train').glob("**/*.mid"))
    val_files = list((project_dir / 'data' / 'chunks_val').glob("**/*.mid"))

    # --- TRAIN DATASET ---
    train_dataset = DatasetMIDI(
        files_paths=train_files,
        tokenizer=tokenizer,
        max_seq_len=src.utils.CONTEXT_SIZE,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    train_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=train_collator, num_workers=19)

    # --- VAL DATASET ---
    val_dataset = DatasetMIDI(
        files_paths=val_files,
        tokenizer=tokenizer,
        max_seq_len=CONTEXT_SIZE,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    val_collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=val_collator, num_workers=19)

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
    model = MidiQwen(tokenizer, train_loader)
    # model.load_checkpoint_expanding_pos_emb("checkpoints/a.ckpt")
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        logger=wandb_logger,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        val_check_interval=100
    )

    trainer.fit(model, train_loader, val_loader)

    print("Done")