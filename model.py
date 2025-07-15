import lightning as pl
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoConfig
import torch

class MidiGPT2(pl.LightningModule):
    def __init__(self, tokenizer, dataloader, lr=5e-5, warmup_steps=500):
        super().__init__()
        self.save_hyperparameters()

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=8192,
            n_ctx=8192,
            n_embd=512,
            n_layer=6,
            n_head=8,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pad_token_id=tokenizer.pad_token_id,
        )
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = tokenizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.dataloader = dataloader

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        from train import EPOCHS

        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=len(self.dataloader) * EPOCHS,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class MidiQwen(pl.LightningModule):
    def __init__(self, tokenizer, dataloader, lr=5e-5, warmup_steps=500):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        config.hidden_size = 512  # 1024
        config.num_hidden_layers = 12  # 28
        config.num_attention_heads = 8  # 16
        config.intermediate_size = 2048  # 3072
        config.max_position_embeddings = 1024
        config.bos_token_id = tokenizer["BOS_None"]
        config.eos_token_id = tokenizer["EOS_None"]
        config.pad_token_id = tokenizer.pad_token_id
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.dataloader = dataloader


    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        from train import EPOCHS

        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=len(self.dataloader) * EPOCHS,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

