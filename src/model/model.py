import lightning as pl
from peft import LoraConfig, get_peft_model, TaskType
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoConfig
import torch

from src.utils import CONTEXT_SIZE


# Copied from https://github.com/EleutherAI/aria/blob/main/aria/training/train.py
def _get_optim(
    lr: float,
    model: nn.Module,
    num_epochs: int,
    steps_per_epoch: int,
    warmup: int = 100,
    end_ratio: float = 0.1,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-5,
    )

    warmup_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.000001,
        end_factor=1,
        total_iters=warmup,
    )
    linear_decay_lrs = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=end_ratio,
        total_iters=(num_epochs * steps_per_epoch) - warmup,
    )

    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lrs, linear_decay_lrs],
        milestones=[warmup],
    )

    return optimizer, lr_scheduler


class MidiGPT2(pl.LightningModule):
    def __init__(self, tokenizer, dataloader, lr=3e-4, warmup_steps=1000):
        super().__init__()
        self.save_hyperparameters()

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=CONTEXT_SIZE,
            n_ctx=CONTEXT_SIZE,
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
        from src.train import EPOCHS

        steps_per_epoch = len(self.dataloader)
        optimizer, scheduler = _get_optim(
            lr=self.lr,
            model=self,
            num_epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            warmup=self.warmup_steps,
            end_ratio=0.1,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def load_checkpoint_expanding_pos_emb(self, checkpoint_path):
        """Load checkpoint and expand positional embeddings if needed"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        model_pos_emb = self.model.transformer.wpe.weight  # shape: [8192, dim]
        old_pos_emb = state_dict["model.transformer.wpe.weight"]  # from checkpoint

        if old_pos_emb.shape[0] < model_pos_emb.shape[0]:
            print(f"Expanding position embeddings: {old_pos_emb.shape[0]} → {model_pos_emb.shape[0]}")
            model_pos_emb.data[:old_pos_emb.shape[0]] = old_pos_emb
            state_dict["model.transformer.wpe.weight"] = model_pos_emb
        else:
            print(f"Loading position embeddings without resizing.")

        self.load_state_dict(state_dict, strict=False)



class MidiQwen(pl.LightningModule):
    def __init__(self, tokenizer, dataloader, lr=5e-5, warmup_steps=1000):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        config.hidden_size = 512  # 1024
        config.num_hidden_layers = 14  # 28
        config.num_attention_heads = 8  # 16
        config.num_key_value_heads = 8
        config.intermediate_size = 2048  # 3072
        config.max_position_embeddings = CONTEXT_SIZE
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
        from src.train import EPOCHS

        steps_per_epoch = len(self.dataloader)
        optimizer, scheduler = _get_optim(
            lr=self.lr,
            model=self,
            num_epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            warmup=self.warmup_steps,
            end_ratio=0.1,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def load_checkpoint_expanding_pos_emb(self, checkpoint_path):
        """Load checkpoint and expand positional embeddings if needed"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        self.load_state_dict(state_dict, strict=False)


import lightning as pl
import torch



class MidiAria(pl.LightningModule):
    def __init__(self, tokenizer, dataloader, lr=5e-5, warmup_steps=1000):
        super().__init__()
        # self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.dataloader = dataloader

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        # Handle both tuple and object returns from PEFT models
        if isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        # Handle both tuple and object returns from PEFT models
        if isinstance(outputs, tuple):
            val_loss = outputs[0]
        else:
            val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        from src.train import EPOCHS

        steps_per_epoch = len(self.dataloader)
        optimizer, scheduler = _get_optim(
            lr=self.lr,
            model=self,
            num_epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            warmup=self.warmup_steps,
            end_ratio=0.1,
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def load_checkpoint_expanding_pos_emb(self, checkpoint_path):
        """Load checkpoint and expand position embeddings if needed."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]

        model_pos_emb = self.model.model.tok_embeddings.weight  # (vocab_size, hidden_dim)
        old_pos_emb = state_dict.get("model.model.tok_embeddings.weight")

        if old_pos_emb is not None and old_pos_emb.shape[0] < model_pos_emb.shape[0]:
            print(f"Expanding embeddings: {old_pos_emb.shape[0]} → {model_pos_emb.shape[0]}")
            model_pos_emb.data[:old_pos_emb.shape[0]] = old_pos_emb
            state_dict["model.model.tok_embeddings.weight"] = model_pos_emb
        else:
            print("Loading embeddings without resizing.")

        self.load_state_dict(state_dict, strict=False)

    def to_lora(self):
        # FREEZE WEIGHTS
        for param in self.model.parameters():
            param.requires_grad = False

        # LoRa
        config = LoraConfig(
            r=128,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules = [
                "mixed_qkv",
                "att_proj_linear",
                "ff_gate_proj",
                "ff_up_proj",
                "ff_down_proj"
            ]
        )
        self.model = get_peft_model(self.model, config)
