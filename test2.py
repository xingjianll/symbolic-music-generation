import torch

from src.model.model import MidiQwenNew
from src.train_model_new import MidiDataset4DStreaming, custom_collate_fn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

def load_model_from_checkpoint(checkpoint_path: str, device='cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create dummy tokenizer (same as training)
    class DummyTokenizer:
        pad_token_id = 0
        def __getitem__(self, key):
            return 0

    dummy_tokenizer = DummyTokenizer()
    dummy_dataloader = []  # Not used for inference

    try:
        # Try normal loading first
        model = MidiQwenNew.load_from_checkpoint(
            checkpoint_path,
            tokenizer=dummy_tokenizer,
            dataloader=dummy_dataloader,
            map_location=device
        )
    except Exception as e:
        print(f"Normal loading failed: {e}")
        print("Trying manual state dict loading...")

        # Manual loading as fallback
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create fresh model
        model = MidiQwenNew(dummy_tokenizer, dummy_dataloader)

        # Load only the model weights, ignore other hyperparameters
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.eval()
    model.to(device)

    return model.model


def test_causal_invariance(model, seq_len=10, cut_len=5, num_heads=1, head_dim=64):
    print("\n=== Testing causal invariance ===")

    attn = model.model.layers[0].self_attn
    attention_fn = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]
    print("Using attention:", attn.config._attn_implementation)

    # --------- Generate Q/K/V ----------
    q = torch.randn(1, num_heads, cut_len, head_dim)
    k = torch.randn(1, num_heads, cut_len, head_dim)
    v = torch.randn(1, num_heads, cut_len, head_dim)
    full_mask = torch.zeros(1, 1, cut_len, cut_len)

    # --------- Full attention call ----------
    full_out, _ = attention_fn(
        attn,
        q, k, v,
        attention_mask=full_mask,
        dropout=0.0 if not model.model.layers[0].self_attn.training else model.model.layers[0].self_attn.attention_dropout,
        scaling=model.model.layers[0].self_attn.scaling
    )

    # --------- Truncated attention call ----------
    q_cut = q[:, :, :cut_len, :]
    k_cut = k[:, :, :cut_len, :]
    v_cut = v[:, :, :cut_len, :]
    full_mask = torch.zeros(1, 1, cut_len, cut_len)

    cut_out, _ = attention_fn(
        attn,
        q_cut, k_cut, v_cut,
        attention_mask=None,
        dropout=0.0 if not model.model.layers[0].self_attn.training else model.model.layers[0].self_attn.attention_dropout,
        scaling=model.model.layers[0].self_attn.scaling
    )

    # --------- Compare prefix outputs ----------
    # Option 1: compare all prefix tokens at once
    full_prefix = full_out[:, 4, :, :]
    cut_prefix = cut_out[:, 4, :, :]
    diff_all = (full_prefix - cut_prefix).abs().max().item()
    print(f"Max diff over entire prefix: {diff_all:.6f}")

    return diff_all


def main():
    print("here")
    model = load_model_from_checkpoint("/Users/kevin/Downloads/last_1.ckpt")
    print("1")
    print(model)
    # input = model.model.embed_vector.unsqueeze(0).unsqueeze(0).expand(1, 4096, -1)
    attention_interface = ALL_ATTENTION_FUNCTIONS[model.model.layers[0].self_attn.config._attn_implementation]
    print(model.model.layers[0].self_attn.config._attn_implementation)

    test_causal_invariance(model)


if __name__ == "__main__":
    main()