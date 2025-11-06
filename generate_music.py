"""
Generate music using a trained 4D continuous music generation model.

Usage:
    python generate_music.py --checkpoint path/to/checkpoint.ckpt --output output.mid --length 100
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import symusic
from src.model.model import MidiQwenNew
from src.train_model_new import MidiDataset4DStreaming, MidiDataset4D, custom_collate_fn
from src.utils import CONTEXT_SIZE


def load_model_from_checkpoint(checkpoint_path: str, device='cuda'):
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
    
    return model


def clamp_4d_positions(positions_4d: torch.Tensor) -> torch.Tensor:
    """Clamp 4D positions to reasonable musical ranges."""
    # Clamp to reasonable ranges
    positions_4d[:, :, 0] = torch.clamp(positions_4d[:, :, 0], 0, 300)      # time: 0-300 seconds
    positions_4d[:, :, 1] = torch.clamp(positions_4d[:, :, 1], 0, 10)       # duration: 0-10 seconds  
    positions_4d[:, :, 2] = torch.clamp(positions_4d[:, :, 2], 0, 127)      # pitch: 0-127
    positions_4d[:, :, 3] = torch.clamp(positions_4d[:, :, 3], 0, 127)      # velocity: 0-127
    
    return positions_4d


def positions_to_midi(positions_4d: torch.Tensor, output_path: str):
    """Convert 4D positions back to MIDI file."""
    positions = positions_4d[0].cpu().numpy()  # Take first batch item
    
    # Create new score
    score = symusic.Score()
    track = symusic.Track(program=0, is_drum=False, name="Generated Music")
    
    current_time = 0.0
    for i, pos in enumerate(positions):
        start_time, duration, pitch, velocity = pos
        
        # Skip special tokens
        if pitch <= 2:  # BOS, EOS, PAD tokens
            continue
            
        # Ensure reasonable values
        if duration <= 0 or velocity <= 0 or pitch < 21 or pitch > 108:
            continue
            
        # Create note
        note = symusic.Note(
            start=int(start_time * 480),  # Convert to ticks (480 ticks per quarter note)
            end=int((start_time + duration) * 480),
            pitch=int(pitch),
            velocity=int(velocity)
        )
        track.notes.append(note)
    
    score.tracks.append(track)
    score.dump_midi(output_path)
    print(f"Generated MIDI saved to {output_path}")


def generate_music(model, prompt_length: int = 50, total_length: int = 200, device='cuda'):
    """Generate music using the trained model."""
    print(f"Generating {total_length} tokens...")
    
    # Start with a simple prompt: BOS token
    current_positions = torch.tensor([[[0.0, 0.0, 0, 0]]], device=device)  # (1, 1, 4) - BOS
    
    with torch.no_grad():
        for step in range(total_length):
            if step % 20 == 0:
                print(f"Step {step}/{total_length}")
            
            # Prepare input (pad to context size if needed)
            seq_len = current_positions.shape[1]
            if seq_len >= CONTEXT_SIZE:
                # Take the last CONTEXT_SIZE-1 tokens to leave room for new generation
                current_positions = current_positions[:, -(CONTEXT_SIZE-1):]
                seq_len = current_positions.shape[1]
            
            # Pad to context size
            padded_positions = torch.zeros(1, CONTEXT_SIZE, 4, device=device)
            padded_positions[:, :seq_len] = current_positions
            
            # Create input_ids (all zeros for vocab_size=1)
            input_ids = torch.zeros(1, CONTEXT_SIZE, dtype=torch.long, device=device)
            
            # Create attention mask
            attention_mask = torch.zeros(1, CONTEXT_SIZE, dtype=torch.long, device=device)
            attention_mask[:, :seq_len] = 1
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_tensors=padded_positions
            )
            
            # Get the prediction for the next token (last non-padded position)
            next_4d_prediction = outputs.logits[0, seq_len-1:seq_len]  # (1, 4)
            
            # Clamp to reasonable ranges
            next_position = clamp_4d_positions(next_4d_prediction.unsqueeze(0))  # (1, 1, 4)

            values = next_position[0, 0].cpu().numpy()
            print(f"Step {step}: time={values[0]:.2f}s, duration={values[1]:.2f}s, pitch={values[2]:.0f}, velocity={values[3]:.0f}")
            
            # Append to sequence
            current_positions = torch.cat([current_positions, next_position], dim=1)
    
    return current_positions


def main():
    parser = argparse.ArgumentParser(description='Generate music using trained 4D model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='generated_music.mid', help='Output MIDI file path')
    parser.add_argument('--length', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # Generate music
    generated_positions = generate_music(model, total_length=args.length, device=args.device)
    
    # Convert to MIDI
    positions_to_midi(generated_positions, args.output)
    
    print(f"Music generation complete! Check {args.output}")


if __name__ == "__main__":
    main()