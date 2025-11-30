import random
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import symusic
from ariautils.midi import MidiDict
from src.utils import merge_score_tracks
from src.model.model import MidiAria

device = "cuda"


# -------------------------------
# Utility: dump notes → midi temp
# -------------------------------
def dump_temp(notes):
    import tempfile
    score = symusic.Score()
    track = symusic.Track()
    track.notes.extend(notes)
    score.tracks.append(track)

    tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
    score.dump_midi(tmp.name)
    return tmp.name


# -------------------------------
# Replicates EXACT dataset slicing
# -------------------------------
def slice_notes_like_dataset(score):
    merge_score_tracks(score)
    track = score.tracks[0]
    notes = list(track.notes)
    notes.sort(key=lambda n: n.start)

    notes = notes[:512]
    if len(notes) < 50:
        return None

    total = min(len(notes), 512)
    gap_len = int(total * random.uniform(0.1, 0.4))
    start_idx = random.randint(10, total - gap_len - 10)

    prefix_notes = notes[:start_idx]    # Part B (to be predicted)
    suffix_notes = notes[start_idx:]    # Part A (prompt)

    return prefix_notes, suffix_notes


# -------------------------------
# MAIN TEST
# -------------------------------
def test_infill(model_path):
    # Load tokenizer and base HF model
    tokenizer = AutoTokenizer.from_pretrained(
        "loubb/aria-medium-base",
        trust_remote_code=True,
        add_eos_token=True,
        add_dim_token=True
    )
    tokenizer.preprocess_score = lambda x: x

    # Load your LoRA-wrapped model
    print("Loading trained infill model...")
    midiaria = MidiAria(tokenizer, None)
    midiaria.to_lora()
    ckpt = torch.load(model_path, map_location="cpu")
    raw = ckpt["state_dict"]

    # Strip lightning prefix
    clean = {k.replace("model.", "", 1): v for k, v in raw.items()}

    midiaria.model.load_state_dict(clean, strict=False)
    midiaria.eval().to(device)

    # Pick one random training MIDI
    project_dir = Path(__file__).resolve().parents[1]

    melody_train_files = sorted((project_dir / 'data' / 'aria-midi-v1-unique-ext').glob("**/*.mid"))[:3000]
    midi_path = melody_train_files[0]
    print(f"Testing on file: {midi_path.name}")

    # Load score and slice like dataset
    score = symusic.Score.from_file(str(midi_path))
    sliced = slice_notes_like_dataset(score)
    if sliced is None:
        print("File too short, skipping.")
        return

    prefix_notes, suffix_notes = sliced   # prefix = target, suffix = prompt

    # Convert to temp MIDI
    prefix_midi = dump_temp(prefix_notes)
    suffix_midi = dump_temp(suffix_notes)

    # Tokenize
    prefix_ids = tokenizer._tokenizer.encode(
        tokenizer.tokenize(MidiDict.from_midi(prefix_midi),
                           add_eos_token=True,
                           add_dim_token=True)
    )
    suffix_ids = tokenizer._tokenizer.encode(
        tokenizer.tokenize(MidiDict.from_midi(suffix_midi),
                           add_eos_token=True,
                           add_dim_token=True)
    )

    os.unlink(prefix_midi)
    os.unlink(suffix_midi)

    # Build input = suffix + prefix
    midi_dict_output = tokenizer.decode(suffix_ids)
    midi_dict_output.to_midi().save("./output0.mid")

    # -------------------------------
    # GENERATE PREDICTED PREFIX
    # -------------------------------
    print("Generating completion...")

    continuation = midiaria.model.generate(
        torch.tensor([suffix_ids + prefix_ids[:64]], device='cuda'),
        max_length=len(suffix_ids)+64+100,
        do_sample=True,
        temperature=0.97,
        top_p=0.95,
        use_cache=True,
    )

    prompt_len = len(suffix_ids)
    generated_tokens = continuation[0][prompt_len:]  # only the new part
    print(f"Generated length is: {len(generated_tokens)}")

# Decode just the harmonization
    midi_dict_output = tokenizer.decode(generated_tokens.tolist())
    midi_dict_output.to_midi().save("./output.mid")


if __name__ == "__main__":
    # path to one of your checkpoints
    model_ckpt = "/workspace/symbolic-music-generation/checkpoints/aria-infill-epoch=11-val_loss=2.6428.ckpt"

    test_infill(model_ckpt)
