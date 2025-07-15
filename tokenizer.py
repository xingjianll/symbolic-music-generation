import os

from pathlib import Path
import dotenv
from miditok import TokenizerConfig, PerTok

import utils


def get_tokenizer(load=True):
    config = TokenizerConfig(
        num_velocities=8,
        use_velocities=True,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
        use_pitch_intervals=False,
        use_programs=False,
        use_pitchdrum_tokens=False,
        ticks_per_quarter=320,
        use_microtiming=False,
        max_microtiming_shift=0.125
    )
    tokenizer = PerTok(config)
    if load:
        tokenizer.from_pretrained("xingjianll/midi-tokenizer")

    return tokenizer


if __name__ == "__main__":
    dotenv.load_dotenv()

    # Creating a multitrack tokenizer, read the doc to explore all the parameters
    tokenizer = utils.get_tokenizer(load=False)

    # Train the tokenizer with Byte Pair Encoding (BPE)
    project_dir = Path(__file__).resolve().parent
    midi_dir = project_dir.joinpath("data/midi")
    midis = midi_dir.glob("**/*.mid")
    for mid in midis:
        print(mid)
    files_paths = list(midis)
    tokenizer.train(vocab_size=30000, files_paths=files_paths)
    tokenizer.save(Path(project_dir, "tokenizer.json"))
    # And pushing it to the Hugging Face hub (you can download it back with .from_pretrained)
    tokenizer.push_to_hub("xingjianll/midi-tokenizer-v2", private=False, token=os.environ["HF_TOKEN"])

    print("Done")