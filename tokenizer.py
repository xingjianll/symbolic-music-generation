import os

from pathlib import Path
import dotenv

import utils

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


if __name__ == "__main__":
    print("Done")