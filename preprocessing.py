import shutil
from pathlib import Path

import mido
from dotenv import load_dotenv
import os

from miditok import TokenizerConfig, REMI
from miditok.utils import split_files_for_training
from mido import MidiFile
from symusic import Score

import utils

load_dotenv()


def collect():
    raw_data_dir = os.environ.get("RAW_DATA_DIR")
    root_dir = os.path.join("./data", raw_data_dir)

    dest_dir = os.path.join("./data", "raw_midi")

    for dir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, dir)

        if not os.path.isdir(subdir_path):
            continue

        files = os.listdir(subdir_path)
        for file in files:
            if not file.endswith(".mid"):
                continue

            source_file = os.path.join(subdir_path, file)
            dest_file = os.path.join(dest_dir, file)

            shutil.copy(source_file, dest_file)
            print(f"Copied: {source_file} -> {dest_file}")


def merge_tracks():
    raw_midi_dir = os.path.join("./data", "raw_midi")
    dest_dir = os.path.join("./data", "midi")
    for file in os.listdir(raw_midi_dir):
        midi_path = os.path.join(raw_midi_dir, file)
        mid = MidiFile(midi_path)
        track = mido.merge_tracks(mid.tracks)
        del mid.tracks[:]
        mid.tracks.append(track)
        mid.save(os.path.join(dest_dir, file))

def transpose():
    project_dir = Path(__file__).resolve().parent
    midi_dir = project_dir.joinpath("data/midi")
    scores = []
    for mid in  os.listdir(midi_dir):
        if mid.endswith(".mid"):
            score = Score(midi_dir.joinpath(mid))
            for i in range(6):
                scores.append((score.shift_pitch(i), str(i)+mid))

            for i in range(-6, 0):
                scores.append((score.shift_pitch(i), str(i)+mid))
    for score in scores:
        score[0].dump_midi(midi_dir.joinpath(score[1]))

def chunk():
    project_dir = Path(__file__).resolve().parent
    midi_dir = project_dir.joinpath("data/midi")
    midi_paths = list(midi_dir.glob("**/*.mid"))
    dataset_chunks_dir = project_dir.joinpath("data/chunks")

    tokenizer = utils.get_tokenizer()

    split_files_for_training(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        save_dir=dataset_chunks_dir,
        max_seq_len=utils.CONTEXT_SIZE,
        num_overlap_bars=16
    )

if __name__ == "__main__":
    chunk()