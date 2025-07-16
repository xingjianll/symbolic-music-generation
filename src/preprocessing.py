import shutil
from pathlib import Path

import mido
from dotenv import load_dotenv
import os

from miditok.utils import split_files_for_training
from mido import MidiFile
from symusic import Score
from symusic.types import Track

import utils

load_dotenv()

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"


def split():
    raw_data_dir = data_dir / os.environ.get("RAW_DATA_DIR")

    single_track_dir = data_dir / "single_track"
    duo_track_dir = data_dir / "duo_track"

    for file in os.listdir(raw_data_dir):
        print(file)
        if file.endswith(".mid"):
            score = Score.from_file(os.path.join(raw_data_dir, file))
            a: Track = score.tracks[0]
            # if len(score.tracks) > 1:
            #     source_file = raw_data_dir / file
            #     dest_file = duo_track_dir / file
            #     shutil.copy(source_file, dest_file)
            if len(score.tracks) == 1:
                source_file = raw_data_dir / file
                dest_file = single_track_dir / file
                shutil.copy(source_file, dest_file)


def merge_tracks():
    raw_midi_dir = data_dir / os.environ.get("RAW_DATA_DIR2")
    dest_dir = data_dir / "single_track"
    for file in os.listdir(raw_midi_dir):
        midi_path = os.path.join(raw_midi_dir, file)
        print(midi_path)
        mid = MidiFile(midi_path)
        track = mido.merge_tracks(mid.tracks)
        del mid.tracks[:]
        mid.tracks.append(track)
        mid.save(os.path.join(dest_dir, file))


def simplify():
    single_track_dir = data_dir / 'single_track'
    dest_dir = data_dir / "single_track"
    for file in os.listdir(single_track_dir):
        score = Score(single_track_dir / file)
        a: Track = score.tracks[0]
        for

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
        num_overlap_bars=2
    )

if __name__ == "__main__":
    merge_tracks()