import shutil
from pathlib import Path

import mido
import symusic
from dotenv import load_dotenv
import os

from miditok.utils import split_files_for_training
from mido import MidiFile
from symusic import Score
from symusic.core import NoteTick, NoteTickList
from symusic.types import Track

import utils

load_dotenv()

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"


def copy_single_track(source_dir: str, dest_dir: str) -> None:
    """
    Copy single tracks from the source dir to the destination dir.
    :param source_dir:
    :param dest_dir:
    :return:
    """
    raw_midi_dir = data_dir / source_dir

    single_track_dir = data_dir / dest_dir

    for file in os.listdir(raw_midi_dir):
        if not file.endswith(".mid"):
            continue
        print(file)
        score = Score.from_file(os.path.join(raw_midi_dir, file))
        if len(score.tracks) == 1:
            source_file = raw_midi_dir / file
            dest_file = single_track_dir / file
            shutil.copy(source_file, dest_file)


def merge_tracks(source_dir: str, dest_dir: str) -> None:
    raw_midi_dir = data_dir / source_dir
    dest_dir = data_dir / dest_dir
    for file in os.listdir(raw_midi_dir):
        if not file.endswith(".mid"):
            continue
        midi_path = os.path.join(raw_midi_dir, file)
        print(midi_path)
        mid = MidiFile(midi_path)
        track = mido.merge_tracks(mid.tracks)
        del mid.tracks[:]
        mid.tracks.append(track)
        mid.save(os.path.join(dest_dir, file))


def extract_melody(t: Track) -> Track:
    melody = []
    z: NoteTick
    # notes: NoteTickList = t.notes.filter(lambda note: note.duration >= 240, inplace=True)
    notes: NoteTickList = t.notes
    notes.sort(key=lambda x: x.start, inplace=True)
    z: NoteTick
    concurrent = []
    curr = None
    while notes:
        note = notes.pop(0)
        concurrent.append(note)
        if notes:
            note2 = notes[0]
            if note2.start > note.start:
                # pop highest pitch
                max = concurrent.pop()
                while concurrent:
                    a = concurrent.pop()
                    if a.pitch > max.pitch:
                        max = a
                if curr is None or max.pitch > curr.pitch or max.start >= curr.end - 20:
                    melody.append(max)
                    curr = max

    new_track = symusic.Track()
    for note in melody:
        new_track.notes.append(note)
    return new_track


def transpose(source_dir: str, dest_dir: str) -> None:
    midi_dir = data_dir / source_dir
    dest_dir = data_dir / dest_dir
    scores = []
    for mid in  os.listdir(midi_dir):
        if not mid.endswith(".mid"):
            continue
        score = Score(midi_dir.joinpath(mid))
        for i in range(6):
            scores.append((score.shift_pitch(i), str(i)+mid))

        for i in range(-6, 0):
            scores.append((score.shift_pitch(i), str(i)+mid))

    for score in scores:
        score[0].dump_midi(dest_dir.joinpath(score[1]))


# def chunk():
#     project_dir = Path(__file__).resolve().parent
#     midi_dir = project_dir.joinpath("data/midi")
#     midi_paths = list(midi_dir.glob("**/*.mid"))
#     dataset_chunks_dir = project_dir.joinpath("data/chunks")
#
#     tokenizer = utils.get_tokenizer()
#
#     split_files_for_training(
#         files_paths=midi_paths,
#         tokenizer=tokenizer,
#         save_dir=dataset_chunks_dir,
#         max_seq_len=utils.CONTEXT_SIZE,
#         num_overlap_bars=2
#     )

if __name__ == "__main__":
    merge_tracks()