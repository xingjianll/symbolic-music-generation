import shutil
from pathlib import Path
from typing import List

import mido
import numpy as np
import symusic
from dotenv import load_dotenv
import os

from miditok.utils import split_files_for_training, get_bars_ticks
from mido import MidiFile
from symusic import Score
from symusic.core import NoteTick, NoteTickList
from symusic.types import Track


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
        # print(file)
        score = Score.from_file(os.path.join(raw_midi_dir, file))
        if len(score.tracks) == 1:
            source_file = raw_midi_dir / file
            dest_file = single_track_dir / file
            shutil.copy(source_file, dest_file)
        else:
            print(file)
            print(score.tracks)


def merge_tracks(source_dir: str, dest_dir: str) -> None:
    raw_midi_dir = data_dir / source_dir
    dest_dir = data_dir / dest_dir
    for file in os.listdir(raw_midi_dir):
        if not file.endswith(".mid"):
            continue

        print(file)
        score = Score.from_file(os.path.join(raw_midi_dir, file))
        notes = []
        for track in score.tracks:
            for note in track.notes:
                notes.append(note)
        score.tracks.clear()
        track = symusic.Track()
        score.tracks.append(track)
        for note in notes:
            track.notes.append(note)
        score.dump_midi(dest_dir / file)

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


def split_notes_into_bars(track: Track, score: Score) -> List[List[NoteTick]]:
    """
    Split track notes into bars based on bar tick positions.

    Args:
        track: Input track to split
        score: The full Score object (needed to compute barlines)

    Returns:
        List of lists, where each inner list contains notes for one bar
    """
    # Get bar tick positions
    bar_ticks = sorted(get_bars_ticks(score))
    if not bar_ticks:
        return []

    notes = list(track.notes)
    notes.sort(key=lambda x: x.start)

    # Split notes into bars
    bars = []
    current_bar_notes = []
    bar_idx = 0
    tick_tolerance = 20

    for note in notes:
        if note.duration == 0:
            continue

       # Find which bar this note belongs to
        # Check if note should be in the next bar (with tolerance for timing inaccuracies)
        while (bar_idx < len(bar_ticks) - 1 and
               note.start + tick_tolerance >= bar_ticks[bar_idx + 1]):
            # Finish current bar and start new one
            if current_bar_notes:
                bars.append(current_bar_notes)
                current_bar_notes = []
            bar_idx += 1

        current_bar_notes.append(note)

    # Add the last bar
    if current_bar_notes:
        bars.append(current_bar_notes)

    return bars


def segment_melody_by_bars(track: Track, score: Score, min_bars_per_segment: int = 4, max_bars_per_segment: int = 16, similarity_threshold: float = 0.3) -> List[Track]:
    """
    Segment melody by bars, ensuring splits only happen at bar boundaries.

    Args:
        track: Input track to segment
        score: The full Score object (needed to compute barlines)
        min_bars_per_segment: Minimum number of bars per segment
        max_bars_per_segment: Maximum number of bars per segment
        similarity_threshold: Threshold for musical similarity (0-1, lower = more segments)

    Returns:
        List of Track objects representing the segmented passages
    """
    if not track.notes or len(track.notes) < 2:
        return [track]

    # Split notes into bars
    bars = split_notes_into_bars(track, score)

    if not bars:
        return [track]

    # Now segment bars using musical similarity
    segments = []
    current_segment_bars = [bars[0]]

    for i in range(1, len(bars)):
        current_bar = bars[i]
        prev_bar = bars[i - 1]

        if not current_bar or not prev_bar:
            current_segment_bars.append(current_bar)
            continue

        # Calculate bar-level features
        dissimilarity = calculate_bar_dissimilarity(prev_bar, current_bar)

        current_segment_length = len(current_segment_bars)

        should_split = (
            dissimilarity > similarity_threshold or
            (current_segment_length >= min_bars_per_segment and dissimilarity > similarity_threshold * 0.7) or
            current_segment_length >= max_bars_per_segment
        )

        if should_split and current_segment_length >= min_bars_per_segment:
            segments.append(current_segment_bars)
            current_segment_bars = [current_bar]
        else:
            current_segment_bars.append(current_bar)

    # Add the last segment
    if current_segment_bars:
        segments.append(current_segment_bars)

    # Convert bar segments back to Track objects
    track_segments = []
    for segment_bars in segments:
        new_track = symusic.Track()
        new_track.program = track.program
        new_track.is_drum = track.is_drum
        new_track.name = track.name

        # Add all notes from all bars in this segment
        for bar_notes in segment_bars:
            for note in bar_notes:
                new_track.notes.append(note)

        track_segments.append(new_track)

    return track_segments


def calculate_bar_dissimilarity(bar1_notes: List, bar2_notes: List) -> float:
    """
    Calculate dissimilarity between two bars based on musical features.

    Args:
        bar1_notes: List of notes in the first bar
        bar2_notes: List of notes in the second bar

    Returns:
        Dissimilarity score (0-1, higher = more dissimilar)
    """
    if not bar1_notes or not bar2_notes:
        return 1.0

    # Feature 1: Average pitch difference
    avg_pitch1 = np.mean([note.pitch for note in bar1_notes])
    avg_pitch2 = np.mean([note.pitch for note in bar2_notes])
    pitch_diff = abs(avg_pitch1 - avg_pitch2)
    pitch_diff_norm = min(pitch_diff / 12, 1.0)  # Normalize by octave

    # Feature 2: Note count difference
    count_diff = abs(len(bar1_notes) - len(bar2_notes))
    count_diff_norm = min(count_diff / max(len(bar1_notes), len(bar2_notes)), 1.0)

    # Feature 3: Average duration difference
    avg_dur1 = np.mean([note.duration for note in bar1_notes])
    avg_dur2 = np.mean([note.duration for note in bar2_notes])
    dur_ratio = max(avg_dur1, avg_dur2) / min(avg_dur1, avg_dur2)
    dur_diff_norm = min((dur_ratio - 1) / 3, 1.0)  # Normalize duration ratio

    # Feature 4: Pitch range difference
    pitch_range1 = max(note.pitch for note in bar1_notes) - min(note.pitch for note in bar1_notes)
    pitch_range2 = max(note.pitch for note in bar2_notes) - min(note.pitch for note in bar2_notes)
    range_diff = abs(pitch_range1 - pitch_range2)
    range_diff_norm = min(range_diff / 12, 1.0)

    # Feature 5: Time gap between bars
    bar1_end = max(note.end for note in bar1_notes)
    bar2_start = min(note.start for note in bar2_notes)
    time_gap = bar2_start - bar1_end
    time_gap_norm = min(time_gap / 240, 1.0) if time_gap > 0 else 0

    # Weighted combination
    dissimilarity = (
        pitch_diff_norm * 0.3 +
        count_diff_norm * 0.2 +
        dur_diff_norm * 0.2 +
        range_diff_norm * 0.2 +
        time_gap_norm * 0.1
    )

    return dissimilarity


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
    merge_tracks('duo_track', 'single_track_combined')