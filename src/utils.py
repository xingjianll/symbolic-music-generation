import symusic
from symusic.core import ScoreTick, TempoTick

CONTEXT_SIZE = 2048


def merge_score_tracks(score: symusic.Score) -> None:
    """
    Merge tracks in a score by combining their notes into a single track.
    """
    notes = []
    for track in score.tracks:
        for note in track.notes:
            notes.append(note)
    score.tracks.clear()
    track = symusic.Track()
    score.tracks.append(track)
    for note in notes:
        track.notes.append(note)


def handle_tempos(score: symusic.Score) -> None:
    """
    Handle tempo changes in a score by adjusting the ticks of notes.
    """
    for idx, tempo in enumerate(score.tempos):
        tempo: TempoTick
        if idx+1 >= len(score.tempos):
            if tempo.time < 0:
                tempo.time = 0
            continue

        next_tempo = score.tempos[idx + 1]
        if tempo.time < 0 < next_tempo.time:
            tempo.time = 0
    score.tempos.filter(lambda x: x.time >= 0, inplace=True)


def handle_key_sigs(score: symusic.Score) -> None:
    """
    Handle key signatures changes in a score by adjusting the ticks of notes.
    """
    for idx, key_sig in enumerate(score.key_signatures):
        tempo: TempoTick
        if idx+1 >= len(score.key_signatures):
            if key_sig.time < 0:
                key_sig.time = 0
            continue

        next_key_sig = score.key_signatures[idx + 1]
        if key_sig.time < 0 < next_key_sig.time:
            key_sig.time = 0
    score.key_signatures.filter(lambda x: x.time >= 0, inplace=True)


def handle_time_sigs(score: symusic.Score) -> None:
    """
    Handle time signature changes in a score by adjusting their tick times.
    Removes or corrects any with negative time values.
    """
    for idx, time_sig in enumerate(score.time_signatures):
        if idx + 1 >= len(score.time_signatures):
            if time_sig.time < 0:
                time_sig.time = 0
            continue

        next_time_sig = score.time_signatures[idx + 1]
        if time_sig.time < 0 < next_time_sig.time:
            time_sig.time = 0

    score.time_signatures.filter(lambda x: x.time >= 0, inplace=True)
