import symusic
from symusic.core import ScoreTick, TempoTick

CONTEXT_SIZE = 1024


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