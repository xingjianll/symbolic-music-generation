import symusic

from generate_music import notes_to_midi
from src.train_model_new import _create_position_tensors
from src.utils import merge_score_tracks

score = symusic.Score.from_file(str("/Users/kevin/PycharmProjects/symbolic-music-generation/data/aria-midi-v1-unique-ext/data/cs/140922_0.mid"))



# Use preprocessing functions to clean up the score (in tick format)
merge_score_tracks(score)

# Convert to seconds after preprocessing
score = score.to("second")


track = score.tracks[0]
all_notes = list(track.notes)

# Sort notes by start time
all_notes.sort(key=lambda x: x.start)

# Create 4D position tensors [start_time, duration, pitch, velocity]
position_tensors = _create_position_tensors(all_notes, score)

if __name__ == "__main__":
    print(position_tensors[:10, :])
    notes_to_midi(position_tensors[1:, :].numpy(), "./out9.mid")