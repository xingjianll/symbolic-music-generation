import torch
from pathlib import Path

from src.train_model_new import MidiDataset4D


def compute_stats(files, max_seq_len):
    dataset = MidiDataset4D(files, max_seq_len=max_seq_len)

    sum_dt = 0.0
    sum_dur = 0.0
    sum_vel = 0.0

    sum_dt_sq = 0.0
    sum_dur_sq = 0.0
    sum_vel_sq = 0.0

    count = 0

    for item in dataset:
        labels = item["labels"]  # shape (seq_len, 4)

        # Mask out padded positions (labels == -100)
        mask = (labels[:, 0] != -100)

        valid = labels[mask]

        if valid.numel() == 0:
            continue

        dt = valid[:, 0]
        dur = valid[:, 1]
        vel = valid[:, 3]

        sum_dt += dt.sum().item()
        sum_dur += dur.sum().item()
        sum_vel += vel.sum().item()

        sum_dt_sq += (dt ** 2).sum().item()
        sum_dur_sq += (dur ** 2).sum().item()
        sum_vel_sq += (vel ** 2).sum().item()

        count += valid.shape[0]

    # Means
    mean_dt = sum_dt / count
    mean_dur = sum_dur / count
    mean_vel = sum_vel / count

    # Variances
    var_dt = sum_dt_sq / count - mean_dt ** 2
    var_dur = sum_dur_sq / count - mean_dur ** 2
    var_vel = sum_vel_sq / count - mean_vel ** 2

    print("Mean Δt:", mean_dt)
    print("Var  Δt:", var_dt)
    print()
    print("Mean duration:", mean_dur)
    print("Var  duration:", var_dur)
    print()
    print("Mean velocity:", mean_vel)
    print("Var  velocity:", var_vel)

    return {
        "mean_dt": mean_dt,
        "var_dt": var_dt,
        "mean_dur": mean_dur,
        "var_dur": var_dur,
        "mean_vel": mean_vel,
        "var_vel": var_vel,
    }


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = project_dir / "data" / "aria-midi-v1-unique-ext" / "data"
    files = list(sorted(data_dir.glob("**/*.mid")))
    print(len(files))
    compute_stats(files[:1000], max_seq_len=1024)  # use a subset to test
