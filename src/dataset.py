"""
dataset.py
----------
Coordinate normalization + sliding-window sequence construction + trajectory-level
train/val/test split (Section 5, Phase 3 of the timeline). Split happens on
trajectory_id BEFORE windowing, so correlated neighbouring frames from the same
trajectory never leak across splits.
"""
import numpy as np
import pandas as pd

IMG_W, IMG_H = 640, 512
HIST, HORIZON = 10, 10  # 10 history steps, predict up to 10 steps ahead


def load_and_split(path, seed=42, train_frac=0.7, val_frac=0.15):
    df = pd.read_csv(path)
    df["x_n"] = df["x"] / IMG_W
    df["y_n"] = df["y"] / IMG_H
    df["az_n"] = df["azimuth"] / 360.0
    df["el_n"] = df["elevation"] / 90.0

    traj_ids = df["trajectory_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(traj_ids)
    n = len(traj_ids)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_ids = set(traj_ids[:n_train])
    val_ids = set(traj_ids[n_train:n_train + n_val])
    test_ids = set(traj_ids[n_train + n_val:])

    return (df[df.trajectory_id.isin(train_ids)],
            df[df.trajectory_id.isin(val_ids)],
            df[df.trajectory_id.isin(test_ids)])


def make_windows(df, hist=HIST, horizon=HORIZON, spatial_only=False):
    """Returns X (N, hist, F), Y (N, horizon, 2) in normalized [0,1] coords."""
    feat_cols = ["x_n", "y_n"] if spatial_only else ["x_n", "y_n", "az_n", "el_n"]
    xs, ys = [], []
    for _, g in df.groupby("trajectory_id"):
        g = g.sort_values("frame_idx")
        arr = g[feat_cols].values
        xy = g[["x_n", "y_n"]].values
        for i in range(len(g) - hist - horizon + 1):
            xs.append(arr[i:i + hist])
            ys.append(xy[i + hist:i + hist + horizon])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)
