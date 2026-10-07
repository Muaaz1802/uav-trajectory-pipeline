"""
simulate_data.py
-----------------
Stands in for Phase 1-2 of the synopsis (video -> frame -> (x,y,azimuth,elevation)
extraction) until real Anti-UAV footage is processed. Produces trajectories with
the SAME schema your real pipeline will produce, so nothing downstream changes
when you swap in real data:

    trajectory_id, scene, frame_idx, x, y, azimuth, elevation

Each trajectory simulates a UAV moving through a camera's image plane (640x512,
matching the SIDD/Anti-UAV infrared frame size) with piecewise-linear / gently
curving motion, realistic sensor jitter, and per-scene motion-noise profiles
(sky / urban / mountain), mirroring the "environment/scene" categories in your
Table 4.1 literature review and RQ3.

Azimuth/elevation are derived as the instantaneous heading of motion in the
image plane (a defensible proxy for "directional viewing cues associated with
the observation", exactly as your synopsis frames them in Section 5) plus
independent sensor noise, so Model B has a genuinely different, only-partially-
redundant signal from Model A -- otherwise the ablation would be trivially
either "identical" or "useless" by construction.
"""
import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)
IMG_W, IMG_H = 640, 512
SCENES = ["sky", "urban", "mountain"]
SCENE_NOISE = {"sky": 0.5, "urban": 1.5, "mountain": 2.5}  # px jitter, easy->hard


def make_trajectory(traj_id, scene, n_frames):
    noise = SCENE_NOISE[scene]
    x0, y0 = RNG.uniform(50, IMG_W - 50), RNG.uniform(50, IMG_H - 50)
    vx, vy = RNG.uniform(-4, 4), RNG.uniform(-3, 3)
    xs, ys, azs, els = [], [], [], []
    x, y = x0, y0
    for t in range(n_frames):
        # gentle random-walk acceleration -> curved, non-trivial paths
        vx += RNG.normal(0, 0.15)
        vy += RNG.normal(0, 0.15)
        vx = np.clip(vx, -6, 6)
        vy = np.clip(vy, -6, 6)
        x = np.clip(x + vx + RNG.normal(0, noise), 0, IMG_W)
        y = np.clip(y + vy + RNG.normal(0, noise), 0, IMG_H)
        # azimuth/elevation as heading-of-motion proxy + independent sensor noise
        az = (np.degrees(np.arctan2(vy, vx)) + RNG.normal(0, 3)) % 360
        el = np.clip(30 + 10 * np.sin(t / 15) + RNG.normal(0, 2), 0, 90)
        xs.append(x); ys.append(y); azs.append(az); els.append(el)
    return pd.DataFrame({
        "trajectory_id": traj_id, "scene": scene, "frame_idx": np.arange(n_frames),
        "x": xs, "y": ys, "azimuth": azs, "elevation": els,
    })


def build_dataset(n_trajectories=180, min_len=40, max_len=90):
    rows = []
    for i in range(n_trajectories):
        scene = SCENES[i % 3]
        n_frames = RNG.integers(min_len, max_len)
        rows.append(make_trajectory(i, scene, n_frames))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    os.makedirs("/home/claude/uav_traj/data", exist_ok=True)
    df = build_dataset()
    df.to_csv("/home/claude/uav_traj/data/trajectories.csv", index=False)
    print(f"Trajectories: {df['trajectory_id'].nunique()}, total rows: {len(df)}")
    print(df.groupby("scene")["trajectory_id"].nunique())
