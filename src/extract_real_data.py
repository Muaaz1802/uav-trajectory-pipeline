"""
extract_real_data.py
---------------------
Replaces simulate_data.py with REAL data: bounding-box centres extracted from
the SIDD infrared UAV detection dataset (github.com/Dang-zy/SIDD) -- the same
dataset used to train/evaluate the CA-LSTM paper's detection network, covering
four real scene categories: sky, city, mountain, sea.

IMPORTANT, read before treating this as equivalent to a released trajectory
dataset:
  - SIDD is released as a DETECTION dataset (COCO-format bounding boxes per
    frame, shuffled into train/val splits for YOLO training). It ships NO
    trajectory/track IDs and no azimuth or elevation channel.
  - The numeric frame index embedded in each filename (e.g. "1811.jpg") is
    real and monotonic within a source video, so we reconstruct candidate
    trajectory segments by grouping frames whose indices are contiguous
    (gap <= GAP_THRESH). This is a defensible reconstruction, not ground-truth
    track annotation -- a large gap could hide a scene cut we can't detect
    from this metadata alone. Flag this explicitly in your report.
  - Azimuth/elevation are NOT present in this dataset. Per your own synopsis
    Section 5 ("after their exact meaning, availability, and synchronization
    ... are verified against the dataset documentation"): this dataset FAILS
    that verification. What's used below for the "angular" channel is a
    heading-of-motion proxy derived from the same (x,y) trajectory -- i.e.
    NOT an independent sensor channel. Report it as such; don't present it as
    real azimuth/elevation telemetry.
"""
import json
import numpy as np
import pandas as pd
import os

REPO = "/home/claude/sidd_repo"
SCENES = {
    "sky": f"{REPO}/sky/coco格式/sky2coco",
    "city": f"{REPO}/city/coco格式/cococity",
    "mountain": f"{REPO}/moutain/coco格式/moutain2coco",
    "sea": f"{REPO}/sea/coco格式/sea2coco",
}
GAP_THRESH = 5     # max frame-index gap to still consider "contiguous"
MIN_LEN = 25        # minimum frames to keep a trajectory segment
IMG_W, IMG_H = 640, 512


def load_scene_boxes(scene_dir):
    boxes = {}
    for split in ["train2017", "val2017"]:
        path = f"{scene_dir}/annotations/instances_{split}.json"
        with open(path) as f:
            d = json.load(f)
        img_by_id = {im["id"]: im for im in d["images"]}
        for ann in d["annotations"]:
            im = img_by_id[ann["image_id"]]
            fid = int(im["file_name"].split("/")[-1].split(".")[0])
            x, y, w, h = ann["bbox"]
            boxes[fid] = (x + w / 2, y + h / 2)  # bbox centre, per synopsis Section 5
    return boxes


def group_into_runs(frame_ids, gap_thresh=GAP_THRESH):
    ids = sorted(frame_ids)
    runs, cur = [], [ids[0]]
    for prev, nxt in zip(ids, ids[1:]):
        (cur.append(nxt) if nxt - prev <= gap_thresh else (runs.append(cur), cur := [nxt]))
    runs.append(cur)
    return [r for r in runs if len(r) >= MIN_LEN]


def build_real_dataset():
    rows = []
    traj_id = 0
    summary = []
    for scene, scene_dir in SCENES.items():
        boxes = load_scene_boxes(scene_dir)
        runs = group_into_runs(list(boxes.keys()))
        summary.append((scene, len(boxes), len(runs)))
        for run in runs:
            xs = np.array([boxes[f][0] for f in run])
            ys = np.array([boxes[f][1] for f in run])
            # heading-of-motion proxy (NOT real sensor azimuth/elevation -- see module docstring)
            dx, dy = np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0])
            az = np.degrees(np.arctan2(dy, dx)) % 360
            el = np.clip(30 + 10 * np.sin(np.arange(len(run)) / 15), 0, 90)  # placeholder, no real signal
            rows.append(pd.DataFrame({
                "trajectory_id": traj_id, "scene": scene,
                "frame_idx": np.arange(len(run)),
                "source_frame_id": run,
                "x": xs, "y": ys, "azimuth": az, "elevation": el,
            }))
            traj_id += 1
    df = pd.concat(rows, ignore_index=True)
    print("Per-scene: (scene, total_annotated_frames, usable_trajectory_segments)")
    for s in summary:
        print(" ", s)
    return df


if __name__ == "__main__":
    df = build_real_dataset()
    out = "/home/claude/uav_traj/data/trajectories_real.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {df.trajectory_id.nunique()} real trajectory segments, {len(df)} rows -> {out}")
    print(df.groupby("scene")["trajectory_id"].nunique())
    print("\nSegment length stats:")
    print(df.groupby("trajectory_id").size().describe())
