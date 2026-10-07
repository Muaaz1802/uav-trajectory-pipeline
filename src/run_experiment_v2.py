"""
run_experiment_v2.py
---------------------
Paper-grade experimental protocol on top of the same real SIDD-derived data.
Adds, relative to run_experiment.py:

  1. Small validation-based hyperparameter search (hidden size x LR) done once,
     reused across seeds -- documented explicitly, not hidden.
  2. FOUR model variants: Spatial LSTM, Spatial+Angular LSTM, Spatial GRU,
     Spatial Bi-LSTM (GRU/Bi-LSTM cited in your own related-work table).
  3. THREE baselines: Constant Velocity, Linear Regression, Kalman filter
     (constant-velocity Kalman -- the standard trajectory-tracking baseline
     reviewers expect and that CV/LR alone don't cover).
  4. 5-seed repetition of the trajectory-level split + training -> mean +/- std
     for every metric, not single-run point estimates.
  5. Paired significance testing (t-test + Wilcoxon) of the best LSTM vs each
     baseline, on the SAME test windows from one fixed seed.
  6. Leave-one-scene-out generalization check (train on 3 scenes, test on the
     4th) -- the only way to get a generalization signal out of 49 trajectories.
  7. REAL end-to-end latency: actual OpenCV image decode on real SIDD frames,
     not a simulated placeholder.

READ THE CAVEATS IN README.md before writing these numbers into a paper --
this makes the pilot far more rigorous, it does not manufacture data that
doesn't exist (real track-level trajectories, real azimuth/elevation).
"""
import glob
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2

from dataset import load_and_split, make_windows, HIST, HORIZON, IMG_W, IMG_H
from baselines import constant_velocity_predict, linear_regression_predict
from kalman_baseline import kalman_predict
from models import build_model
from stats_utils import paired_test, aggregate_seeds

DATA_PATH = "/home/claude/uav_traj/data/trajectories_real.csv"
SEEDS = [0, 1, 2, 3, 4]
HORIZONS = (1, 3, 5, 10)
SIDD_IMAGES_GLOB = "/home/claude/sidd_repo/*/coco格式/*/train2017/*.jpg"


def to_pixels(arr_norm):
    out = arr_norm.copy()
    out[..., 0] *= IMG_W
    out[..., 1] *= IMG_H
    return out


def per_window_ade(pred_px, true_px, h):
    err = np.linalg.norm(pred_px[:, :h] - true_px[:, :h], axis=-1)
    return err.mean(axis=1)  # (N,) -- one ADE per test window, for pairing


def ade_fde_rmse(pred_px, true_px, horizons=HORIZONS):
    results = {}
    err = np.linalg.norm(pred_px - true_px, axis=-1)
    for h in horizons:
        h = min(h, err.shape[1])
        results[h] = {
            "ADE": err[:, :h].mean(),
            "FDE": err[:, h - 1].mean(),
            "RMSE": np.sqrt(((pred_px[:, :h] - true_px[:, :h]) ** 2).mean()),
        }
    return results


def train_model(model, X_train, Y_train, X_val, Y_val, epochs=40, lr=1e-3, batch=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt, Yt = torch.tensor(X_train), torch.tensor(Y_train)
    Xv, Yv = torch.tensor(X_val), torch.tensor(Y_val)
    n = len(Xt)
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


# ---------------------------------------------------------------------------
# Step 1: small hyperparameter search (once, on seed 0), for the Spatial LSTM
# ---------------------------------------------------------------------------
def hyperparam_search():
    train_df, val_df, _ = load_and_split(DATA_PATH, seed=0)
    Xtr, Ytr = make_windows(train_df, spatial_only=True)
    Xva, Yva = make_windows(val_df, spatial_only=True)
    grid = list(itertools.product([32, 64, 96], [1e-3, 5e-4]))
    results = []
    for hidden, lr in grid:
        model = build_model("lstm", in_features=2, horizon=HORIZON, hidden=hidden)
        _, val_loss = train_model(model, Xtr, Ytr, Xva, Yva, epochs=25, lr=lr, seed=0)
        results.append({"hidden": hidden, "lr": lr, "val_loss": val_loss})
        print(f"  hidden={hidden:3d} lr={lr:.0e} -> val_loss={val_loss:.5f}")
    best = min(results, key=lambda r: r["val_loss"])
    print(f"Best config: hidden={best['hidden']}, lr={best['lr']}")
    return best["hidden"], best["lr"], pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Step 2: multi-seed main comparison (4 models + 3 baselines)
# ---------------------------------------------------------------------------
def run_multiseed(hidden, lr):
    MODEL_SPECS = [
        ("lstm", True, "Spatial LSTM"),
        ("lstm", False, "Spatial+Angular LSTM"),
        ("gru", True, "Spatial GRU"),
        ("bilstm", True, "Spatial Bi-LSTM"),
    ]
    all_rows = []
    fixed_seed_test_windows = {}  # for significance testing, seed 0 only

    for seed in SEEDS:
        train_df, val_df, test_df = load_and_split(DATA_PATH, seed=seed)

        for model_name, spatial_only, tag in MODEL_SPECS:
            Xtr, Ytr = make_windows(train_df, spatial_only=spatial_only)
            Xva, Yva = make_windows(val_df, spatial_only=spatial_only)
            Xte, Yte = make_windows(test_df, spatial_only=spatial_only)
            if len(Xtr) == 0 or len(Xte) == 0:
                continue
            n_feat = Xtr.shape[-1]
            model = build_model(model_name, in_features=n_feat, horizon=HORIZON, hidden=hidden)
            model, _ = train_model(model, Xtr, Ytr, Xva, Yva, epochs=40, lr=lr, seed=seed)
            pred = model(torch.tensor(Xte)).detach().numpy()
            pred_px, true_px = to_pixels(pred), to_pixels(Yte)
            metrics = ade_fde_rmse(pred_px, true_px)
            for h, m in metrics.items():
                all_rows.append({"Method": tag, "Seed": seed, "Horizon": h, **m})
            if seed == 0:
                fixed_seed_test_windows[tag] = (pred_px, true_px)

        # baselines (deterministic, still recomputed per seed since test split changes)
        Xte_xy, Yte_xy = make_windows(test_df, spatial_only=True)
        for name, fn in [("Constant Velocity", constant_velocity_predict),
                          ("Linear Regression", linear_regression_predict),
                          ("Kalman Filter", kalman_predict)]:
            pred = fn(Xte_xy[..., :2], HORIZON)
            pred_px, true_px = to_pixels(pred), to_pixels(Yte_xy)
            metrics = ade_fde_rmse(pred_px, true_px)
            for h, m in metrics.items():
                all_rows.append({"Method": name, "Seed": seed, "Horizon": h, **m})
            if seed == 0:
                fixed_seed_test_windows[name] = (pred_px, true_px)

        print(f"Seed {seed} done.")

    return pd.DataFrame(all_rows), fixed_seed_test_windows


def summarize_multiseed(df):
    rows = []
    for (method, horizon), g in df.groupby(["Method", "Horizon"]):
        rows.append({
            "Method": method, "Horizon": horizon,
            "ADE_mean": g["ADE"].mean(), "ADE_std": g["ADE"].std(),
            "FDE_mean": g["FDE"].mean(), "FDE_std": g["FDE"].std(),
            "RMSE_mean": g["RMSE"].mean(), "RMSE_std": g["RMSE"].std(),
            "n_seeds": len(g),
        })
    return pd.DataFrame(rows).sort_values(["Horizon", "ADE_mean"])


# ---------------------------------------------------------------------------
# Step 3: significance testing (seed 0, paired on same test windows)
# ---------------------------------------------------------------------------
def significance_tests(fixed_windows, best_model_tag="Spatial LSTM", horizon=10):
    if best_model_tag not in fixed_windows:
        return pd.DataFrame()
    pred_a, true_a = fixed_windows[best_model_tag]
    ade_a = per_window_ade(pred_a, true_a, horizon)
    rows = []
    for baseline in ["Constant Velocity", "Linear Regression", "Kalman Filter"]:
        if baseline not in fixed_windows:
            continue
        pred_b, true_b = fixed_windows[baseline]
        n = min(len(ade_a), len(pred_b))
        ade_b = per_window_ade(pred_b[:n], true_b[:n], horizon)
        res = paired_test(ade_a[:n], ade_b[:n])
        res.update({"Model": best_model_tag, "Baseline": baseline, "Horizon": horizon})
        rows.append(res)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: leave-one-scene-out generalization
# ---------------------------------------------------------------------------
def leave_one_scene_out(hidden, lr, horizon=5):
    df = pd.read_csv(DATA_PATH)
    df["x_n"] = df["x"] / IMG_W
    df["y_n"] = df["y"] / IMG_H
    scenes = df["scene"].unique()
    rows = []
    for held_out in scenes:
        train_scenes_df = df[df.scene != held_out]
        test_scene_df = df[df.scene == held_out]
        traj_ids = train_scenes_df.trajectory_id.unique()
        rng = np.random.default_rng(0)
        rng.shuffle(traj_ids)
        n_val = max(1, int(len(traj_ids) * 0.15))
        val_ids = set(traj_ids[:n_val])
        train_ids = set(traj_ids[n_val:])

        Xtr, Ytr = make_windows(train_scenes_df[train_scenes_df.trajectory_id.isin(train_ids)], spatial_only=True)
        Xva, Yva = make_windows(train_scenes_df[train_scenes_df.trajectory_id.isin(val_ids)], spatial_only=True)
        Xte, Yte = make_windows(test_scene_df, spatial_only=True)
        if len(Xtr) == 0 or len(Xte) == 0 or len(Xva) == 0:
            rows.append({"held_out_scene": held_out, "n_test_windows": len(Xte), "ADE": np.nan, "note": "insufficient data"})
            continue
        model = build_model("lstm", in_features=2, horizon=HORIZON, hidden=hidden)
        model, _ = train_model(model, Xtr, Ytr, Xva, Yva, epochs=40, lr=lr, seed=0)
        pred = model(torch.tensor(Xte)).detach().numpy()
        pred_px, true_px = to_pixels(pred), to_pixels(Yte)
        m = ade_fde_rmse(pred_px, true_px, horizons=(horizon,))[horizon]
        rows.append({"held_out_scene": held_out, "n_test_windows": len(Xte), **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 5: REAL end-to-end latency (actual OpenCV decode of real SIDD frames)
# ---------------------------------------------------------------------------
def real_end_to_end_latency(model, hist=HIST, n_trials=100):
    img_paths = glob.glob(SIDD_IMAGES_GLOB)[:hist * 3]
    if len(img_paths) < hist:
        img_paths = (img_paths * (hist // max(1, len(img_paths)) + 1))[:hist]

    def pipeline_once():
        frames = []
        for p in img_paths[:hist]:
            img = cv2.imread(p)                       # real disk read + decode
            img = cv2.resize(img, (IMG_W, IMG_H))      # real resize
            frames.append(img)
        # real feature stand-in: normalized dummy (x,y) since we don't re-run
        # detection here -- decode/resize cost is the real, measured part
        feats = np.random.rand(1, hist, model.rnn.input_size).astype(np.float32)
        with torch.no_grad():
            model(torch.tensor(feats))

    model.eval()
    for _ in range(5):
        pipeline_once()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        pipeline_once()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return {"mean_ms": times.mean(), "median_ms": np.median(times),
            "std_ms": times.std(), "p95_ms": np.percentile(times, 95),
            "fps": 1000.0 / times.mean(), "n_real_frames_used": min(hist, len(img_paths))}


def model_only_latency(model, X_sample, n_trials=200):
    model.eval()
    x = torch.tensor(X_sample[:1])
    with torch.no_grad():
        for _ in range(20):
            model(x)
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return {"mean_ms": times.mean(), "median_ms": np.median(times),
            "std_ms": times.std(), "p95_ms": np.percentile(times, 95),
            "fps": 1000.0 / times.mean()}


def main():
    out = "/home/claude/uav_traj/outputs"

    print("=== Step 1: Hyperparameter search (Spatial LSTM, seed 0) ===")
    hidden, lr, hp_df = hyperparam_search()
    hp_df.to_csv(f"{out}/hyperparam_search.csv", index=False)

    print("\n=== Step 2: Multi-seed comparison (5 seeds x 4 models + 3 baselines) ===")
    raw_df, fixed_windows = run_multiseed(hidden, lr)
    raw_df.to_csv(f"{out}/multiseed_raw_results.csv", index=False)
    summary_df = summarize_multiseed(raw_df)
    summary_df.to_csv(f"{out}/multiseed_summary_mean_std.csv", index=False)
    print(summary_df[summary_df.Horizon == 10][["Method", "ADE_mean", "ADE_std"]].to_string(index=False))

    print("\n=== Step 3: Significance testing (Spatial LSTM vs each baseline, horizon=10, seed 0) ===")
    sig_df = significance_tests(fixed_windows, "Spatial LSTM", horizon=10)
    sig_df.to_csv(f"{out}/significance_tests.csv", index=False)
    print(sig_df[["Baseline", "mean_diff", "t_pvalue", "wilcoxon_pvalue"]].round(4).to_string(index=False))

    print("\n=== Step 4: Leave-one-scene-out generalization (horizon=5) ===")
    loso_df = leave_one_scene_out(hidden, lr, horizon=5)
    loso_df.to_csv(f"{out}/leave_one_scene_out.csv", index=False)
    print(loso_df.to_string(index=False))

    print("\n=== Step 5: REAL end-to-end latency (best model = Spatial LSTM) ===")
    train_df, val_df, test_df = load_and_split(DATA_PATH, seed=0)
    Xtr, Ytr = make_windows(train_df, spatial_only=True)
    Xva, Yva = make_windows(val_df, spatial_only=True)
    Xte, _ = make_windows(test_df, spatial_only=True)
    final_model = build_model("lstm", in_features=2, horizon=HORIZON, hidden=hidden)
    final_model, _ = train_model(final_model, Xtr, Ytr, Xva, Yva, epochs=40, lr=lr, seed=0)
    model_lat = model_only_latency(final_model, Xte)
    e2e_lat = real_end_to_end_latency(final_model)
    latency_df = pd.DataFrame([
        {"stage": "model_only", **model_lat},
        {"stage": "end_to_end_real_decode", **e2e_lat},
    ])
    latency_df.to_csv(f"{out}/real_latency_report.csv", index=False)
    print(latency_df.round(3).to_string(index=False))

    print(f"\nAll outputs written to {out}/")


if __name__ == "__main__":
    main()
