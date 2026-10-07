"""
run_experiment.py
------------------
End-to-end experiment matching Sections 5, 7 and RQ2-RQ4 of the synopsis:
  1. Train Spatial-only LSTM (Model A) and Spatial+Angular LSTM (Model B)
  2. Compare vs Constant-Velocity and Linear-Regression baselines
  3. Report ADE / FDE / RMSE at horizons 1, 3, 5, 10
  4. Angular-feature ablation (Model A vs Model B) -> RQ2
  5. Prediction-model latency vs simulated end-to-end latency, P95, FPS -> RQ4
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import load_and_split, make_windows, HIST, HORIZON, IMG_W, IMG_H
from baselines import constant_velocity_predict, linear_regression_predict
from models import LightweightLSTM

torch.manual_seed(42)
DEVICE = "cpu"  # "lightweight" + reproducible on any laptop, per synopsis intent


def to_pixels(arr_norm):
    out = arr_norm.copy()
    out[..., 0] *= IMG_W
    out[..., 1] *= IMG_H
    return out


def ade_fde_rmse(pred_px, true_px, horizons=(1, 3, 5, 10)):
    """pred_px, true_px: (N, HORIZON, 2) in pixel space."""
    results = {}
    err = np.linalg.norm(pred_px - true_px, axis=-1)  # (N, HORIZON)
    for h in horizons:
        h = min(h, err.shape[1])
        ade = err[:, :h].mean()
        fde = err[:, h - 1].mean()
        rmse = np.sqrt(((pred_px[:, :h] - true_px[:, :h]) ** 2).mean())
        results[h] = {"ADE": ade, "FDE": fde, "RMSE": rmse}
    return results


def train_lstm(model, X_train, Y_train, X_val, Y_val, epochs=40, lr=1e-3, batch=64):
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
            pred = model(Xt[idx])
            loss = loss_fn(pred, Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), Yv).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def measure_latency(model, X_sample, n_trials=200):
    """Prediction-model latency: preprocessed features -> trajectory."""
    model.eval()
    x = torch.tensor(X_sample[:1])
    times = []
    with torch.no_grad():
        for _ in range(20):  # warmup
            model(x)
        for _ in range(n_trials):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000  # ms
    return {
        "mean_ms": times.mean(), "median_ms": np.median(times),
        "std_ms": times.std(), "p95_ms": np.percentile(times, 95),
        "fps": 1000.0 / times.mean(),
    }


def measure_end_to_end_latency(model, hist=HIST, n_trials=200):
    """Full pipeline: simulated frame read -> feature extraction -> normalize -> LSTM.
    Frame extraction / bbox-to-(x,y) is simulated here (no real video decode in this
    demo run); swap in real OpenCV frame reads once real footage is wired in."""
    model.eval()
    times = []
    for _ in range(20):
        _simulate_frame_pipeline(model, hist)  # warmup
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _simulate_frame_pipeline(model, hist)
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    return {
        "mean_ms": times.mean(), "median_ms": np.median(times),
        "std_ms": times.std(), "p95_ms": np.percentile(times, 95),
        "fps": 1000.0 / times.mean(),
    }


def _simulate_frame_pipeline(model, hist):
    # frame read + bbox->(x,y) extraction stand-in (replace with real OpenCV/YOLO calls)
    raw = np.random.rand(hist, 640, 512, 3).astype(np.float32) * 0  # decode cost proxy
    feats = np.random.rand(1, hist, model.lstm.input_size).astype(np.float32)
    with torch.no_grad():
        model(torch.tensor(feats))


def main(data_path="/home/claude/uav_traj/data/trajectories_real.csv"):
    print(f"Loading + splitting data (trajectory-level split) from {data_path} ...")
    train_df, val_df, test_df = load_and_split(data_path)
    print(f"Train/Val/Test trajectories: {train_df.trajectory_id.nunique()}/"
          f"{val_df.trajectory_id.nunique()}/{test_df.trajectory_id.nunique()}")

    results_rows = []
    latency_rows = []

    for spatial_only, tag in [(True, "Spatial-only (Model A)"), (False, "Spatial+Angular (Model B)")]:
        Xtr, Ytr = make_windows(train_df, spatial_only=spatial_only)
        Xva, Yva = make_windows(val_df, spatial_only=spatial_only)
        Xte, Yte = make_windows(test_df, spatial_only=spatial_only)

        n_feat = Xtr.shape[-1]
        model = LightweightLSTM(in_features=n_feat, horizon=HORIZON).to(DEVICE)
        print(f"\nTraining {tag} | params={model.n_params()} | train windows={len(Xtr)}")
        model = train_lstm(model, Xtr, Ytr, Xva, Yva)

        pred = model(torch.tensor(Xte)).detach().numpy()
        pred_px, true_px = to_pixels(pred), to_pixels(Yte)
        metrics = ade_fde_rmse(pred_px, true_px)
        for h, m in metrics.items():
            results_rows.append({"Method": tag, "Horizon": h, **m})

        lat = measure_latency(model, Xte)
        e2e = measure_end_to_end_latency(model)
        latency_rows.append({"Method": tag, "n_params": model.n_params(),
                              **{f"model_{k}": v for k, v in lat.items()},
                              **{f"e2e_{k}": v for k, v in e2e.items()}})

    # Baselines: always spatial (x, y) history only
    Xte_xy, Yte_xy = make_windows(test_df, spatial_only=True)
    for name, fn in [("Constant Velocity", constant_velocity_predict),
                      ("Linear Regression", linear_regression_predict)]:
        pred = fn(Xte_xy[..., :2], HORIZON)
        pred_px, true_px = to_pixels(pred), to_pixels(Yte_xy)
        metrics = ade_fde_rmse(pred_px, true_px)
        for h, m in metrics.items():
            results_rows.append({"Method": name, "Horizon": h, **m})

    results_df = pd.DataFrame(results_rows)
    latency_df = pd.DataFrame(latency_rows)
    suffix = "real" if "real" in data_path else "sim"
    results_df.to_csv(f"/home/claude/uav_traj/outputs/results_ade_fde_rmse_{suffix}.csv", index=False)
    latency_df.to_csv(f"/home/claude/uav_traj/outputs/latency_report_{suffix}.csv", index=False)

    print("\n=== ADE / FDE / RMSE (pixels, 640x512 image space) ===")
    print(results_df.pivot_table(index="Method", columns="Horizon", values="ADE").round(2))
    print("\n=== Latency (ms) ===")
    print(latency_df[["Method", "n_params", "model_mean_ms", "model_p95_ms", "model_fps",
                       "e2e_mean_ms", "e2e_p95_ms", "e2e_fps"]].round(3))

    # Angular ablation summary (RQ2)
    a = results_df[results_df.Method.str.contains("Model A")].set_index("Horizon")["ADE"]
    b = results_df[results_df.Method.str.contains("Model B")].set_index("Horizon")["ADE"]
    print("\n=== RQ2: Angular ablation (ADE, lower=better) ===")
    print(pd.DataFrame({"Spatial-only": a, "Spatial+Angular": b,
                         "Improvement %": (a - b) / a * 100}).round(2))


if __name__ == "__main__":
    main()
