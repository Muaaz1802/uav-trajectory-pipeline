"""
kalman_baseline.py -- constant-velocity Kalman filter baseline.

A standard, reviewer-expected comparison point for trajectory forecasting
(state = [x, y, vx, vy], linear constant-velocity motion model, run forward
open-loop for the prediction horizon after filtering the history window).
Not in your original synopsis baseline list, but its absence is exactly the
kind of thing a reviewer flags -- CV/Linear-Regression alone read as "the
easy baselines were beaten," a Kalman filter is the one that's harder to beat
and makes the comparison credible.
"""
import numpy as np


def _kalman_filter_1traj(xy_hist, horizon, process_var=1e-2, meas_var=4.0):
    """xy_hist: (hist, 2) pixel positions. Returns (horizon, 2) predicted positions."""
    dt = 1.0
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    Q = process_var * np.eye(4)
    R = meas_var * np.eye(2)

    v0 = xy_hist[1] - xy_hist[0]
    state = np.array([xy_hist[0, 0], xy_hist[0, 1], v0[0], v0[1]])
    P = np.eye(4) * 10.0

    for t in range(1, len(xy_hist)):
        # predict
        state = F @ state
        P = F @ P @ F.T + Q
        # update
        z = xy_hist[t]
        y = z - H @ state
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        state = state + K @ y
        P = (np.eye(4) - K @ H) @ P

    # forecast forward open-loop (no further measurements) for `horizon` steps
    preds = []
    s = state.copy()
    for _ in range(horizon):
        s = F @ s
        preds.append(s[:2].copy())
    return np.array(preds)


def kalman_predict(X_xy, horizon):
    """X_xy: (N, hist, 2) -> (N, horizon, 2)"""
    N = X_xy.shape[0]
    out = np.zeros((N, horizon, 2), dtype=np.float32)
    for i in range(N):
        out[i] = _kalman_filter_1traj(X_xy[i], horizon)
    return out
