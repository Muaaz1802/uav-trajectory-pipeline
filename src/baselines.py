"""
baselines.py -- Constant Velocity and Linear Regression extrapolation (Section 5).
Both operate on the (x, y) history only, regardless of whether angular features
are present in X, since neither baseline is meant to use them.
"""
import numpy as np
from sklearn.linear_model import LinearRegression


def constant_velocity_predict(X_xy, horizon):
    """X_xy: (N, hist, 2). Uses last velocity, extrapolated linearly."""
    last = X_xy[:, -1, :]
    v = X_xy[:, -1, :] - X_xy[:, -2, :]
    steps = np.arange(1, horizon + 1).reshape(1, -1, 1)
    return last[:, None, :] + v[:, None, :] * steps


def linear_regression_predict(X_xy, horizon):
    """Fits a per-sample linear trend over the history window, per coordinate."""
    N, hist, _ = X_xy.shape
    t_hist = np.arange(hist).reshape(-1, 1)
    t_future = np.arange(hist, hist + horizon).reshape(-1, 1)
    preds = np.zeros((N, horizon, 2), dtype=np.float32)
    for i in range(N):
        for c in range(2):
            model = LinearRegression().fit(t_hist, X_xy[i, :, c])
            preds[i, :, c] = model.predict(t_future)
    return preds
