"""
stats_utils.py -- paired significance testing between methods on the SAME
test windows (so pairing is valid), plus multi-seed aggregation.
"""
import numpy as np
from scipy import stats


def paired_test(err_a, err_b):
    """err_a, err_b: (N,) per-window errors (e.g. per-window ADE) from the SAME
    test windows for two methods. Returns dict with paired t-test AND Wilcoxon
    (more robust for small/non-normal samples, which a 49-trajectory pilot is)."""
    t_stat, t_p = stats.ttest_rel(err_a, err_b)
    try:
        w_stat, w_p = stats.wilcoxon(err_a, err_b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    return {"t_stat": t_stat, "t_pvalue": t_p, "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
            "mean_diff": np.mean(err_a - err_b)}


def aggregate_seeds(per_seed_metrics):
    """per_seed_metrics: list of dicts {metric_name: value}. Returns {metric: (mean, std)}."""
    keys = per_seed_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics])
        out[k] = (vals.mean(), vals.std())
    return out
