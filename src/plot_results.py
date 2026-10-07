import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

suffix = sys.argv[1] if len(sys.argv) > 1 else "real"
df = pd.read_csv(f"/home/claude/uav_traj/outputs/results_ade_fde_rmse_{suffix}.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for method, g in df.groupby("Method"):
    g = g.sort_values("Horizon")
    axes[0].plot(g["Horizon"], g["ADE"], marker="o", label=method)
    axes[1].plot(g["Horizon"], g["FDE"], marker="o", label=method)

axes[0].set_title(f"ADE vs Prediction Horizon ({suffix} data)"); axes[0].set_xlabel("Horizon (steps)")
axes[0].set_ylabel("ADE (pixels)"); axes[0].grid(alpha=0.3)
axes[1].set_title(f"FDE vs Prediction Horizon ({suffix} data)"); axes[1].set_xlabel("Horizon (steps)")
axes[1].set_ylabel("FDE (pixels)"); axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8, loc="upper left")
plt.tight_layout()
out = f"/home/claude/uav_traj/outputs/horizon_error_curves_{suffix}.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
