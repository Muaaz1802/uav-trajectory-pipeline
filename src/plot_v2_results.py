import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = "/home/claude/uav_traj/outputs"
summary = pd.read_csv(f"{out}/multiseed_summary_mean_std.csv")
loso = pd.read_csv(f"{out}/leave_one_scene_out.csv")

# --- Plot 1: ADE mean+-std vs horizon, all methods ---
fig, ax = plt.subplots(figsize=(8, 5))
for method, g in summary.groupby("Method"):
    g = g.sort_values("Horizon")
    ax.errorbar(g["Horizon"], g["ADE_mean"], yerr=g["ADE_std"], marker="o",
                capsize=3, label=method)
ax.set_xlabel("Prediction horizon (steps)")
ax.set_ylabel("ADE (pixels), mean +/- std over 5 seeds")
ax.set_title("ADE vs horizon, 5-seed mean +/- std (real SIDD-derived trajectories)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out}/multiseed_ade_horizon.png", dpi=150)
print("Saved multiseed_ade_horizon.png")

# --- Plot 2: leave-one-scene-out bar chart ---
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(loso["held_out_scene"], loso["ADE"], color="#4C72B0")
ax.set_ylabel("ADE (pixels), horizon=5")
ax.set_xlabel("Held-out scene (trained on the other 3)")
ax.set_title("Leave-one-scene-out generalization (Spatial LSTM)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{out}/leave_one_scene_out.png", dpi=150)
print("Saved leave_one_scene_out.png")
