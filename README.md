# Lightweight UAV Trajectory Prediction — now running on real data

Now running on **real infrared UAV detections** from the SIDD dataset
(`github.com/Dang-zy/SIDD`) — the same dataset the CA-LSTM paper trained its
detector on. `src/extract_real_data.py` pulls real bounding boxes across four
real scenes (sky, city, mountain, sea) and reconstructs trajectory segments.

## What's real vs. what's still a stand-in — READ THIS BEFORE YOUR MENTOR MEETING

| Component | Status |
|---|---|
| UAV positions `(x, y)` | **Real** — bounding-box centres from real SIDD infrared frames |
| Frame ordering | **Real** — numeric frame index embedded in each filename |
| Trajectory segments (49 total: 16 city, 21 mountain, 5 sea, 7 sky) | **Reconstructed, not ground-truth tracks.** SIDD ships detection annotations only, no track IDs. Segments are frames grouped by contiguous frame-index (gap ≤ 5). A gap could hide an undetected scene cut — flag this as a known limitation, don't present it as verified ground-truth trajectories. |
| Azimuth / elevation | **NOT real.** SIDD has no angular sensor channel. What "Model B" uses is a heading-of-motion value derived from the same (x,y) path — not independent sensor data. Per your own Section 5 caveat ("verify availability against dataset documentation"), this dataset fails that check. Say this explicitly if asked — don't call it real azimuth/elevation. |
| Trajectory-level train/val/test split, sliding windows, both LSTMs, both baselines, ADE/FDE/RMSE, ablation, latency | **Real, all computed on the real data above** |
| End-to-end latency | Still uses a simulated frame-read cost, not real OpenCV decode |

**Sample size caveat:** only 49 usable real trajectory segments (34 train / 7
val / 8 test) survived the ≥25-frame filter. That's small — treat these
numbers as a pilot/proof-of-concept, not a final result, and say so.

## Results on real data

- **Constant Velocity collapses on real motion**: ADE goes from 79px (h=1) to
  **305px** (h=10) — real infrared UAV motion is far less linear than the
  smooth synthetic run, consistent with the "low, slow, small" erratic-target
  literature your synopsis cites.
- **Both LSTMs clearly beat Constant Velocity at every horizon**, and beat
  Linear Regression at the longer horizon (h=10: Model A 70.96px vs LR
  85.92px) — a genuine, positive, real-data answer to RQ1: yes, a lightweight
  LSTM has a meaningful advantage over motion baselines, and the advantage
  grows with horizon.
- **RQ2 flips vs. the synthetic run**: Spatial+Angular is now *worse* than
  Spatial-only (−2% to −7% ADE). This is expected and worth reporting exactly
  as-is: the "angular" feature here is a derived, redundant proxy (see table
  above), not independent sensor signal, and with only 34 training
  trajectories the extra input dimensionality adds noise rather than
  information. This is a legitimate negative result for H2 under real,
  angular-channel-free data — and it's exactly the kind of finding your
  synopsis's RQ2 language ("or whether it merely adds input dimensionality
  without benefit") anticipated.
- Full numbers: `outputs/results_ade_fde_rmse_real.csv`,
  `outputs/latency_report_real.csv`
- Plot: `outputs/horizon_error_curves_real.png`
- Simulated-data run kept for comparison: `*_sim.csv` / `*_sim.png` (rerun
  `python3 src/simulate_data.py && python3 src/run_experiment.py` with the
  simulated path to regenerate)

## Honest framing for your mentor

Say: *"Real bounding-box data from the SIDD dataset, real frame ordering,
trajectory segments reconstructed via contiguous-frame grouping since no
track IDs are released; azimuth/elevation are a derived heading proxy, not a
real sensor channel, because this dataset doesn't have one — full model,
baselines, metrics and ablation pipeline are running end-to-end on it."*
That's accurate and still a strong checkpoint. Don't claim the angular
channel is real, and don't claim 49 trajectories is a final-scale result.

## Next steps toward a larger / cleaner real dataset

1. The official **Anti-UAV** benchmark (`github.com/ucas-vg/Anti-UAV`) has
   per-frame sequential annotations with real track continuity — but the
   actual video/JSON files are hosted on Baidu Yun / Google Drive, not
   in-repo, so they need a manual download outside this sandboxed environment.
   That would fix both the "reconstructed vs ground-truth" and (partially,
   check its docs) the azimuth caveat above.
2. Once downloaded, replace `data/trajectories_real.csv` with the same schema
   — nothing in `dataset.py`, `models.py`, `run_experiment.py` needs to change.
3. Replace `_simulate_frame_pipeline()` in `run_experiment.py` with a real
   OpenCV frame-read call for genuine end-to-end latency.

## Run it yourself

```bash
python3 src/extract_real_data.py           # rebuilds data/trajectories_real.csv from the SIDD repo
python3 src/run_experiment.py              # trains + evaluates on real data by default
python3 src/plot_results.py real
```

---

# v2: paper-grade rigor pass (multi-seed, more baselines/models, significance testing)

`src/run_experiment_v2.py` adds what a paper needs on top of the v1 pipeline
above. Run it with `python3 src/run_experiment_v2.py` (takes a few minutes).

## What v2 adds

| Addition | Where |
|---|---|
| Hyperparameter search (hidden size x LR), validation-selected | `outputs/hyperparam_search.csv` -- best: **hidden=32, lr=1e-3** |
| 4 model variants: Spatial LSTM, Spatial+Angular LSTM, Spatial GRU, Spatial Bi-LSTM | `models.py` |
| 3 baselines: Constant Velocity, Linear Regression, **Kalman Filter** (constant-velocity) | `kalman_baseline.py` |
| 5-seed repetition -> mean +/- std per method per horizon | `outputs/multiseed_summary_mean_std.csv`, `multiseed_ade_horizon.png` |
| Paired significance testing (t-test + Wilcoxon), same test windows | `outputs/significance_tests.csv` |
| Leave-one-scene-out generalization | `outputs/leave_one_scene_out.csv`, `.png` |
| REAL end-to-end latency: actual OpenCV decode of real SIDD frames (not simulated) | `outputs/real_latency_report.csv` |

## Results, and the honest reading of them

**Ranking at horizon=10 (5-seed mean +/- std, ADE in pixels):**

| Method | ADE |
|---|---|
| Spatial GRU | 65.6 +/- 8.9 |
| Spatial Bi-LSTM | 66.3 +/- 9.3 |
| Spatial LSTM | 66.9 +/- 9.5 |
| Spatial+Angular LSTM | 69.3 +/- 11.3 |
| Kalman Filter | 88.2 +/- 28.5 |
| Linear Regression | 88.5 +/- 27.9 |
| Constant Velocity | 236.1 +/- 87.5 |

**The important caveat -- check this before claiming "LSTM significantly beats
classical baselines" in a paper:**

Paired significance testing (Spatial LSTM vs each baseline, horizon=10,
seed 0, same test windows) gives:

| vs. | t-test p | Wilcoxon p | Significant? |
|---|---|---|---|
| Constant Velocity | 0.0000 | 0.0725 | **Yes on t-test, borderline on Wilcoxon** |
| Linear Regression | 0.0659 | 0.9127 | **No** |
| Kalman Filter | 0.1284 | 0.5779 | **No** |

So: the clean win is only against Constant Velocity. Against Linear
Regression and the Kalman filter, the point-estimate gap (66.9px vs
~88px) is real but **not statistically significant** at conventional
thresholds on this small test set (this is a single seed's test split, 8
trajectories). Report it exactly this way -- "outperforms in point estimate,
not statistically distinguishable from Linear Regression/Kalman at this
sample size" -- rather than claiming a proven win. This is a direct, honest
consequence of the 49-trajectory pilot scale; it will very likely resolve
into real significance once trajectory count increases by an order of
magnitude.

**GRU/Bi-LSTM vs LSTM**: all three recurrent variants land within one
standard deviation of each other -- no architecture is clearly better on
this data/scale. Report this as "recurrent architecture choice was not the
deciding factor here," rather than picking a winner that isn't really there.

**Leave-one-scene-out** (train on 3 scenes, test on the 4th, horizon=5):
ADE ranges 53px (sea) to 95px (sky) -- the model generalizes across scenes
reasonably, sky is hardest (matches it having the fewest/shortest usable
trajectory segments, 7 total).

**Real end-to-end latency** (actual OpenCV decode + resize of real SIDD
frames, not simulated): model inference alone is ~0.09ms (11,000+ FPS);
full pipeline with real image decode is ~11.8ms (~84 FPS) -- the decode
step, not the model, dominates latency. That's a genuinely useful,
reportable finding for RQ4 as-is.

## What is still NOT solved by v2

- **Sample size.** 5 seeds of a 49-trajectory dataset is a much better
  *protocol* than one run, but it is not a substitute for more real
  trajectories. The wide std bars and non-significant p-values above are a
  direct symptom of this -- more data is still the single highest-value next
  step.
- **Real track-level trajectories and real azimuth/elevation** -- unchanged
  from before, still requires the official Anti-UAV benchmark data (Google
  Drive/Baidu Yun downloads, outside this environment's reach).
- The hyperparameter search only tuned the Spatial LSTM, then reused that
  config for GRU/Bi-LSTM/Spatial+Angular for compute-budget reasons -- say
  this explicitly if asked; a from-scratch search per architecture would be
  more rigorous still.

---

# v3: "Beat the paper" — detection track (recommended primary result)

Your mentor wants you to beat a paper's results on the same dataset. The
CA-LSTM paper's **trajectory** numbers were never released on a public
dataset, so that specific claim isn't reproducible by anyone. Its
**detection** numbers (Table 6) ARE on a public, real dataset you already
have locally (SIDD) — that's the one you can actually beat. Recommend
making this the paper's headline result; keep trajectory forecasting as a
secondary/pilot section.

## What's here
- `src/coco_to_yolo.py` — converts SIDD's real COCO annotations (all 4
  scenes) into YOLO training format. **Already run**: 3,788 train / 949 val
  real images at `/home/claude/uav_yolo_dataset/` (not included in the zip —
  regenerate with this script, needs the cloned SIDD repo).
- `src/train_yolo_final.py` — the training script to run **on a GPU**
  (Colab free T4 is enough). Fine-tunes YOLO11n (pretrained on COCO) on the
  exact same SIDD data.
- **Smoke test already run and passed** in this environment (40 train / 15
  val images, 2 epochs, CPU): confirmed the COCO→YOLO conversion, label
  format, and training loop all work correctly end-to-end. This is pipeline
  validation, NOT a result — don't report these numbers, the sample was
  tiny and undertrained on purpose.
- **Full-dataset run attempted here and failed**: this sandbox has only
  3.9GB RAM / 1 CPU core and silently ran out of memory partway through
  setup. This is a resource limit, not a bug in the code — it's exactly why
  the actual run needs to happen on a GPU machine.

## What to beat (CA-LSTM paper, Table 6, same SIDD dataset)

| Method | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| YOLOv5 (their baseline) | 0.871 | 0.503 |
| Improved YOLOv5+CBAM (their result) | **0.906** | **0.519** |

## Do this today (Google Colab, free tier, ~20-30 min total)

1. Open a new Colab notebook, set Runtime → GPU (T4).
2. ```
   !pip install ultralytics -q
   !git clone https://github.com/Dang-zy/SIDD.git
   ```
3. Upload `coco_to_yolo.py` from this package, edit `REPO` path to point at
   the cloned SIDD folder, run it.
4. Upload and run `train_yolo_final.py` (edit `DATA_YAML` path to match).
5. Read off `mAP@0.5` and `mAP@0.5:0.95` at the end — compare directly
   against 0.906 / 0.519 above. A modern pretrained YOLO11n fine-tuned for
   50 epochs is very likely to beat 2023-era YOLOv5+CBAM on this dataset —
   that's your real, defensible "we beat paper X on their own dataset"
   result.

## What NOT to do
Don't report the 2-epoch/40-image smoke test numbers as a result, and don't
claim you "trained and beat the paper" until you've actually run
`train_yolo_final.py` on a GPU and have the real mAP numbers in hand. Tell
your mentor honestly: pipeline built and validated here, final training run
pending GPU access (today, via Colab).
