# KARMA

**K-order Approximation via Markov chains for Retrospective Attribution**

KARMA is a model-agnostic framework for retrospective causal attribution in multivariate time series. It recovers a sparse lag-order DAG over the variables by building a Markov transition kernel from a pre-trained oracle, then ranking edges by their kernel TV contribution (ρ).

---

## Reproducing Paper Results

All tables and figures in the paper come from three experiment scripts plus the KARMA pipeline itself. The steps below reproduce them end-to-end.

### 1 — Install dependencies

```bash
pip install poetry
poetry install
```

Install WinIT (required for baseline comparisons):

```bash
git clone https://github.com/layer6ai-labs/WinIT.git
cd WinIT && pip install -e . && cd ..
```

### 2 — Prepare data

Raw datasets must be placed under `data/raw/<dataset>/` before generating `.npy` splits. The pipeline expects pre-processed windows at `data/generated/<dataset>/X_train.npy` and `X_val.npy`.

### 3 — Train oracles (LSTM / TCN)

Checkpoints are already provided under `outputs/checkpoints/`. To retrain from scratch:

```bash
# Single dataset + architecture
python -m pipeline.training_pipeline --dataset exchange_rate --model lstm

# All datasets, both architectures
python -m pipeline.training_pipeline --dataset all --model both --epochs 50
```

Checkpoints are saved to the path specified in `configs/datasets/<dataset>.yaml` (e.g. `outputs/checkpoints/exchange_rate_lstm/best.pt`).

---

### 4 — Run KARMA (main paper results)

The KARMA pipeline runs all three pillars (discretisation → K*/b* selection → kernel estimation + DAG recovery) and writes results to `results/<dataset>/<model>/`.

**Single dataset**

```bash
python -m pipeline.karma_pipeline --dataset exchange_rate --model lstm
python -m pipeline.karma_pipeline --dataset weather --model tcn
```

**All datasets in the paper (Table 2 / Table 3)**

```bash
python -m pipeline.karma_pipeline --dataset all --model both
```

Key CLI flags (all have defaults in `configs/experiments/karma.yaml`):

| Flag           | Default | Description                                                           |
| -------------- | ------- | --------------------------------------------------------------------- |
| `--N`          | 3       | Bins per variable for discretisation                                  |
| `--eps`        | 0.05    | Δ^pred stopping tolerance (Pillar 2 certificate)                      |
| `--lam`        | 0.025   | Edge-trimming threshold ρ for DAG recovery                            |
| `--M`          | 100     | Monte Carlo draws per history (Pillar 3)                              |
| `--K_max`      | 10      | Maximum lag order to search                                           |
| `--seed`       | 42      | Random seed                                                           |
| `--mega_batch` | off     | Collect all H×M windows into one GPU batch (faster on large datasets) |
| `--plot_fmt`   | `both`  | Save figures as `png`, `pdf`, or both                                 |

**Output files per run** (written to `results/<dataset>/<model>/`):

| File                    | Contents                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| `karma_results.json`    | Full results dict: config, Pillar 2 (K\*, b\*, Δ^pred), Pillar 3 (ρ matrix, VI scores, retained edges) |
| `karma_dag.csv`         | Retained edges sorted by ρ — used directly for Table 2/3                                               |
| `karma_convergence.csv` | Δ^pred vs lag-order convergence curve                                                                  |
| `karma_kernels.csv`     | Sample transition kernel rows                                                                          |
| `figures/`              | DAG heatmaps and convergence plots                                                                     |

---

### 5 — Synthetic VAR experiment (Figure 3 / Table 1)

Runs KARMA and three baselines (FO, DynaMask, WinIT) on a known VAR(3) process with analytical ground truth. Reports Kendall's τ vs G\*.

```bash
# Standard D=4, K=3 setting
python -m experiments.comparison_var

# Skip slow baselines
python -m experiments.comparison_var --skip_dynamask --skip_winit

# Custom seed / training length
python -m experiments.comparison_var --T_train 5000 --seed 42
```

Results are written to `results/var/`.

**Multi-scale VAR experiment** (scaling to larger D and K, Appendix):

```bash
python -m experiments.comparison_var_multi
python -m experiments.comparison_var_multi --configs tiny small medium large xlarge
```

Results are written to `results/comparison_var_multi/`.

---

### 6 — Real-data AUC comparison (Figure 4)

Runs KARMA and baselines (FO, DynaMask, WinIT, IG, TimeShap) on real forecasting datasets. The metric is edge-removal AUC (prediction change as top-ρ edges are progressively masked).

```bash
# All datasets
python -m experiments.comparison_realdata --load_existing

# Specific datasets only
python -m experiments.comparison_realdata --datasets etth1 exchange_rate beijing_pm25 --load_existing

# Skip specific baselines
python -m experiments.comparison_realdata --skip_karma --skip_winit --load_existing
```

Results are written to `results/realdata/<dataset>_results.json`.

---

### 7 — Plot lag-AUC curves (Figure 4 panels)

After running `comparison_realdata`, generate the multi-panel PDF used in the paper:

```bash
python -m experiments.plot_lag_auc_curves
```

Output: `results/realdata/lag_auc_curves.pdf` (and `.png`).

---

## Project layout

```
configs/
  datasets/       # per-dataset YAML (paths, var names, window size)
  experiments/    # karma.yaml  — default hyperparameters
  models/         # lstm.yaml, tcn.yaml

pipeline/
  training_pipeline.py   # train LSTM/TCN oracles
  karma_pipeline.py      # full KARMA run (entry point)
  visualization.py       # DAG heatmaps, convergence plots

experiments/
  comparison_var.py          # Synthetic VAR(3) vs baselines
  comparison_var_multi.py    # Multi-scale VAR sweep
  comparison_realdata.py     # Real-data AUC comparison
  plot_lag_auc_curves.py     # Figure 4 plotting script

karma/
  markov_approximation/      # Pillar 2: K* / b* selection
  causal_recovery/           # Pillar 3: edge contributions
  utils/                     # Discretiser, SuffixPool, BStarKernelEstimator

outputs/checkpoints/         # Pre-trained oracle checkpoints (best.pt)
results/                     # All experiment outputs
data/generated/              # Pre-processed .npy splits
```

---

## Quick reference: paper → script mapping

| Paper element                        | Script / command                                                                             |
| ------------------------------------ | -------------------------------------------------------------------------------------------- |
| Table 1 — VAR(3) Kendall's τ         | `python -m experiments.comparison_var`                                                       |
| Table 2/3 — Real-data DAG edges & VI | `python -m pipeline.karma_pipeline --dataset all --model both`                               |
| Figure 3 — K\* convergence           | Generated automatically by `karma_pipeline` → `figures/`                                     |
| Figure 4 — Lag-AUC removal curves    | `python -m experiments.comparison_realdata` then `python -m experiments.plot_lag_auc_curves` |
| Appendix — multi-scale VAR           | `python -m experiments.comparison_var_multi`                                                 |
