# Experiment Run Prompts

All commands are run from the repo root with the virtual environment active.

```bash
source .venv/bin/activate   # or: .venv/bin/python -m ...
```

---

## 1. `comparison_var` — VAR(3) D=4 single dataset

Full comparison of all methods on a synthetic VAR(3) D=4 dataset.
Metric: Kendall's τ vs ground-truth coefficient magnitudes.

### Full run (all methods, defaults)
```bash
python -m experiments.comparison_var
```

### All methods, longer training
```bash
python -m experiments.comparison_var --T_train 10000 --gen_epochs 100 --dyn_epochs 200
```

### Only IG + TimeShap
```bash
python -m experiments.comparison_var --skip_karma --skip_fo --skip_dynamask --skip_winit
```

### Only KARMA
```bash
python -m experiments.comparison_var --skip_fo --skip_dynamask --skip_winit --skip_ig --skip_timeshap
```

### Only FO + DynaMask
```bash
python -m experiments.comparison_var --skip_karma --skip_winit --skip_ig --skip_timeshap
```

### Only WinIT
```bash
python -m experiments.comparison_var --skip_karma --skip_fo --skip_dynamask --skip_ig --skip_timeshap
```

### Load saved WinIT generators (skip retraining)
```bash
python -m experiments.comparison_var --load_generators
```

### Fast smoke test (small data, fewer epochs, few test windows)
```bash
python -m experiments.comparison_var --T_train 1000 --T_test 200 --n_test 20 \
  --dyn_epochs 20 --gen_epochs 10 --ts_nsamples 50 --M 50
```

### Custom TimeShap coalition budget
```bash
python -m experiments.comparison_var --skip_karma --skip_fo --skip_dynamask --skip_winit \
  --ts_nsamples 500
```

### Custom KARMA parameters
```bash
python -m experiments.comparison_var --skip_fo --skip_dynamask --skip_winit --skip_ig --skip_timeshap \
  --N 2 --M 300 --K_max 6 --lam 0.01 --eps 0.03
```

### Custom output directory + seed
```bash
python -m experiments.comparison_var --seed 123 --output_dir results/var_seed123
```

---

## 2. `comparison_var_multi` — VAR scaling across 5 configurations

Runs each of the 5 VAR configurations (tiny → xlarge) with increasing D, K, and T_train.
Metric: Kendall's τ vs G* per configuration, printed as a summary table.

Configurations: `tiny` (D=2,K=1), `small` (D=4,K=2), `medium` (D=4,K=3), `large` (D=6,K=3), `xlarge` (D=8,K=4).

### Full run (all configs, all methods)
```bash
python -m experiments.comparison_var_multi
```

### All methods, specific configs only
```bash
python -m experiments.comparison_var_multi --configs tiny small medium
```

### Only IG + TimeShap, all configs
```bash
python -m experiments.comparison_var_multi --skip_karma --skip_fo --skip_dynamask --skip_winit
```

### Only IG + TimeShap, specific configs
```bash
python -m experiments.comparison_var_multi --configs tiny small \
  --skip_karma --skip_fo --skip_dynamask --skip_winit
```

### Only KARMA, all configs
```bash
python -m experiments.comparison_var_multi --skip_fo --skip_dynamask --skip_winit --skip_ig --skip_timeshap
```

### Only FO + DynaMask
```bash
python -m experiments.comparison_var_multi --skip_karma --skip_winit --skip_ig --skip_timeshap
```

### Only WinIT
```bash
python -m experiments.comparison_var_multi --skip_karma --skip_fo --skip_dynamask --skip_ig --skip_timeshap
```

### Fast smoke test (small test set, fewer epochs)
```bash
python -m experiments.comparison_var_multi --configs tiny small \
  --n_test 20 --dyn_epochs 20 --gen_epochs 10 --ts_nsamples 50
```

### Verbose KARMA output
```bash
python -m experiments.comparison_var_multi --skip_fo --skip_dynamask --skip_winit --skip_ig --skip_timeshap \
  --verbose
```

### Custom TimeShap budget
```bash
python -m experiments.comparison_var_multi --skip_karma --skip_fo --skip_dynamask --skip_winit \
  --ts_nsamples 400 --configs tiny small medium
```

### Custom output directory
```bash
python -m experiments.comparison_var_multi --output_dir results/var_multi_run2 --seed 99
```

---

## 3. `comparison_realdata` — AUC on real forecasting datasets

Trains a GRU regression model per dataset, runs attribution methods, and measures
prediction-change AUC under progressive feature removal.

Datasets: `etth1` (D=7), `exchange_rate` (D=8), `beijing_pm25` (D=11).

KARMA is automatically skipped for D > 15 (web_traffic, electricity).

### Full run (all datasets, all methods)
```bash
python -m experiments.comparison_realdata
```

### Specific datasets only
```bash
python -m experiments.comparison_realdata --datasets etth1 exchange_rate beijing_pm25
```

### Only small-D datasets (KARMA runs on all three)
```bash
python -m experiments.comparison_realdata --datasets etth1 exchange_rate beijing_pm25
```

### Only FO + DynaMask (fast, no generator training)
```bash
python -m experiments.comparison_realdata --skip_karma --skip_winit
```

### Only KARMA (small-D datasets)
```bash
python -m experiments.comparison_realdata --datasets etth1 exchange_rate beijing_pm25 \
  --skip_fo --skip_dynamask --skip_winit
```

### Only FO
```bash
python -m experiments.comparison_realdata --skip_karma --skip_dynamask --skip_winit
```

### Only WinIT
```bash
python -m experiments.comparison_realdata --skip_karma --skip_fo --skip_dynamask
```

### Only DynaMask
```bash
python -m experiments.comparison_realdata --skip_karma --skip_fo --skip_winit
```

### Load pre-trained GRU and WinIT generators (skip retraining)
```bash
python -m experiments.comparison_realdata --load_existing
```

### Load existing models, specific datasets
```bash
python -m experiments.comparison_realdata --datasets etth1 exchange_rate --load_existing
```

### Custom DynaMask / WinIT epoch counts
```bash
python -m experiments.comparison_realdata --dynamask_epochs 200 --winit_epochs 200
```

### Quiet mode (suppress KARMA verbosity)
```bash
python -m experiments.comparison_realdata --quiet
```

### Custom checkpoint and output directories
```bash
python -m experiments.comparison_realdata \
  --ckpt_dir outputs/checkpoints/realdata_v2 \
  --out_dir results/realdata_v2
```

### Full run, quiet, load existing
```bash
python -m experiments.comparison_realdata --load_existing --quiet
```

---

## Flag reference

### `comparison_var`

| Flag | Default | Description |
|---|---|---|
| `--T_train` | 5000 | Training timesteps |
| `--T_test` | 1000 | Test timesteps |
| `--N` | 3 | KARMA discretiser bins |
| `--W` | 10 | Oracle window size |
| `--eps` | 0.05 | KARMA Δ_pred stopping tolerance |
| `--lam` | 0.025 | KARMA edge trimming threshold |
| `--M` | 200 | KARMA MC draws per history |
| `--n_pool_min` | 5 | KARMA min pool size |
| `--K_max` | 5 | KARMA max K to search |
| `--n_test` | 100 | Test windows for WinIT/DynaMask/IG/TimeShap |
| `--gen_epochs` | 50 | WinIT generator training epochs |
| `--dyn_epochs` | 100 | DynaMask optimisation epochs |
| `--ts_nsamples` | 200 | TimeShap KernelSHAP coalitions per window |
| `--skip_karma` | off | Skip KARMA |
| `--skip_fo` | off | Skip Feature Occlusion |
| `--skip_dynamask` | off | Skip DynaMask |
| `--skip_winit` | off | Skip WinIT |
| `--skip_ig` | off | Skip Integrated Gradients |
| `--skip_timeshap` | off | Skip TimeShap |
| `--load_generators` | off | Load saved WinIT generators instead of retraining |
| `--seed` | 42 | Random seed |
| `--output_dir` | `results/comparison_var` | Output directory |
| `--verbose` | on | KARMA verbose output |

### `comparison_var_multi`

| Flag | Default | Description |
|---|---|---|
| `--configs` | all | Which configs to run: `tiny small medium large xlarge` |
| `--lam` | 0.025 | KARMA edge trimming threshold |
| `--n_test` | 100 | Test windows per config |
| `--gen_epochs` | 50 | WinIT generator training epochs |
| `--dyn_epochs` | 100 | DynaMask optimisation epochs |
| `--winit_samples` | 3 | WinIT counterfactual samples |
| `--ts_nsamples` | 200 | TimeShap KernelSHAP coalitions per window |
| `--skip_karma` | off | Skip KARMA |
| `--skip_fo` | off | Skip Feature Occlusion |
| `--skip_dynamask` | off | Skip DynaMask |
| `--skip_winit` | off | Skip WinIT |
| `--skip_ig` | off | Skip Integrated Gradients |
| `--skip_timeshap` | off | Skip TimeShap |
| `--seed` | 42 | Random seed |
| `--output_dir` | `results/comparison_var_multi` | Output directory |
| `--verbose` | off | KARMA verbose output |

### `comparison_realdata`

| Flag | Default | Description |
|---|---|---|
| `--datasets` | all | Datasets: `etth1 exchange_rate beijing_pm25` |
| `--device` | auto | `cpu` or `cuda` |
| `--ckpt_dir` | `outputs/checkpoints/realdata` | GRU + WinIT checkpoint directory |
| `--out_dir` | `results/realdata` | Results output directory |
| `--dynamask_epochs` | 100 | DynaMask optimisation epochs |
| `--winit_epochs` | 100 | WinIT generator training epochs |
| `--skip_karma` | off | Skip KARMA (also auto-skipped for D > 15) |
| `--skip_fo` | off | Skip Feature Occlusion |
| `--skip_dynamask` | off | Skip DynaMask |
| `--skip_winit` | off | Skip WinIT |
| `--load_existing` | off | Load pre-trained GRU and WinIT generators |
| `--quiet` | off | Suppress KARMA pipeline verbosity |
