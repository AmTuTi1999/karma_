"""Plot lag AUC removal curves for all datasets, architectures, and methods."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent.parent / "results" / "realdata"
OUT_PATH = Path(__file__).parent.parent / "results" / "realdata" / "lag_auc_curves.pdf"

DATASETS = [
    ("etth1", "ETTh1"),
    ("exchange_rate", "Exchange Rate"),
    ("beijing_pm25", "Beijing PM2.5"),
]
ARCHS = ["gru", "lstm", "tcn"]
METHODS = [
    ("fo",       "FO",       "#888888", "--"),
    ("dynamask", "DynaMask", "#e69f00", "-."),
    ("winit",    "WinIT",    "#56b4e9", ":"),
    ("ig",       "IG",       "#009e73", "--"),
    ("timeshap", "TimeShap", "#cc79a7", "-."),
    ("karma",    "KARMA",    "#d55e00", "-"),
]
N_STEPS = 10
X = np.linspace(1 / N_STEPS, 1.0, N_STEPS)

fig, axes = plt.subplots(
    nrows=len(DATASETS),
    ncols=len(ARCHS),
    figsize=(11, 8),
    sharex=True,
)

for row, (ds_key, ds_label) in enumerate(DATASETS):
    fpath = RESULTS_DIR / f"{ds_key}_results.json"
    data = json.loads(fpath.read_text())

    for col, arch in enumerate(ARCHS):
        ax = axes[row][col]

        for method_key, method_label, color, ls in METHODS:
            curve_key = f"{arch}_{method_key}_lag_changes"
            if curve_key not in data:
                continue
            y = np.array(data[curve_key])
            lw = 2.2 if method_key == "karma" else 1.4
            ax.plot(X, y, label=method_label, color=color, linestyle=ls, linewidth=lw)

        ax.set_xlim(0, 1.05)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if row == 0:
            ax.set_title(arch.upper(), fontsize=10, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{ds_label}\n$|\\Delta\\hat{{y}}|$", fontsize=9)
        if row == len(DATASETS) - 1:
            ax.set_xlabel("Fraction removed", fontsize=8)

# shared legend below the grid
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=len(METHODS),
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Lag AUC curves — whole-timestep removal (higher = better)",
    fontsize=11,
    fontweight="bold",
    y=1.01,
)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
print(f"Saved to {OUT_PATH}")
plt.show()
