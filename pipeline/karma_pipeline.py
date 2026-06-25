"""
Unified KARMA pipeline for all datasets and model architectures.

K-order Approximation via Markov chains for Retrospective Attribution

Pipeline
--------
1. Discretise X_train                          Discretiser
2. Find K*, b* via surrogate validity          select_K_and_baseline
   (Pillar 2 certificate — direct oracle test)
3. Build suffix pool from X_train              SuffixPool
4. Estimate transition kernel via MC           TreeKernelEstimator
5. DAG recovery + variable importance          compute_variable_importance

Usage:
    (poetry run) python -m pipeline.karma_pipeline --dataset etth1 --model lstm
    (poetry run) python -m pipeline.karma_pipeline --dataset exchange_rate --model both
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from karma.causal_recovery.edge_contribution import (
    compute_variable_importance,
    compute_variable_importance_multistep,
)
from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils import check_design, decode_history
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import TreeKernelEstimator
from karma.utils.oracle import load_pretrained_oracle
from karma.utils.sampling import SuffixPool
from pipeline.visualization import visualize

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
DATASETS = [
    "etth1",
    "exchange_rate",
    "beijing",
]


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _reconstruct_ts(X: np.ndarray) -> np.ndarray:
    """Convert sliding-window array (n, W, D) to contiguous time series (T, D)."""
    if X.ndim == 3:
        return np.vstack([X[0]] + [X[i, -1:] for i in range(1, len(X))])
    return X


def run_karma(
    dataset_name: str,
    model_type: str,
    N: int = None,
    eps: float = None,
    lam: float = None,
    M: int = None,
    n_pool_min: int = None,
    K_max: int = None,
    seed: int = None,
    output_dir: str = None,
    verbose: bool = True,
    mega_batch: bool = False,
    mega_batch_size: int = 4096,
    plot: bool = True,
    plot_fmt: str = "both",
    tau: list = None,
) -> dict:
    """
    Full KARMA pipeline for a given dataset and oracle architecture.

    Parameters
    ----------
    dataset_name : one of DATASETS
    model_type   : 'lstm' or 'tcn'
    N            : bins per variable (default from configs/experiments/karma.yaml)
    eps          : Δ^pred stopping tolerance
    lam          : edge-trimming threshold for DAG recovery
    M            : Monte Carlo draws per history (Pillar 3)
    n_pool_min   : minimum pool size per history
    K_max        : maximum lag order to search
    seed         : random seed
    output_dir   : results directory (default: <results_dir>/<model_type>)
    verbose      : print progress

    Returns
    -------
    dict with keys: config, pillar2, pillar3
    """
    dataset_cfg = _load_yaml(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg = _load_yaml(CONFIGS_DIR / "models" / f"{model_type}.yaml")
    exp_cfg = _load_yaml(CONFIGS_DIR / "experiments" / "karma.yaml")

    # Apply config defaults for params not explicitly provided
    N = N if N is not None else exp_cfg["N"]
    eps = eps if eps is not None else exp_cfg["eps"]
    lam = lam if lam is not None else exp_cfg["lam"]
    M = M if M is not None else exp_cfg["M"]
    n_pool_min = n_pool_min if n_pool_min is not None else exp_cfg["n_pool_min"]
    K_max = K_max if K_max is not None else exp_cfg["K_max"]
    seed = seed if seed is not None else exp_cfg["seed"]

    D_cfg = dataset_cfg.get("D")
    W = dataset_cfg.get("W") or 24
    data_dir = dataset_cfg["data_dir"]
    checkpoint_path = dataset_cfg["checkpoints"][model_type] + "/best.pt"

    if output_dir is None:
        output_dir = f"{dataset_cfg['results_dir']}/{model_type}"
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.load(os.path.join(data_dir, "X_train.npy")).astype(np.float32)
    X_val = np.load(os.path.join(data_dir, "X_val.npy")).astype(np.float32)

    if X_train.ndim == 3:
        W_data = X_train.shape[1]
        if W != W_data and verbose:
            print(f"Warning: W={W} != data window {W_data}; using {W_data}")
        W = W_data
    if X_val.ndim == 3:
        W = X_val.shape[1]

    X_train_ts = _reconstruct_ts(X_train)

    D = D_cfg if D_cfg is not None else X_train_ts.shape[-1]
    var_names = dataset_cfg.get("var_names") or [f"X^{d}" for d in range(D)]
    T_train = len(X_train_ts)
    T_held = len(X_val)

    if verbose:
        print("=" * 60)
        print(f"KARMA Pipeline  D={D} N={N} W={W} eps={eps} lam={lam}")
        print(f"  dataset={dataset_name}  oracle={model_type}")
        print(f"  T_train={T_train:,}  T_held={T_held:,}  seed={seed}")
        print("=" * 60)

    disc = Discretiser(N=N)
    disc.fit(X_train_ts)

    oracle_kwargs = dict(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        input_size=D,
        output_size=D,
        forecast_steps=dataset_cfg.get("forecast_steps", 1),
    )
    if model_type == "lstm":
        oracle_kwargs.update(
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
        )
    else:
        oracle_kwargs.update(
            num_channels=model_cfg["num_channels"],
            kernel_size=model_cfg["kernel_size"],
        )
    f_oracle = load_pretrained_oracle(**oracle_kwargs)

    if verbose:
        print(f"\n{'='*60}")
        print("PILLAR 2: K* and b* selection")
        print(f"  Stopping criterion: Δ^pred < {eps}  (sole certificate)")
        print("=" * 60)

    karma_result = select_K_and_baseline(
        f=f_oracle,
        X_train=X_train_ts,
        X_val=X_val,
        disc=disc,
        W=W,
        eps=eps,
        K_max=K_max,
        loss="regression",
        verbose=verbose,
    )

    K_star = karma_result["K_star"]
    b_star = karma_result["b_star"]
    pi_star = karma_result["pi_star"]
    delta_pred = karma_result["delta_pred"]

    pillar2_results = {
        "K_star": K_star,
        "b_star_name": karma_result["b_star_name"],
        "compression_ratio": round(W / K_star, 2),
        "delta_pred": round(delta_pred, 4),
        "memory_fraction": round(K_star / W, 3),
        "certified_zeros": f"lags k > {K_star}",
        "n_queries": len(X_val) - W,
        "convergence_history": karma_result["history"],
    }

    if verbose:
        print(f"\n  K*              = {K_star}")
        print(f"  b*              = {karma_result['b_star_name']}")
        print(f"  Compression     = {W}/{K_star} = {W/K_star:.1f}x")
        print(f"  Δ^pred [CERT]   = {delta_pred:.4f} < {eps}")

    if verbose:
        print(f"\n{'='*60}")
        print("PILLAR 3: Transition kernel + DAG recovery")
        print("=" * 60)

    check_design(disc, K=K_star, T_train=T_train, n_pool_min=n_pool_min, lam=lam, W=W)

    pool = SuffixPool(disc=disc, K=K_star, W=W)
    pool.build(X_train_ts)
    stats = pool.coverage_stats(pi_star, n_pool_min=n_pool_min)

    if verbose:
        print(f"  |H+_K*|        = {stats['n_observed_histories']:,}")
        print(f"  Mean pool size = {stats['mean_pool_size']:.1f}")
        print(f"  Pool coverage  = {stats['pool_coverage']:.1%}")

    est = TreeKernelEstimator(
        disc=disc,
        pool=pool,
        K=K_star,
        W=W,
        b_star=b_star,
        # delta_pred=delta_pred,
        M=M,
        rng=np.random.default_rng(seed),
    )
    est.fit(
        f=f_oracle,
        pi_star=pi_star,
        verbose=verbose,
        mega_batch=mega_batch,
        mega_batch_size=mega_batch_size,
    )

    horizon = getattr(est, "horizon", 1)
    if horizon > 1:
        vi = compute_variable_importance_multistep(est, pi_star, disc, tau=tau, lam=lam)
    else:
        vi = compute_variable_importance(est, pi_star, disc, lam=lam)

    if verbose:
        print("\nVariable Importance (normalised):")
        for d in range(D):
            print(
                f"  {var_names[d]}: Phi={vi['Phi_n'][d]:.3f}  "
                f"lag profile={np.round(vi['phi'][d], 4)}"
            )

    retained_edges = vi["edges"]
    if verbose:
        print(f"\nRetained edges (rho >= {lam}): {len(retained_edges)}")
        for e in sorted(retained_edges, key=lambda x: -x["rho"]):
            tgt_str = f" -> {var_names[e['tgt']]}" if "tgt" in e else ""
            print(
                f"  {var_names[e['src']]} lag {e['lag']}{tgt_str}"
                f"  rho={e['rho']:.4f}"
            )

    # Sample kernel rows for inspection
    sample_kernels = []
    shown = 0
    for h_idx in est.estimated_histories:
        if pool.pool_size(h_idx) >= 5 and shown < 5:
            probs = est.predict(h_idx)[0]
            floor = est.noise_floor(h_idx)
            h_arr = decode_history(h_idx, K_star, D, disc.N)
            sample_kernels.append(
                {
                    "h_idx": h_idx,
                    "h_arr": h_arr.tolist(),
                    "pool_size": pool.pool_size(h_idx),
                    "floor": round(float(floor), 4),
                    "probs": {var_names[d]: probs[d].tolist() for d in range(D)},
                }
            )
            shown += 1

    pillar3_results = {
        "pool_stats": stats,
        "variable_importance": {
            "rho_mat": vi["rho_mat"].tolist(),
            "phi": vi["phi"].tolist(),
            "Phi": vi["Phi"].tolist(),
            "Phi_n": vi["Phi_n"].tolist(),
        },
        "retained_edges": [
            {
                "src": e["src"],
                **(
                    {"tgt": e["tgt"], "tgt_name": var_names[e["tgt"]]}
                    if "tgt" in e
                    else {}
                ),
                **({"tgt_step": e["tgt_step"]} if "tgt_step" in e else {}),
                "lag": e["lag"],
                "rho": round(e["rho"], 4),
                "src_name": var_names[e["src"]],
            }
            for e in sorted(retained_edges, key=lambda x: -x["rho"])
        ],
        "sample_kernels": sample_kernels,
    }

    pd.DataFrame(pillar3_results["retained_edges"]).to_csv(
        f"{output_dir}/karma_dag.csv", index=False
    )
    if sample_kernels:
        kernel_rows = []
        for sk in sample_kernels:
            for d in range(D):
                for s in range(N):
                    kernel_rows.append(
                        {
                            "h_idx": sk["h_idx"],
                            "variable": var_names[d],
                            "bin": s,
                            "prob": round(sk["probs"][var_names[d]][s], 4),
                            "pool_size": sk["pool_size"],
                        }
                    )
        pd.DataFrame(kernel_rows).to_csv(f"{output_dir}/karma_kernels.csv", index=False)

    pd.DataFrame(karma_result["history"]).to_csv(
        f"{output_dir}/karma_convergence.csv", index=False
    )

    results = {
        "config": {
            "dataset": dataset_name,
            "oracle": model_type,
            "D": D,
            "N": N,
            "W": W,
            "eps": eps,
            "lam": lam,
            "M": M,
            "T_train": T_train,
            "T_held": T_held,
            "K_max": K_max,
            "seed": seed,
            "var_names": var_names,
        },
        "pillar2": pillar2_results,
        "pillar3": pillar3_results,
    }

    results_json_path = f"{output_dir}/karma_results.json"
    with open(results_json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    if plot:
        figures_dir = Path(output_dir) / "figures"
        if verbose:
            print(f"\nGenerating figures → {figures_dir}/")
        try:
            visualize(
                results_path=results_json_path,
                figures_dir=figures_dir,
                var_names=var_names,
                fmt=plot_fmt,
                verbose=verbose,
            )
        except Exception as exc:
            print(f"  Warning: visualization failed: {exc}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"KARMA complete. Results saved to {output_dir}/")
        print("=" * 60)

    return results


def main():
    p = argparse.ArgumentParser(description="Run KARMA pipeline")
    p.add_argument("--dataset", required=True, choices=DATASETS + ["all"])
    p.add_argument("--model", default="both", choices=["lstm", "tcn", "both"])
    p.add_argument("--N", type=int, default=None, help="Bins per variable")
    p.add_argument("--eps", type=float, default=None, help="Δ^pred tolerance")
    p.add_argument("--lam", type=float, default=None, help="Edge-trim threshold")
    p.add_argument("--M", type=int, default=None, help="MC draws per history")
    p.add_argument("--n_pool_min", type=int, default=None)
    p.add_argument("--K_max", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no_plot", action="store_true", help="Skip figure generation")
    p.add_argument(
        "--plot_fmt",
        default="both",
        choices=["png", "pdf", "both"],
        help="Figure format (default: png)",
    )
    p.add_argument(
        "--mega_batch",
        action="store_true",
        help="Collect all H*M windows into one batch for GPU throughput",
    )
    p.add_argument(
        "--mega_batch_size",
        type=int,
        default=4096,
        help="Chunk size for mega-batch oracle calls (default 4096)",
    )
    p.add_argument(
        "--tau",
        type=str,
        default=None,
        help="Comma-separated forecast step indices for multistep VI, e.g. '0,1,2'. "
        "Defaults to all horizon steps.",
    )
    args = p.parse_args()

    tau = [int(t) for t in args.tau.split(",")] if args.tau else None

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    models = ["lstm", "tcn"] if args.model == "both" else [args.model]

    for ds in datasets:
        for m in models:
            run_karma(
                dataset_name=ds,
                model_type=m,
                N=args.N,
                eps=args.eps,
                lam=args.lam,
                M=args.M,
                n_pool_min=args.n_pool_min,
                K_max=args.K_max,
                seed=args.seed,
                output_dir=args.output_dir,
                verbose=not args.quiet,
                mega_batch=args.mega_batch,
                mega_batch_size=args.mega_batch_size,
                plot=not args.no_plot,
                plot_fmt=args.plot_fmt,
                tau=tau,
            )


if __name__ == "__main__":
    main()
