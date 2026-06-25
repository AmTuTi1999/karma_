import json
import os
import numpy as np
import pandas as pd

from karma.markov_approximation.markov_surrogacy import select_K_and_baseline
from karma.utils import check_design, decode_history
from karma.utils.discretiser import Discretiser
from karma.utils.kernel_estimator import (
    BStarKernelEstimator,
    TreeKernelEstimator,
    FactoredKernelEstimator,
)
from karma.utils.sampling import SuffixPool
from karma.causal_recovery.edge_contribution import compute_variable_importance
from karma.utils._dag_utils import evaluate_dag


from karma.utils.oracle import _make_oracle
from karma.uncertainty_estimation.uncertainty_estimation import (
    compute_uncertainty_attributes,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class KARMA:

    def __init__(
        self,
        dimension: int,
        bins: int,
        window_size: int,
        markov_kernel_approximation_threshold: float,
        edge_trimming_threshold: float,
        monte_carlo_draws: int,
        min_pool_size: int,
        max_K: int,
        seed: int,
    ):
        self.dimension = dimension
        self.bins = bins
        self.window_size = window_size
        self.markov_kernel_approximation_threshold = (
            markov_kernel_approximation_threshold
        )
        self.edge_trimming_threshold = edge_trimming_threshold
        self.monte_carlo_draws = monte_carlo_draws
        self.min_pool_size = min_pool_size
        self.max_K = max_K
        self.seed = seed

    def explain(
        self,
        trained_model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        true_edges: list = None,
        output_dir: str = "results",
        verbose: bool = True,
        kernel_approximation_strategy: str = "RegressionTree",
    ) -> dict:

        oracle = _make_oracle(trained_model)
        os.makedirs(output_dir, exist_ok=True)
        var_names = [f"X^{d}" for d in range(self.dimension)]
        T_train, T_held = X_train.shape[0], X_test.shape[0]
        verbose_output = {}
        if verbose:
            verbose_output["header"] = {
                "pipeline": f"KARMA Pipeline D={self.dimension} N={self.bins} W={self.window_size} eps={self.markov_kernel_approximation_threshold} lam={self.edge_trimming_threshold}",
                "data": f"T_train={T_train:,} T_held={T_held:,} seed={self.seed}",
            }

            print("=" * 60)
            print(
                f"KARMA Pipeline  D={self.dimension} N={self.bins} W={self.window_size} eps={self.markov_kernel_approximation_threshold} lam={self.edge_trimming_threshold}"
            )

        disc = Discretiser(N=self.bins)
        disc.fit(X_train)
        if verbose:
            print(f"\nDiscretiser fitted: D={self.dimension}, N={self.bins}")
            print(f"\n{'='*60}")
            print("PILLAR 2: K* and b* selection")
            print(
                f"  Stopping criterion: Δ^pred < {self.markov_kernel_approximation_threshold}  (sole certificate)"
            )
            print("  Δ^f reported as diagnostic only")
            print("=" * 60)
            verbose_output["quantiles"] = {
                "quantiles": disc.quantiles_.tolist(),
            }
        p2_results = select_K_and_baseline(
            f=oracle,
            X_train=X_train,
            X_test=X_test,
            disc=disc,
            W=self.window_size,
            eps=self.markov_kernel_approximation_threshold,
            K_max=self.max_K,
            loss="regression",
            verbose=verbose,
        )
        p2_results["compression_ratio"] = round(
            self.window_size / p2_results["K_star"], 2
        )
        p2_results["memory_fraction"] = round(
            p2_results["K_star"] / self.window_size, 3
        )
        p2_results["certified_zeros"] = f"lags k > {p2_results['K_star']}"
        p2_results["n_queries"] = len(X_test) - self.window_size
        p2_results["convergence_history"] = p2_results["history"]

        K_star = p2_results["K_star"]
        b_star = p2_results["b_star"]
        pi_star = p2_results["pi_star"]
        delta_pred = p2_results["delta_pred"]
        delta_f = p2_results["delta_f"]

        if verbose:
            print("\nPillar 2 results:")
            print(f"  K*              = {K_star}")
            print(f"  b*              = {p2_results['b_star_name']}")
            print(
                f"  Compression     = W/K* = {self.window_size}/{K_star} = {self.window_size/K_star:.1f}x"
            )
            print(
                f"  Δ^pred [CERT]   = {delta_pred:.4f} < {self.markov_kernel_approximation_threshold}"
            )
            print(f"  Δ^f    [diag]   = {delta_f:.4f}")
            print(f"  Certified zeros for lags k > {K_star}")

        if verbose:
            print(f"\n{'='*60}")
            print("PILLAR 3: Transition kernel + DAG recovery")
            print("=" * 60)

        p3_results = {}

        if kernel_approximation_strategy == "RegressionTree":
            kernel_type = "C"
        elif kernel_approximation_strategy == "BStar":
            kernel_type = "B"
        else:
            kernel_type = "A"
        if verbose:
            check_design(
                disc=disc,
                K=K_star,
                T_train=T_train,
                n_pool_min=self.min_pool_size,
                lam=self.edge_trimming_threshold,
                W=self.window_size,
                kernel_type=kernel_type,
            )

        if verbose:
            print(f"\nBuilding suffix pool (K*={K_star})...")
        pool = SuffixPool(disc=disc, K=K_star, W=self.window_size)
        pool.build(X_train)
        stats = pool.coverage_stats(pi_star, n_pool_min=self.min_pool_size)

        if verbose:
            print(f"  |H+_K*|        = {stats['n_observed_histories']:,}")
            print(f"  Mean pool size = {stats['mean_pool_size']:.1f}")
            print(
                f"  Pool coverage  = {stats['pool_coverage']:.1%}"
                f"  (target >= 0.9 for reliable Pillar 3)"
            )

        if stats["pool_coverage"] < 0.5:
            print(f"\n  WARNING: Pool coverage {stats['pool_coverage']:.1%} is low.")
            print(
                "  Kernel estimates for thin-pool histories fall back to "
                "Hamming-1 neighbours or uniform prior."
            )
            print("  Increase T_train or decrease N for better coverage.")

        if kernel_type == "B":
            kernel_estimator = BStarKernelEstimator(
                disc=disc,
                pool=pool,
                K=K_star,
                W=self.window_size,
                b_star=b_star,
                delta_pred=delta_pred,
                M=self.monte_carlo_draws,
                rng=np.random.default_rng(self.seed),
            )
            kernel_estimator.fit(oracle, pi_star, verbose=verbose)
            coverage = kernel_estimator.coverage(
                pi_star, lam=self.edge_trimming_threshold
            )
        elif kernel_type == "C":
            kernel_estimator = TreeKernelEstimator(
                disc,
                K=K_star,
                W=self.window_size,
                M=self.monte_carlo_draws,
                n_pool=self.min_pool_size,
                pool=pool,
                b_star=b_star,
            )
            kernel_estimator.fit(
                f=oracle,
                pi_star=pi_star,
                verbose=True,
            )
            coverage = kernel_estimator.coverage(
                pi_star, lam=self.edge_trimming_threshold
            )
        elif kernel_type == "A":
            print("Using default FactoredKernelEstimator (no pooling, no tree).")
            kernel_estimator = FactoredKernelEstimator(
                disc=disc,
                K=K_star,
                alpha=0.5,
            )
            print("I'll figure this out later")
            coverage = 0.0

        if verbose:
            print(
                f"\nCoverage at lam={self.edge_trimming_threshold}, M={self.monte_carlo_draws}: {coverage:.1%}"
            )
            print(
                f"  MC noise floor: sqrt(1/2M) = {np.sqrt(1/(2*self.monte_carlo_draws)):.4f}"
            )
            req_M = int(2 / self.edge_trimming_threshold**2)
            if coverage < 0.9:
                print(
                    f"  WARNING: Coverage is low. Consider increasing M to > {req_M:,} for better MC estimates."
                )

        vi = compute_variable_importance(
            kernel_estimator, pi_star, disc, lam=self.edge_trimming_threshold
        )

        if verbose:
            verbose_output["variable_importance"] = []
            for d in range(self.dimension):
                verbose_output["variable_importance"].append(
                    {
                        "variable": var_names[d],
                        "Phi": round(vi["Phi_n"][d], 3),
                        "lag_profile": np.round(vi["phi"][d], 4).tolist(),
                    }
                )

        sample_kernels = []
        shown = 0
        for h_idx in kernel_estimator.estimated_histories:
            if pool.pool_size(h_idx) >= 5 and shown < 5:
                probs = kernel_estimator.predict(h_idx)
                noise_floor = kernel_estimator.noise_floor(h_idx)
                h_arr = decode_history(h_idx, K_star, self.dimension, disc.N)
                sample_kernels.append(
                    {
                        "h_idx": h_idx,
                        "h_arr": h_arr.tolist(),
                        "pool_size": pool.pool_size(h_idx),
                        "noise_floor": round(float(noise_floor), 4),
                        "probs": probs.tolist(),
                        "h_idx_coverage": max(
                            0, 1 - noise_floor / (self.edge_trimming_threshold / 2)
                        ),
                    }
                )
                shown += 1

        # Explanation Level 1 and 2: Value Importance and lag variables
        p3_results = {
            "variable_importance": {
                "phi": vi["phi"].tolist(),
                "Phi": vi["Phi"].tolist(),
                "Phi_n": vi["Phi_n"].tolist(),
            },
            "sample_kernels": sample_kernels,
        }

        ############################################################
        # Explanations Level 4: Counterfactual Explanation and ACE
        ##############################################################

        retained_edges = [
            {
                "src": e[0],
                "tgt": e[1],
                "lag": e[2],
                "rho": round(e[3], 4),
                "src_name": var_names[e[0]],
                "tgt_name": var_names[e[1]],
            }
            for e in sorted(vi["edges"], key=lambda x: -x[3])
        ]
        if verbose:
            verbose_output["retained_edges"] = {
                "threshold": self.edge_trimming_threshold,
                "count": len(retained_edges),
                "edges": [],
            }
            for e in sorted(retained_edges, key=lambda x: -x[3]):
                verbose_output["retained_edges"]["edges"].append(
                    {
                        "src": var_names[e[0]],
                        "tgt": var_names[e[1]],
                        "lag": e[2],
                        "rho": round(e[3], 4),
                    }
                )

        #####################################################################################
        # Explanations Level 5: Uncertainty estimates for Aleatoric and Epistemic uncertainty
        #####################################################################################

        uncertainty_estimates_all_dims = {}
        for d in range(self.dimension):
            uncertainty_estimates_all_dims[d] = compute_uncertainty_attributes(
                estimator=kernel_estimator,
                sample_kernels=sample_kernels,
                d=d,
            )
        print(uncertainty_estimates_all_dims)

        ####################################################################
        # MODEL AND EXPLANATION DAG AUDITING
        #####################################################################
        if true_edges is not None:

            dag_eval = evaluate_dag(retained_edges, true_edges, var_names)
            if verbose:
                verbose_output["dag_evaluation"] = {
                    "precision": round(dag_eval["precision"], 3),
                    "recall": round(dag_eval["recall"], 3),
                    "f1": round(dag_eval["f1"], 3),
                    "true_positive_edges": dag_eval["true_positive_edges"],
                    "false_positive_edges": dag_eval["false_positive_edges"],
                    "false_negative_edges": dag_eval["false_negative_edges"],
                }
        dag_df = pd.DataFrame(p3_results["retained_edges"])
        dag_df.to_csv(f"{output_dir}/karma_dag.csv", index=False)

        if sample_kernels:
            kernel_rows = []
            for sk in sample_kernels:
                for d in range(self.dimension):
                    for s in range(self.bins):
                        kernel_rows.append(
                            {
                                "h_idx": sk["h_idx"],
                                "variable": var_names[d],
                                "bin": s,
                                "prob": round(sk["probs"][d][s], 4),
                                "pool_size": sk["pool_size"],
                            }
                        )
            pd.DataFrame(kernel_rows).to_csv(
                f"{output_dir}/karma_kernels.csv", index=False
            )

        conv_df = pd.DataFrame(p2_results["history"])
        conv_df.to_csv(f"{output_dir}/karma_convergence.csv", index=False)

        results = {
            "config": {
                "D": self.dimension,
                "N": self.bins,
                "W": self.window_size,
                "eps": self.markov_kernel_approximation_threshold,
                "lam": self.edge_trimming_threshold,
                "M": self.markov_kernel_approximation_threshold,
                "T_train": T_train,
                "T_held": T_held,
                "K_max": self.max_K,
                "seed": self.seed,
            },
            "true_edges": [list(e) for e in true_edges],
            "pillar2": p2_results,
            "pillar3": p3_results,
        }

        with open(f"{output_dir}/karma_results.json", "w") as fh:
            json.dump(results, fh, indent=2, cls=NumpyEncoder)

        if verbose:
            verbose_output["completion"] = {
                "message": "KARMA complete. Results saved to:",
                "files": [
                    f"{output_dir}/karma_results.json",
                    f"{output_dir}/karma_dag.csv",
                    f"{output_dir}/karma_kernels.csv",
                    f"{output_dir}/karma_convergence.csv",
                ],
            }
            with open(f"{output_dir}/karma_verbose_output.json", "w") as fh:
                json.dump(verbose_output, fh, indent=2)
        return results
