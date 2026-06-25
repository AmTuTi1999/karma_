import numpy as np

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


def run_pcmci(data, K, var_names, alpha=0.05):
    dataframe = pp.DataFrame(data, datatime=np.arange(len(data)), var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=K, pc_alpha=alpha)

    edges = []
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    for tgt in range(data.shape[1]):
        for src in range(data.shape[1]):
            for lag in range(1, K):
                p = p_matrix[src, tgt, lag]
                val = val_matrix[src, tgt, lag]
                if p < alpha:
                    edges.append(
                        {
                            "src": src,
                            "tgt": tgt,
                            "lag": lag,
                            "src_name": var_names[src],
                            "tgt_name": var_names[tgt],
                            "p_value": round(float(p), 5),
                            "coef": round(float(val), 5),
                        }
                    )
    return edges


def edge_metrics(recovered_edges, true_edges):
    true_set = {(src_var, tgt_var, lag) for (src_var, tgt_var, lag) in true_edges}
    rec_set = {(e["src"], e["tgt"], e["lag"]) for e in recovered_edges}
    tp = len(true_set & rec_set)
    fp = len(rec_set - true_set)
    fn = len(true_set - rec_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
