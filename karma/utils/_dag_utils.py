def evaluate_dag(
    recovered_edges: list,
    true_edges: list,
    var_names: list,
) -> dict:
    """
    Compute precision, recall, F1 of recovered DAG vs true DAG.

    Parameters
    ----------
    recovered_edges : list of (src_var, tgt_var, lag, rho)
    true_edges      : list of (src_var, tgt_var, lag)
    var_names       : list of str

    Returns
    -------
    dict with precision, recall, f1, tp, fp, fn, and edge details
    """
    true_set = {(e["src"], e["tgt"], e["lag"]) for e in true_edges}
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

    # human-readable edge descriptions
    def edge_str(src_var, tgt_var, lag):
        return f"{var_names[src_var]}(t-{lag}) -> {var_names[tgt_var]}(t)"

    true_positive_edges = [
        edge_str(src_var, tgt_var, lag)
        for (src_var, tgt_var, lag) in (true_set & rec_set)
    ]
    false_positive_edges = [
        edge_str(src_var, tgt_var, lag)
        for (src_var, tgt_var, lag) in (rec_set - true_set)
    ]
    false_negative_edges = [
        edge_str(src_var, tgt_var, lag)
        for (src_var, tgt_var, lag) in (true_set - rec_set)
    ]

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "true_positive_edges": true_positive_edges,
        "false_positive_edges": false_positive_edges,
        "false_negative_edges": false_negative_edges,
    }
