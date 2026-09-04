"""Conformal prediction implementation (split conformal + adaptive variants)."""

import numpy as np


def nonconformity_score(probs, labels):
    """s_i = 1 - p_hat(y_i | x_i)"""
    return 1.0 - probs[np.arange(len(labels)), labels]


def calibrate_threshold(cal_scores, alpha=0.05):
    """Compute conformal quantile from calibration scores."""
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(cal_scores, q_level)


def prediction_set(probs, q_hat):
    """Return prediction set for each sample."""
    sets = []
    for i in range(len(probs)):
        s = set(np.where(probs[i] >= 1 - q_hat)[0])
        if len(s) == 0:
            s = {np.argmax(probs[i])}  # Empty set fallback: forced singleton
        sets.append(s)
    return sets


def adaptive_threshold(q_base, difficulty_signal, gamma):
    """
    TACP/EACP: scale threshold by difficulty signal.
    TACP: difficulty_signal = topological complexity from persistent homology
    EACP: difficulty_signal = epistemic uncertainty (K/S for EDL)
    gamma: scaling factor (swept over [0.0, 0.1, ..., 1.0])
    """
    return q_base * (1 + gamma * difficulty_signal)


def evaluate_coverage(prediction_sets, labels):
    """Compute marginal and class-conditional coverage."""
    covered = np.array([labels[i] in prediction_sets[i]
                        for i in range(len(labels))])
    results = {"marginal_coverage": covered.mean()}
    for cls in np.unique(labels):
        mask = labels == cls
        results[f"class_{cls}_coverage"] = covered[mask].mean()
    sizes = np.array([len(s) for s in prediction_sets])
    results["mean_set_size"] = sizes.mean()
    results["singleton_frac"] = (sizes == 1).mean()
    return results
