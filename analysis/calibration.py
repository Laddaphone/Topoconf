"""Calibration analysis: ECE, reliability diagrams, operating points."""
import numpy as np
from sklearn.metrics import roc_curve

def ece(probs, labels, preds, n_bins=10):
    """Expected Calibration Error."""
    conf = probs.max(1); correct = (preds == labels).astype(float)
    total = 0
    for b in range(n_bins):
        lo, hi = b/n_bins, (b+1)/n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() > 0:
            total += mask.sum() * abs(correct[mask].mean() - conf[mask].mean())
    return total / len(labels)

def reliability_diagram(probs, labels, preds, n_bins=10):
    """Return (bin_conf, bin_acc) for plotting."""
    conf = probs.max(1); correct = (preds == labels).astype(float)
    bins_conf, bins_acc = [], []
    for b in range(n_bins):
        lo, hi = b/n_bins, (b+1)/n_bins
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() > 0:
            bins_conf.append(conf[mask].mean())
            bins_acc.append(correct[mask].mean())
    return np.array(bins_conf), np.array(bins_acc)

def sensitivity_at_specificity(labels, scores, target_spec=0.90):
    """Find sensitivity at a given specificity threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    spec = 1 - fpr
    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0: return float('nan'), float('nan')
    idx = valid[np.argmin(spec[valid])]
    return float(tpr[idx]), float(thresholds[idx])
