"""
Bootstrap confidence intervals and cluster-aware resampling.
Reviewer 6: patient-level bootstrap, not seed-level inference.
"""
import numpy as np

def bootstrap_ci(data, n_boot=5000, ci=0.95, seed=42):
    """Percentile bootstrap CI."""
    rng = np.random.RandomState(seed)
    means = [np.mean(rng.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1-ci)/2*100)
    hi = np.percentile(means, (1+ci)/2*100)
    return float(lo), float(hi), float(np.mean(data))

def cluster_bootstrap(values, cluster_ids, n_boot=5000, seed=42):
    """Resample clusters (datasets), not individual observations."""
    rng = np.random.RandomState(seed)
    clusters = np.unique(cluster_ids)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        vals = np.concatenate([values[cluster_ids == c] for c in sampled])
        means.append(vals.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
