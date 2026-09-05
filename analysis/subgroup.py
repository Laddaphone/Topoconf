"""
Clinical subgroup analysis.
Stratification by breast density, lesion type, subtlety.
Results: cbis_subgroup_analysis.csv on HuggingFace.
"""
import pandas as pd, numpy as np
from scipy import stats

def s_ratio_by_group(subgroup_csv, group_col):
    """Compute S-ratio for each subgroup value."""
    df = pd.read_csv(subgroup_csv)
    rows = []
    for val in sorted(df[group_col].dropna().unique()):
        sub = df[df[group_col] == val]
        fn = sub[sub['is_fn']]; corr = sub[sub['is_correct']]
        if len(fn) < 3 or len(corr) < 3: continue
        r = fn['S'].mean() / corr['S'].mean()
        rows.append(dict(group=group_col, value=val, s_ratio=r,
            n_fn=len(fn), n_corr=len(corr)))
    return pd.DataFrame(rows)

def gradient_test(subgroup_csv, group_col):
    """Spearman correlation of S with group variable among FNs."""
    df = pd.read_csv(subgroup_csv)
    fn = df[df['is_fn'] & df[group_col].notna()]
    if len(fn) < 10: return None
    rho, p = stats.spearmanr(fn[group_col], fn['S'])
    return dict(group=group_col, rho=float(rho), p=float(p), n=len(fn))
