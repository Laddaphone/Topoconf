"""
Ablation study analysis.
Results pre-computed in results/ablation/ on HuggingFace.
This script loads and summarizes them.
"""
import pandas as pd, numpy as np
from scipy import stats

def summarize_ablation(csv_path, group_col='tag'):
    """Summarize ablation results grouped by condition."""
    df = pd.read_csv(csv_path)
    rows = []
    for tag, g in df.groupby(group_col):
        row = {'condition': tag, 'n': len(g)}
        for m in ['auc', 'acc', 'sensitivity', 'specificity', 'S_ratio']:
            if m in g.columns:
                v = g[m].dropna()
                row[f'{m}_mean'] = v.mean(); row[f'{m}_std'] = v.std()
        rows.append(row)
    return pd.DataFrame(rows)

def pairwise_significance(csv_path, baseline_tag, metric='auc'):
    """Paired t-test of each condition vs baseline."""
    df = pd.read_csv(csv_path)
    base = df[df['tag']==baseline_tag][metric].dropna().values
    results = []
    for tag, g in df.groupby('tag'):
        if tag == baseline_tag: continue
        vals = g[metric].dropna().values
        n = min(len(base), len(vals))
        if n < 3: continue
        t, p = stats.ttest_rel(base[:n], vals[:n])
        results.append(dict(comparison=f'{baseline_tag}_vs_{tag}', t=t, p=p, n=n))
    return pd.DataFrame(results)
