"""FN triage analysis at matched workloads with patient-level statistics."""
import numpy as np
from scipy import stats

def fn_catch_at_burden(labels, preds, signal, burden=0.42):
    """Fraction of FNs caught by referring top-burden% by signal."""
    n = len(labels)
    fn = (preds == 0) & (labels == 1)
    if fn.sum() == 0: return float('nan')
    nr = int(np.ceil(burden * n))
    idx = np.argsort(signal)[::-1][:nr]
    referred = np.zeros(n, bool); referred[idx] = True
    return float((fn & referred).sum() / fn.sum())

def triage_comparison(npz_path, burdens=None):
    """Compare vacuity vs H_pred triage from a single .npz file."""
    if burdens is None: burdens = [0.25, 0.30, 0.42, 0.50]
    d = np.load(npz_path, allow_pickle=True)
    ty = d['ty']; tp = d['tpred']
    if 't_vacuity' not in d.files: return None
    vac = d['t_vacuity']; pe = d['t_pred_ent']
    results = []
    for b in burdens:
        cv = fn_catch_at_burden(ty, tp, vac, b)
        ch = fn_catch_at_burden(ty, tp, pe, b)
        results.append(dict(burden=b, catch_vac=cv, catch_H=ch,
            vac_wins=int(cv > ch), delta=cv - ch))
    return results

def patient_level_test(npz_paths, burden=0.42):
    """Patient-level Wilcoxon signed-rank test across seeds."""
    from collections import defaultdict
    patient_data = defaultdict(lambda: {'fn': 0, 'vac': 0, 'h': 0})
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        if 't_vacuity' not in d.files: continue
        ty = d['ty']; tp = d['tpred']; n = len(ty)
        fn = (tp == 0) & (ty == 1); vac = d['t_vacuity']; pe = d['t_pred_ent']
        nr = int(np.ceil(burden * n))
        rv = np.zeros(n, bool); rv[np.argsort(vac)[::-1][:nr]] = True
        rh = np.zeros(n, bool); rh[np.argsort(pe)[::-1][:nr]] = True
        pids = [str(p) for p in d['test_pids']] if 'test_pids' in d.files else [str(i) for i in range(n)]
        for i in range(min(len(ty), len(pids))):
            if not fn[i]: continue
            patient_data[pids[i]]['fn'] += 1
            if rv[i]: patient_data[pids[i]]['vac'] += 1
            if rh[i]: patient_data[pids[i]]['h'] += 1
    vac_rates, h_rates = [], []
    for pid, d in patient_data.items():
        if d['fn'] == 0: continue
        vac_rates.append(d['vac'] / d['fn'])
        h_rates.append(d['h'] / d['fn'])
    vac_rates = np.array(vac_rates); h_rates = np.array(h_rates)
    diffs = vac_rates - h_rates; nonzero = diffs[diffs != 0]
    vac_better = (diffs > 0).sum(); h_better = (diffs < 0).sum()
    W, p = stats.wilcoxon(nonzero) if len(nonzero) >= 10 else (float('nan'), float('nan'))
    return dict(n_patients=len(vac_rates), vac_better=int(vac_better),
        h_better=int(h_better), tied=int((diffs==0).sum()), W=float(W), p=float(p))
