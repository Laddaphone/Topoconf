"""
Corrected uncertainty decomposition for all architectures.

Terminology (corrected per Mucsanyi et al. 2024 NeurIPS):
  pred_ent = H[E[p]]      = predictive entropy (TOTAL uncertainty)
  exp_ent  = E[H[p]]      = expected entropy (data/aleatoric)
  mi       = pred - exp    = mutual information (model/epistemic)
  vacuity  = K/S           = evidential uncertainty (EDL only)

NOT "aleatoric" and "epistemic" — these labels were incorrect
in the original submission. E[H] correlates with H_pred at r=0.997.
MI correlates with vacuity at r=0.999. The decomposition is
entangled (Mucsanyi et al. 2024).

For ensembles/MC-Dropout:
  pred_ent = H(p_bar)              = entropy of mean prediction
  exp_ent  = E_theta[H(p_theta)]   = mean of per-member entropies
  mi       = pred_ent - exp_ent    = mutual information
  mc_var   = Var(p_theta)          = prediction variance
"""
import numpy as np
from scipy.special import digamma
from sklearn.metrics import roc_auc_score

def dirichlet_decomp(alphas):
    """Correct Dirichlet decomposition from alpha parameters."""
    S = alphas.sum(axis=1)
    p = alphas / S[:, None]
    K = alphas.shape[1]
    pred_ent = -np.sum(p * np.log(p + 1e-10), axis=1)
    psi_a1 = digamma(alphas + 1)
    psi_S1 = digamma(S + 1)[:, None]
    exp_ent = -np.sum(p * (psi_a1 - psi_S1), axis=1)
    mi = np.maximum(pred_ent - exp_ent, 0)
    return dict(pred_ent=pred_ent, exp_ent=exp_ent, mi=mi,
                vacuity=K/S, probs=p, preds=alphas.argmax(1))

def mc_decomp(softmax_samples):
    """MC-Dropout / Deep Ensemble decomposition. Input: (T, N, K)."""
    p_bar = softmax_samples.mean(0)
    pred_ent = -np.sum(p_bar * np.log(p_bar + 1e-10), axis=1)
    per_pass = -np.sum(softmax_samples * np.log(softmax_samples + 1e-10), axis=2)
    exp_ent = per_pass.mean(0)
    mi = np.maximum(pred_ent - exp_ent, 0)
    return dict(pred_ent=pred_ent, exp_ent=exp_ent, mi=mi,
                mc_var=softmax_samples.var(0).sum(1),
                probs=p_bar, preds=p_bar.argmax(1))

def error_prediction_auroc(signals_dict, errors):
    """Compute AUROC for error prediction from each uncertainty signal."""
    results = {}
    for name, vals in signals_dict.items():
        try: results[name] = roc_auc_score(errors, vals)
        except: results[name] = float('nan')
    return results

def comparable_signals(npz_path):
    """Extract comparable uncertainty signals from any architecture."""
    d = np.load(npz_path, allow_pickle=True)
    ty = d['ty']; tp = d['tp']; tpred = d['tpred']
    errors = (tpred != ty).astype(int)
    signals = {}
    if 't_vacuity' in d.files:
        signals['vacuity'] = d['t_vacuity']
        signals['pred_ent'] = d['t_pred_ent']
        signals['exp_ent'] = d['t_exp_ent']
        signals['mi'] = d['t_mi']
    else:
        pred_ent = -np.sum(tp * np.log(tp + 1e-10), axis=1)
        signals['pred_ent'] = pred_ent
        signals['max_prob'] = 1.0 - tp.max(axis=1)
    return dict(signals=signals, errors=errors, ty=ty, tpred=tpred, probs=tp)
