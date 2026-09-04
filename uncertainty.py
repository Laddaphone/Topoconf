import numpy as np
from scipy.special import digamma


def dirichlet_decomp(alphas):
    """
    Correct Dirichlet uncertainty decomposition.
    Returns dict with: pred_ent, exp_ent, mi, vacuity, probs, preds
    """
    S = alphas.sum(axis=1)
    p = alphas / S[:, None]
    K = alphas.shape[1]
    pred_ent = -np.sum(p * np.log(p + 1e-10), axis=1)
    psi_a1 = digamma(alphas + 1)
    psi_S1 = digamma(S + 1)[:, None]
    exp_ent = -np.sum(p * (psi_a1 - psi_S1), axis=1)
    mi = np.maximum(pred_ent - exp_ent, 0)
    return dict(pred_ent=pred_ent, exp_ent=exp_ent, mi=mi,
                vacuity=K / S, probs=p, preds=alphas.argmax(1))


def mc_decomp(softmax_samples):
    """
    MC Dropout / Deep Ensemble decomposition.
    softmax_samples: (T, N, K)
    """
    p_bar = softmax_samples.mean(0)
    pred_ent = -np.sum(p_bar * np.log(p_bar + 1e-10), axis=1)
    per_pass = -np.sum(softmax_samples * np.log(softmax_samples + 1e-10), axis=2)
    exp_ent = per_pass.mean(0)
    mi = np.maximum(pred_ent - exp_ent, 0)
    return dict(pred_ent=pred_ent, exp_ent=exp_ent, mi=mi,
                mc_var=softmax_samples.var(0).sum(1),
                probs=p_bar, preds=p_bar.argmax(1))
