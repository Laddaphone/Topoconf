"""Split conformal prediction + TACP/EACP adaptive variants."""
import numpy as np

def nonconformity_score(probs, labels):
    return 1.0 - probs[np.arange(len(labels)), labels]

def calibrate_threshold(cal_scores, alpha=0.05):
    n = len(cal_scores)
    return np.quantile(cal_scores, min(np.ceil((n+1)*(1-alpha))/n, 1.0))

def prediction_set(probs, q_hat):
    sets = []
    for i in range(len(probs)):
        s = set(np.where(probs[i] >= 1 - q_hat)[0])
        if not s: s = {np.argmax(probs[i])}
        sets.append(s)
    return sets

def adaptive_threshold(q_base, difficulty, gamma):
    return q_base * (1 + gamma * difficulty)

def evaluate_coverage(pred_sets, labels):
    covered = np.array([labels[i] in pred_sets[i] for i in range(len(labels))])
    sizes = np.array([len(s) for s in pred_sets])
    r = {'marginal': covered.mean(), 'mean_set_size': sizes.mean()}
    for c in np.unique(labels): r[f'class_{c}'] = covered[labels==c].mean()
    return r
