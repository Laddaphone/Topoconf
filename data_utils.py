import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def patient_disjoint_split(graphs, labels, seed=42):
    """Patient-disjoint 4-way split: 50/15/15/20."""
    pids = np.array([str(g.pid_hash.item()) for g in graphs])
    upids = np.unique(pids)
    yp = {}
    for pid, lab in zip(pids, labels):
        yp[pid] = lab
    pl = np.array([yp[p] for p in upids])
    for attempt in range(20):
        s = seed + attempt * 111
        try:
            t1, r1 = next(GroupShuffleSplit(1, test_size=0.50, random_state=s).split(np.arange(len(upids)), pl, upids))
            rp, rl = upids[r1], pl[r1]
            v1, r2 = next(GroupShuffleSplit(1, test_size=0.70, random_state=s).split(np.arange(len(rp)), rl, rp))
            rp2, rl2 = rp[r2], rl[r2]
            c1, t2 = next(GroupShuffleSplit(1, test_size=0.571, random_state=s).split(np.arange(len(rp2)), rl2, rp2))
            sets = {'tr': set(upids[t1]), 'vl': set(rp[v1]), 'cl': set(rp2[c1]), 'ts': set(rp2[t2])}
            result = tuple([i for i, g in enumerate(graphs) if str(g.pid_hash.item()) in sets[k]] for k in ['tr', 'vl', 'cl', 'ts'])
            if all(len(set(labels[i] for i in idx)) >= 2 for idx in result if len(idx) > 0):
                return result
        except:
            continue
    raise RuntimeError(f'Split failed after 20 attempts, seed={seed}')
