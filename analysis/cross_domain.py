"""
Cross-domain evaluation protocol.
Train on CBIS-DDSM, evaluate on INbreast.
Checkpoints: checkpoints/cross_domain/X_CBIS2INB_*.pt
"""
import numpy as np

def cross_domain_eval(npz_path):
    """Extract cross-domain results from saved .npz."""
    d = np.load(npz_path, allow_pickle=True)
    ty = d['ty']; tp = d['tp']; tpred = d['tpred']
    from sklearn.metrics import roc_auc_score
    try: auc = roc_auc_score(ty, tp[:,1])
    except: auc = float('nan')
    return dict(
        auc=auc, acc=(tpred==ty).mean(),
        sens=((tpred==1)&(ty==1)).sum() / max((ty==1).sum(), 1),
        spec=((tpred==0)&(ty==0)).sum() / max((ty==0).sum(), 1),
        n=len(ty), n_mal=int((ty==1).sum()), n_ben=int((ty==0).sum()))
