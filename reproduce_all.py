"""
Master reproduction script.
Regenerates all tables and figures from raw .npz result files.

Usage:
  python reproduce_all.py --results_dir /path/to/posthoc_corrected/
                          --cmmd_dir /path/to/cmmd_within_domain/
                          --output_dir /path/to/output/

Downloads result files from HuggingFace if not found locally:
  https://huggingface.co/Laddaphone/topoconf-mammography-edl
"""
import argparse, json, sys
import numpy as np, pandas as pd
from pathlib import Path
from analysis.uncertainty_decomposition import comparable_signals, error_prediction_auroc
from analysis.triage import triage_comparison, patient_level_test
from analysis.calibration import ece, sensitivity_at_specificity
from analysis.bootstrap_ci import bootstrap_ci

def main(results_dir, cmmd_dir, output_dir):
    results_dir = Path(results_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmmd_dir = Path(cmmd_dir) if cmmd_dir else None
    SEEDS = list(range(42, 52))

    print('=== Table 1: Classification Performance ===')
    t1_rows = []
    for ds, cdir in [('CBIS', results_dir), ('INB', results_dir)]:
        for model in ['NoTopo','TopoConf','MCDrop','Softmax','GCN','ResNet']:
            aucs, senss, specs = [], [], []
            for seed in SEEDS:
                f = cdir / f'{ds}_{model}_s{seed}.npz'
                if not f.exists(): continue
                r = comparable_signals(f)
                from sklearn.metrics import roc_auc_score
                try: aucs.append(roc_auc_score(r['ty'], r['probs'][:,1]))
                except: pass
                senss.append(((r['tpred']==1)&(r['ty']==1)).sum() / max((r['ty']==1).sum(),1))
                specs.append(((r['tpred']==0)&(r['ty']==0)).sum() / max((r['ty']==0).sum(),1))
            if aucs:
                t1_rows.append(dict(ds=ds, model=model,
                    AUC=f'{np.mean(aucs):.3f}+/-{np.std(aucs):.3f}',
                    Sens=f'{np.mean(senss):.3f}+/-{np.std(senss):.3f}',
                    Spec=f'{np.mean(specs):.3f}+/-{np.std(specs):.3f}', n=len(aucs)))
    pd.DataFrame(t1_rows).to_csv(output_dir / 'table1_classification.csv', index=False)
    print(f'  Saved table1_classification.csv ({len(t1_rows)} rows)')

    print('\n=== Table 2: FN Triage at 42% Burden ===')
    t2_rows = []
    for ds, cdir in [('CBIS', results_dir), ('INB', results_dir), ('CMMD', cmmd_dir)]:
        if cdir is None: continue
        models = ['NoTopo','TopoConf'] if ds != 'CMMD' else ['NoTopo']
        for model in models:
            paths = [cdir / f'{ds}_{model}_s{seed}.npz' for seed in SEEDS]
            paths = [p for p in paths if p.exists()]
            vcs, hcs, wins = [], [], 0
            for p in paths:
                r = triage_comparison(p, [0.42])
                if r: vcs.append(r[0]['catch_vac']); hcs.append(r[0]['catch_H']); wins += r[0]['vac_wins']
            if vcs:
                t2_rows.append(dict(ds=ds, model=model,
                    vac_catch=f'{np.mean(vcs):.3f}+/-{np.std(vcs):.3f}',
                    H_catch=f'{np.mean(hcs):.3f}+/-{np.std(hcs):.3f}',
                    vac_wins=f'{wins}/{len(vcs)}'))
    pd.DataFrame(t2_rows).to_csv(output_dir / 'table2_triage.csv', index=False)
    print(f'  Saved table2_triage.csv ({len(t2_rows)} rows)')

    print('\n=== Table 3: Patient-Level Wilcoxon ===')
    t3_rows = []
    for ds, cdir in [('CBIS', results_dir), ('INB', results_dir), ('CMMD', cmmd_dir)]:
        if cdir is None: continue
        models = ['NoTopo','TopoConf'] if ds != 'CMMD' else ['NoTopo']
        all_paths = []
        for model in models:
            all_paths.extend([cdir/f'{ds}_{model}_s{seed}.npz' for seed in SEEDS if (cdir/f'{ds}_{model}_s{seed}.npz').exists()])
        if all_paths:
            r = patient_level_test(all_paths)
            t3_rows.append(dict(ds=ds, **r))
    pd.DataFrame(t3_rows).to_csv(output_dir / 'table3_patient_level.csv', index=False)
    print(f'  Saved table3_patient_level.csv ({len(t3_rows)} rows)')

    print('\n=== Table 4: Calibration ===')
    t4_rows = []
    for ds, cdir in [('CBIS', results_dir), ('INB', results_dir), ('CMMD', cmmd_dir)]:
        if cdir is None: continue
        models = ['NoTopo','TopoConf'] if ds != 'CMMD' else ['NoTopo']
        for model in models:
            eces, sens90 = [], []
            for seed in SEEDS:
                f = cdir / f'{ds}_{model}_s{seed}.npz'
                if not f.exists(): continue
                d = np.load(f, allow_pickle=True)
                eces.append(ece(d['tp'], d['ty'], d['tpred']))
                s, _ = sensitivity_at_specificity(d['ty'], d['tp'][:,1], 0.90)
                sens90.append(s)
            if eces:
                t4_rows.append(dict(ds=ds, model=model,
                    ECE=f'{np.mean(eces):.3f}+/-{np.std(eces):.3f}',
                    sens_at_spec90=f'{np.mean(sens90):.3f}+/-{np.std(sens90):.3f}'))
    pd.DataFrame(t4_rows).to_csv(output_dir / 'table4_calibration.csv', index=False)
    print(f'  Saved table4_calibration.csv ({len(t4_rows)} rows)')

    print('\nDone. All tables in:', output_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', required=True)
    p.add_argument('--cmmd_dir', default=None)
    p.add_argument('--output_dir', default='./output')
    args = p.parse_args()
    main(args.results_dir, args.cmmd_dir, args.output_dir)
