"""Verify patient-disjoint splits from .npz files."""
import numpy as np, sys
from pathlib import Path

if __name__ == '__main__':
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    total = overlaps = 0
    for f in sorted(d.glob('*.npz')):
        data = np.load(f, allow_pickle=True)
        if 'test_pids' not in data.files or 'cal_pids' not in data.files: continue
        ov = set(str(p) for p in data['test_pids']) & set(str(p) for p in data['cal_pids'])
        total += 1
        if ov: overlaps += 1; print(f'  OVERLAP: {f.stem} ({len(ov)} patients)')
    print(f'Checked {total}, overlaps {overlaps}: {"PASS" if overlaps==0 else "FAIL"}')
