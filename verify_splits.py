"""Verify patient-disjoint splits from .npz files."""
import numpy as np, sys
from pathlib import Path

def verify(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    if 'test_pids' not in d.files or 'cal_pids' not in d.files: return None
    return len(set(str(p) for p in d['test_pids']) & set(str(p) for p in d['cal_pids']))

if __name__ == '__main__':
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    total = overlaps = 0
    for f in sorted(d.glob('*.npz')):
        n = verify(f)
        if n is None: continue
        total += 1
        if n > 0: overlaps += 1; print(f'  OVERLAP: {f.stem} ({n} patients)')
    print(f'Checked {total}, overlaps {overlaps}: {"PASS" if overlaps==0 else "FAIL"}')
