"""Verify patient-disjoint splits from saved .npz files."""

import numpy as np
from pathlib import Path
import sys


def verify_split(npz_path):
    """Check that test and cal patients do not overlap."""
    d = np.load(npz_path, allow_pickle=True)
    if "test_pids" not in d.files or "cal_pids" not in d.files:
        return None  # Cannot verify
    test = set(str(p) for p in d["test_pids"])
    cal = set(str(p) for p in d["cal_pids"])
    overlap = test & cal
    return len(overlap)


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    total = 0
    overlaps = 0
    for f in sorted(results_dir.glob("*.npz")):
        n = verify_split(f)
        if n is None:
            continue
        total += 1
        if n > 0:
            overlaps += 1
            print(f"  OVERLAP: {f.stem} ({n} shared patients)")
    print(f"Checked {total} files, {overlaps} overlaps")
    if overlaps == 0:
        print("PASS: All splits are patient-disjoint")
    else:
        print("FAIL: Patient overlap detected")
        sys.exit(1)
