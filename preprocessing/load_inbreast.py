"""Load INbreast dataset from DICOM files."""

import pydicom, numpy as np, torch
from pathlib import Path
import pandas as pd
from .pipeline import img_to_graph


def load_inbreast(dicom_dir, xls_path, cache_dir, ps=128, ts=1024):
    """
    Load INbreast DICOM files and convert to graphs.

    Args:
        dicom_dir: Path to ALL-IMGS/ directory containing .dcm files
        xls_path: Path to INbreast.xls metadata file
        cache_dir: Path to save cached .pt graph files
        ps: Patch size (default 128)
        ts: Target image size (default 1024)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load labels from XLS
    df = pd.read_excel(xls_path)
    lmap = {}
    for _, r in df.iterrows():
        try:
            br = int(r.get("Bi-Rads", 0))
        except:
            continue
        if br <= 2:
            lab = 0  # benign
        elif br >= 4:
            lab = 1  # malignant
        else:
            continue  # skip BI-RADS 3
        try:
            fn = str(int(float(r.get("File Name", ""))))
        except:
            fn = str(r.get("File Name", "")).strip().split(".")[0]
        lmap[fn] = {"label": lab, "pid": fn, "birads": br}

    # Process DICOM files
    cached_ids = {p.stem for p in cache_dir.glob("*.pt")}
    new = 0
    dicom_dir = Path(dicom_dir)
    for path in sorted(dicom_dir.glob("*.dcm")):
        prefix = path.stem.split("_")[0]
        info = lmap.get(prefix)
        if not info:
            for k in lmap:
                if k in path.stem:
                    info = lmap[k]
                    break
        if not info:
            continue
        fid = f"INB_{info['pid']}_{prefix}"
        if fid in cached_ids:
            continue
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            bits = ds.get("BitsStored", 14)
            mx = float(2 ** bits - 1)
            # Orientation normalization
            if ds.get("PhotometricInterpretation") == "MONOCHROME1":
                img = mx - img
            g = img_to_graph(img, info["label"], info["pid"], ps, ts)
            if g:
                torch.save(g, cache_dir / f"{fid}.pt")
                new += 1
        except Exception:
            pass

    total = list(cache_dir.glob("*.pt"))
    print(f"INbreast: {len(total)} graphs ({new} new)")
    return sorted(total)
