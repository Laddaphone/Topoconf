"""Load CBIS-DDSM dataset from JPEG files."""

import cv2, numpy as np, torch
from pathlib import Path
import pandas as pd
from .pipeline import img_to_graph


def load_cbis(cbis_dir, cache_dir, ps=128, ts=1024):
    """
    Load CBIS-DDSM images and convert to graphs.

    Args:
        cbis_dir: Path to CBIS-DDSM root (contains jpeg/ and CSV files)
        cache_dir: Path to save cached .pt graph files
        ps: Patch size (default 128)
        ts: Target image size (default 1024)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find JPEG directory
    jpeg_dir = None
    for d in Path(cbis_dir).rglob("jpeg"):
        if d.is_dir():
            jpeg_dir = d
            break
    if not jpeg_dir:
        raise FileNotFoundError("No jpeg/ directory found")

    uid2f = {f.name: f for f in jpeg_dir.iterdir() if f.is_dir()}

    # Parse CSV metadata
    recs = []
    for pattern in ["mass_case*", "calc_case*"]:
        for cp in sorted(Path(cbis_dir).rglob(pattern)):
            if cp.suffix != ".csv":
                continue
            df = pd.read_csv(cp)
            for _, r in df.iterrows():
                raw = str(r.get("pathology", "")).upper()
                if "MALIGNANT" in raw:
                    lab = 1
                elif "BENIGN" in raw:
                    lab = 0
                else:
                    continue
                pid = str(r.get("patient_id", "")).strip()
                view = str(r.get("image view", "")).strip()
                lat = str(r.get("left or right breast", "")).strip()
                fp = str(r.get("image file path", "")).strip()
                parts = fp.split("/")
                uid = parts[2] if len(parts) > 2 else None
                recs.append({"pid": pid, "view": view,
                             "lat": lat, "label": lab, "uid": uid})

    # Deduplicate
    seen = set()
    unique = []
    for r in recs:
        k = (r["pid"], r["lat"], r["view"])
        if k not in seen:
            seen.add(k)
            unique.append(r)

    # Build graphs
    cached_ids = {p.stem for p in cache_dir.glob("*.pt")}
    new = 0
    for rec in unique:
        fid = f"CBIS_{rec['pid']}_{rec['lat']}_{rec['view']}"
        if fid in cached_ids:
            continue
        folder = uid2f.get(rec["uid"])
        if not folder:
            continue
        jpgs = sorted(folder.glob("*.jpg"),
                      key=lambda p: p.stat().st_size, reverse=True)
        if not jpgs:
            continue
        try:
            img = cv2.imread(str(jpgs[0]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            g = img_to_graph(img.astype(np.float32), rec["label"],
                             rec["pid"], ps, ts)
            if g:
                torch.save(g, cache_dir / f"{fid}.pt")
                new += 1
        except Exception:
            pass

    total = list(cache_dir.glob("*.pt"))
    print(f"CBIS-DDSM: {len(total)} graphs ({new} new)")
    return sorted(total)
