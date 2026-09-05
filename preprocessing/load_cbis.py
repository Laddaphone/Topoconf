"""Load CBIS-DDSM from JPEG files with CSV metadata."""
import cv2, numpy as np, torch, pandas as pd
from pathlib import Path
from .pipeline import img_to_graph

def load_cbis(cbis_dir, cache_dir, ps=128, ts=1024):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    jpeg_dir = next((d for d in Path(cbis_dir).rglob('jpeg') if d.is_dir()), None)
    if not jpeg_dir: raise FileNotFoundError('No jpeg/ directory')
    uid2f = {f.name: f for f in jpeg_dir.iterdir() if f.is_dir()}
    recs = []; seen = set()
    for cp in sorted(Path(cbis_dir).rglob('*.csv')):
        try: df = pd.read_csv(cp)
        except: continue
        for _, r in df.iterrows():
            raw = str(r.get('pathology', '')).upper()
            if 'MALIGNANT' in raw: lab = 1
            elif 'BENIGN' in raw: lab = 0
            else: continue
            pid = str(r.get('patient_id', '')).strip()
            view = str(r.get('image view', '')).strip()
            lat = str(r.get('left or right breast', '')).strip()
            fp = str(r.get('image file path', '')).strip().split('/')
            uid = fp[2] if len(fp) > 2 else None
            k = (pid, lat, view)
            if k not in seen: seen.add(k); recs.append(dict(pid=pid, label=lab, uid=uid))
    cached = {p.stem for p in cache_dir.glob('*.pt')}; new = 0
    for rec in recs:
        fid = f'CBIS_{rec["pid"]}'
        if fid in cached: continue
        folder = uid2f.get(rec['uid'])
        if not folder: continue
        jpgs = sorted(folder.glob('*.jpg'), key=lambda p: p.stat().st_size, reverse=True)
        if not jpgs: continue
        try:
            img = cv2.imread(str(jpgs[0]), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            g = img_to_graph(img.astype(np.float32), rec['label'], rec['pid'], ps, ts)
            if g: torch.save(g, cache_dir / f'{fid}.pt'); new += 1
        except: pass
    print(f'CBIS-DDSM: {len(list(cache_dir.glob("*.pt")))} graphs ({new} new)')
