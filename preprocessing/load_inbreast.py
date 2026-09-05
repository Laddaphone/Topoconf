"""Load INbreast from DICOM with orientation normalization."""
import pydicom, numpy as np, torch, pandas as pd
from pathlib import Path
from .pipeline import img_to_graph

def load_inbreast(dicom_dir, xls_path, cache_dir, ps=128, ts=1024):
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(xls_path); lmap = {}
    for _, r in df.iterrows():
        try: br = int(r.get('Bi-Rads', 0))
        except: continue
        if br <= 2: lab = 0
        elif br >= 4: lab = 1
        else: continue
        try: fn = str(int(float(r.get('File Name', ''))))
        except: fn = str(r.get('File Name', '')).strip().split('.')[0]
        lmap[fn] = {'label': lab, 'pid': fn}
    cached = {p.stem for p in cache_dir.glob('*.pt')}; new = 0
    for path in sorted(Path(dicom_dir).glob('*.dcm')):
        prefix = path.stem.split('_')[0]
        info = lmap.get(prefix)
        if not info:
            for k in lmap:
                if k in path.stem: info = lmap[k]; break
        if not info: continue
        fid = f'INB_{info["pid"]}_{prefix}'
        if fid in cached: continue
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            mx = float(2**ds.get('BitsStored', 14) - 1)
            if ds.get('PhotometricInterpretation') == 'MONOCHROME1': img = mx - img
            g = img_to_graph(img, info['label'], info['pid'], ps, ts)
            if g: torch.save(g, cache_dir / f'{fid}.pt'); new += 1
        except: pass
    print(f'INbreast: {len(list(cache_dir.glob("*.pt")))} graphs ({new} new)')
