"""Raw Image -> Graph Pipeline. Extracted from Topoconf_FInal_v5.ipynb."""
import numpy as np, cv2, torch, gudhi
from scipy.spatial import cKDTree
from scipy.stats import entropy
from torch_geometric.data import Data

PATCH_SIZE = 128; TARGET_SIZE = 1024; SPATIAL_K = 8; TOPO_K = 4; TOPO_EDGE_THRESH = 0.7

def breast_mask(img):
    """Otsu + morphological opening + largest connected component."""
    u8 = (img / max(img.max(), 1) * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nl, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if nl > 1: m = (lb == 1 + np.argmax(st[1:, 4])).astype(np.uint8) * 255
    return m > 0

def resize_pad(img, sz=1024):
    """Resize longest side to sz, zero-pad to square."""
    h, w = img.shape[:2]; s = sz / max(h, w)
    r = cv2.resize(img.astype(np.float32), (int(w*s), int(h*s)))
    dw, dh = sz - r.shape[1], sz - r.shape[0]
    return cv2.copyMakeBorder(r, dh//2, dh-dh//2, dw//2, dw-dw//2, cv2.BORDER_CONSTANT, value=0)

def clahe(img, clip=2.0, grid=(8,8)):
    """CLAHE contrast enhancement. clipLimit=2.0, tileGridSize=(8,8)."""
    u8 = (img / max(img.max(), 1) * 255).astype(np.uint8)
    return cv2.createCLAHE(clip, grid).apply(u8).astype(np.float32) / 255.0

def tda_features(patch):
    """4 TDA features via GUDHI cubical persistent homology."""
    p = patch.astype(np.float64)
    cc = gudhi.CubicalComplex(dimensions=p.shape, top_dimensional_cells=p.flatten())
    pers = cc.persistence(min_persistence=0.001)
    feats = []
    for dim in [0, 1]:
        pairs = [x[1] for x in pers if x[0] == dim]
        if not pairs: feats.extend([0.0, 0.0]); continue
        b = np.array([x[0] for x in pairs])
        d = np.array([x[1] for x in pairs]); d[d == np.inf] = 1.0
        lt = d - b; wa = lt.sum()
        feats.extend([entropy(lt / (wa + 1e-10)), wa])
    return torch.tensor(feats, dtype=torch.float32)

def img_to_graph(img, label, pid_str, ps=128, ts=1024):
    """Full pipeline: resize -> mask -> CLAHE -> patches -> TDA -> edges -> PyG Data."""
    ir = resize_pad(img, ts); mask = breast_mask(ir)
    ic = clahe(ir); it = np.clip(ir / max(ir.max(), 1), 0, 1.0)
    stride = ps // 2; H, W = ic.shape
    pc, pt, co = [], [], []
    for y in range(0, H - ps + 1, stride):
        for x in range(0, W - ps + 1, stride):
            if np.mean(mask[y:y+ps, x:x+ps]) < 0.1: continue
            pc.append(torch.from_numpy(ic[y:y+ps, x:x+ps]).unsqueeze(0))
            pt.append(tda_features(it[y:y+ps, x:x+ps]))
            co.append([y + ps/2, x + ps/2])
    n = len(pc)
    if n < 2: return None
    pos = np.array(co); topo = torch.stack(pt).numpy()
    tree = cKDTree(pos)
    _, sp_idx = tree.query(pos, k=min(SPATIAL_K+1, n))
    sp = set()
    for i, nb in enumerate(sp_idx):
        for j in nb:
            if i != j: sp.add((i, j))
    tn = topo / (np.linalg.norm(topo, axis=1, keepdims=True) + 1e-10)
    ts2 = tn @ tn.T; te = set()
    for i in range(n):
        s = ts2[i].copy(); s[i] = -1
        for j in np.argsort(s)[-TOPO_K:]:
            if s[j] >= TOPO_EDGE_THRESH: te.add((i,j)); te.add((j,i))
    ae = sp | te
    if not ae: return None
    return Data(x=torch.stack(pc).float(), topo=torch.stack(pt).float(),
        edge_index=torch.tensor(list(ae)).t().contiguous(),
        y=torch.tensor([label]),
        pid_hash=torch.tensor(hash(pid_str) % (2**62), dtype=torch.long))
