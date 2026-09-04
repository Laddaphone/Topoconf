import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from .backbone import ResNetBackbone
from .layers import GatedTopoFusion, GATBlk


class TopoConfGNN(nn.Module):
    """
    Evidential Deep Learning model for graph-based mammography.
    forward() returns Dirichlet alphas directly (softplus + 1).
    """
    def __init__(self, topo=True):
        super().__init__()
        self.use_topo = topo
        self.backbone = ResNetBackbone(d_out=128)
        if topo:
            self.fusion = GatedTopoFusion(128, 4, 128)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(128, 128), nn.LayerNorm(128), nn.ELU())
        self.proj = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.3))
        self.gats = nn.ModuleList([GATBlk() for _ in range(3)])
        self.readout = nn.Sequential(
            nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.3))
        self.evi = nn.Sequential(nn.Linear(128, 2), nn.Softplus())

    def forward(self, data):
        vis = self.backbone(data.x)
        if self.use_topo and hasattr(data, 'topo'):
            h = self.fusion(vis, data.topo)
        else:
            h = self.fusion(vis)
        h = self.proj(h)
        for gat in self.gats:
            h = gat(h, data.edge_index)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        hg = self.readout(torch.cat([
            global_mean_pool(h, batch),
            global_max_pool(h, batch)], dim=-1))
        return self.evi(hg) + 1.0
