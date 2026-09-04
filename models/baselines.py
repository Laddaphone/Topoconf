import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from .backbone import ResNetBackbone
from .layers import GATBlk


class BaseSM(nn.Module):
    """Softmax / MC-Dropout baseline. Returns raw logits."""
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone(d_out=128)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.ELU())
        self.gats = nn.ModuleList([GATBlk() for _ in range(3)])
        self.readout = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.3))
        self.head = nn.Linear(128, 2)

    def forward(self, data):
        h = self.proj(self.backbone(data.x))
        for gat in self.gats:
            h = gat(h, data.edge_index)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        hg = self.readout(torch.cat([
            global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1))
        return self.head(hg)


class BaseGCN(nn.Module):
    """GCN baseline. Returns raw logits."""
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone(d_out=128)
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.ELU())
        self.gcns = nn.ModuleList([GCNConv(128, 128) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(128) for _ in range(3)])
        self.readout = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Dropout(0.3))
        self.head = nn.Linear(128, 2)

    def forward(self, data):
        h = self.proj(self.backbone(data.x))
        for gcn, norm in zip(self.gcns, self.norms):
            h = norm(F.elu(gcn(h, data.edge_index)) + h)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        hg = self.readout(torch.cat([
            global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1))
        return self.head(hg)
