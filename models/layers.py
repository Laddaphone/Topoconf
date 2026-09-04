import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GatedTopoFusion(nn.Module):
    """Gated fusion of visual and topological features."""
    def __init__(self, vis_d=128, topo_d=4, out_d=128):
        super().__init__()
        self.topo_mlp = nn.Sequential(nn.Linear(topo_d, vis_d), nn.LayerNorm(vis_d), nn.ELU())
        self.gate = nn.Sequential(nn.Linear(topo_d, vis_d), nn.Sigmoid())
        self.out = nn.Linear(vis_d, out_d)

    def forward(self, vis, topo):
        g = self.gate(topo)
        t = self.topo_mlp(topo)
        return self.out(g * vis + (1 - g) * t)


class GATBlk(nn.Module):
    """GAT block with residual connection and LayerNorm."""
    def __init__(self, di=128, do=128, heads=4, dropout=0.3):
        super().__init__()
        self.gat = GATConv(di, do // heads, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(do)
        self.res = nn.Linear(di, do) if di != do else nn.Identity()

    def forward(self, x, edge_index):
        return self.norm(F.elu(self.gat(x, edge_index)) + self.res(x))
