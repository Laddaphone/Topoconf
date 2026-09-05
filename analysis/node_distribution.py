"""Report node and edge count distributions from cached graphs."""
import torch, numpy as np
from pathlib import Path

def graph_statistics(cache_dir):
    """Compute node/edge count distribution from .pt graph files."""
    nodes, edges = [], []
    for f in sorted(Path(cache_dir).glob('*.pt')):
        try:
            g = torch.load(f, map_location='cpu', weights_only=False)
            nodes.append(g.x.shape[0])
            edges.append(g.edge_index.shape[1] if hasattr(g, 'edge_index') else 0)
        except: pass
    nodes = np.array(nodes); edges = np.array(edges)
    return dict(n_graphs=len(nodes),
        nodes_mean=nodes.mean(), nodes_std=nodes.std(),
        nodes_min=int(nodes.min()), nodes_max=int(nodes.max()),
        nodes_q25=np.percentile(nodes,25), nodes_q50=np.percentile(nodes,50),
        nodes_q75=np.percentile(nodes,75),
        edges_mean=edges.mean(), edges_std=edges.std())
