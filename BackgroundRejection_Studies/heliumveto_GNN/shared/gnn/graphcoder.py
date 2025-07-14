import numpy as np, torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from heliumveto_GNN.shared.util.config import GraphConfig
import heliumveto_GNN.shared.gnn.feature_registry as R

def event_to_graph(event: np.ndarray,
                   xyz:   np.ndarray,        # (3, n_cells)
                   cfg:   GraphConfig,
                   k:     int = 20) -> Data:
    """

    Column layout (deduced, not hard-coded):

        ┌─────────── n_cells ────────────┐
        |  energy[0..n-1] | hit_t[0..n-1] | vertex_x,y,z | vertex_time | ...
        └─────────────────────────────────┘
    """
    n_cells = xyz.shape[1]
    offset  = 0

    # ── per-cell slices --------------------------------------------------------
    energy = event[offset : offset + n_cells]          # (n_cells,)
    offset += n_cells

    hit_t  = None
    if cfg.use_delta_t:                                # need raw hit time
        hit_t  = event[offset : offset + n_cells]      # (n_cells,)
        offset += n_cells

    # ── global slices ----------------------------------------------------------
    vtx = event[offset : offset + 3]                   # (3,)  vertex position
    offset += 3
    vtx_time = event[offset] if cfg.use_delta_t else None  # scalar # (if cfg.use_delta_t=False, vertex_time isn’t stored )

    # ── fired-cell mask --------------------------------------------------------
    
    mask = energy > 0 

    if not np.any(mask):
        raise ValueError("No fired cells")

    # ── node feature matrix ----------------------------------------------------
    
    feats = []
    if cfg.use_energy:
        feats.append(R.energy_column(energy[mask]))
    if cfg.use_xyz:
        feats.append(R.xyz_columns(xyz, mask))
    if cfg.use_phi:
        feats.append(R.phi_column(xyz, mask))
    if cfg.use_delta_t and hit_t is not None:
        delta_t_all = hit_t - vtx_time                 # vectorised subtraction
        feats.append(R.delta_t_column(delta_t_all[mask]))

    node_feats = torch.as_tensor(np.hstack(feats), dtype=torch.float)  # (N,F)

    # ── connectivity & edge features ------------------------------------------
    xyz_fired = torch.as_tensor(xyz.T[mask], dtype=torch.float)         # (N,3)
    k_eff = min(k, len(xyz_fired) - 1) if len(xyz_fired) > 1 else 1
    edge_index = knn_graph(xyz_fired, k=k_eff)

    edge_attr = torch.empty((edge_index.size(1), 0))
    if cfg.add_edge_distance:
        send, recv = edge_index
        r = (xyz_fired[send] - xyz_fired[recv]).norm(dim=1, keepdim=True)
        edge_attr = r

    # ── global feature & Data object ------------------------------------------
    global_feat = torch.as_tensor(vtx.reshape(1, 3), dtype=torch.float)

    data = Data(nodes=node_feats,
                edge_index=edge_index,
                edges=edge_attr,
                graph_globals=global_feat)

    data.batch        = torch.zeros(len(node_feats), dtype=torch.long)
    data.batch_edges  = data.batch[edge_index[0]]
    return data