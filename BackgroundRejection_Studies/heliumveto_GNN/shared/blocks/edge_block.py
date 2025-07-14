# heliumveto_GNN/shared/blocks/edge_block.py
import torch
import torch.nn as nn

class EdgeBlock(nn.Module):
    """
    Update edge features E'  =  φ_e([E, V_sender, V_receiver, U])
    """

    def __init__(self, node_dim, edge_dim, global_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim*2 + edge_dim + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, edge_dim)
        )

    def forward(self, data):
        # data.edge_index: (2, E)
        send, recv = data.edge_index
        src, dst = data.nodes[send], data.nodes[recv]          # (E, node_dim)
        u = data.graph_globals[data.batch_edges]               # align global per-edge
        x = torch.cat([data.edges, src, dst, u], dim=1)
        data.edges = self.mlp(x)
        return data
