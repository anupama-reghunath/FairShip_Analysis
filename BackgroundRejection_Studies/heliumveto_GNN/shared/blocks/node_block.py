# heliumveto_GNN/shared/blocks/node_block.py
import torch
import torch.nn as nn
from heliumveto_GNN.shared.blocks.aggregators import agg_edges_to_nodes

class NodeBlock(nn.Module):
    """
    Update node features V' = φ_v([V, ρ_e→v(E), U])
    """

    def __init__(self, node_dim, edge_dim, global_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, node_dim)
        )

    def forward(self, data):
        send, recv = data.edge_index
        edge_msgs = data.edges                                           # (E, edge_dim)
        agg = agg_edges_to_nodes(edge_msgs, recv, data.nodes.size(0))    # (N, edge_dim)
        u = data.graph_globals[data.batch]                               # align per node
        x = torch.cat([data.nodes, agg, u], dim=1)
        data.nodes = self.mlp(x)
        return data