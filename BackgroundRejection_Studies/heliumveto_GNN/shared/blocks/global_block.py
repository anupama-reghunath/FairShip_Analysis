# heliumveto_GNN/shared/blocks/global_block.py
import torch
import torch.nn as nn
from heliumveto_GNN.shared.blocks.aggregators import (
    agg_nodes_to_globals, agg_edges_to_globals
)

class GlobalBlock(nn.Module):
    """
    Update global feature U' = φ_u([U, ρ_v→u(V), ρ_e→u(E)])
    """

    def __init__(self, node_dim, edge_dim, global_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, global_dim)
        )

    def forward(self, data):
        # Aggregate over each graph in batch
        nodes_sum = agg_nodes_to_globals(data.nodes, data.batch)       # (B, node_dim)
        edges_sum = agg_edges_to_globals(data.edges, data.batch_edges) # (B, edge_dim)
        x = torch.cat([data.graph_globals, nodes_sum, edges_sum], dim=1)
        data.graph_globals = self.mlp(x)
        return data
