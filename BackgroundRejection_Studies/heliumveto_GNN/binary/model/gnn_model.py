import torch.nn as nn
from heliumveto_GNN.shared.gnn.graph_network import GraphNetwork

class GNNBinary(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden=64):
        super().__init__()
        self.gn = GraphNetwork(node_dim, edge_dim, global_dim, hidden)
        self.head = nn.Linear(global_dim, 1)

    def forward(self, data):
        g = self.gn(data)
        logits = self.head(g.graph_globals)
        return logits.squeeze(-1)            # shape (batch,)