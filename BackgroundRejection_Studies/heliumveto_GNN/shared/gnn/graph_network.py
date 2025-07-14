import torch.nn as nn
from heliumveto_GNN.shared.blocks.edge_block import EdgeBlock
from heliumveto_GNN.shared.blocks.node_block import NodeBlock
from heliumveto_GNN.shared.blocks.global_block import GlobalBlock

class GraphNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "edge":   EdgeBlock(node_dim, edge_dim, global_dim, hidden),
                "node":   NodeBlock(node_dim, edge_dim, global_dim, hidden),
                "global": GlobalBlock(node_dim, edge_dim, global_dim, hidden),
            })
            self.layers.append(layer)

    def forward(self, data):
        for l in self.layers:
            data = l["edge"](data)
            data = l["node"](data)
            data = l["global"](data)
        return data