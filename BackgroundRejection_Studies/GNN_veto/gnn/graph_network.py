from GNN_veto.blocks.abstract_module import AbstractModule
from GNN_veto.blocks.edge_block import EdgeBlock
from GNN_veto.blocks.global_block import GlobalBlock
from GNN_veto.blocks.node_block import NodeBlock
import torch
from torch_geometric.nn.models import MLP
from torch.nn import Linear, Sigmoid
from torch_geometric.nn import knn
from torch_geometric.nn.pool.select import SelectTopK
import contextlib



class GraphNetwork(AbstractModule):

    def __init__(self, edge_model, node_model, use_globals, global_model=None, hidden_size=8):

        super(GraphNetwork, self).__init__()
        self._use_globals = use_globals
        with self._enter_variable_scope():
            self._edge_block = EdgeBlock(edge_model_fn=edge_model)
            self._node_block = NodeBlock(node_model_fn=node_model)
            if self._use_globals:
                self._global_block = GlobalBlock(global_model_fn=global_model)



    def forward(self, graph):

        node_input = self._edge_block(graph)

        global_input = self._node_block(node_input)

        if self._use_globals:
            return self._global_block(global_input)

        else:
            return global_input

