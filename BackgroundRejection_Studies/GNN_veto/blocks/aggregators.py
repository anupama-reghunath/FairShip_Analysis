from GNN_veto.blocks.abstract_module import AbstractModule
import torch

from torch_scatter import scatter_add
from torch_scatter import scatter_mean

def globals_to_nodes(graph):
    return graph.graph_globals[graph.batch]

def receiver_nodes_to_edges(graph):

    return graph.nodes[graph.receivers, :]

def sender_nodes_to_edges(graph):

    return graph.nodes[graph.senders, :]

def globals_to_edges(graph):
    return graph.graph_globals[graph.edgepos]

class EdgesToNodesAggregator(AbstractModule):
    """Agregates sent or received edges into the corresponding nodes."""
    def __init__(self, use_sent_edges=False):
        super(EdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges

    def forward(self, graph):
        if graph.nodes is not None and graph.nodes.size()[0] is not None:
            num_nodes = graph.nodes.size()[0]
        else:
            num_nodes = graph.num_nodes

        indices = graph.senders if self._use_sent_edges else graph.receivers
        out = graph.edges.new_zeros(num_nodes, graph.edges.shape[1])
        return scatter_add(graph.edges, indices, out=out, dim=0)




class EdgesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None):
        super(EdgesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1

    def forward(self, graph):
        if self._num_graphs is None:
            out = torch.sum(graph.edges, 0)
        else:
            out = scatter_add(graph.edges, graph.edgepos, dim=0)
        return out


class NodesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None):
        super(NodesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1

    def forward(self, graph):
        if self._num_graphs is None:
            out = torch.sum(graph.nodes, 0)
        else:

            out = scatter_add(graph.nodes, graph.batch, dim=0)

        return out
