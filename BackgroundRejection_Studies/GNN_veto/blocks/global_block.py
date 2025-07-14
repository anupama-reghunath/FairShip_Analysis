from GNN_veto.blocks.abstract_module import AbstractModule
from GNN_veto.blocks.aggregators import EdgesToGlobalsAggregator
from GNN_veto.blocks.aggregators import NodesToGlobalsAggregator
import torch


class GlobalBlock(AbstractModule):


    def __init__(self, global_model_fn, use_edges=True, use_nodes=True, use_globals=True):
        super(GlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = EdgesToGlobalsAggregator()
            if self._use_nodes:
                self._nodes_aggregator = NodesToGlobalsAggregator()


    def forward(self, graph, global_model_kwargs=None):


        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph))

        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph))

        if self._use_globals:
            globals_to_collect.append(graph.graph_globals)


        collected_globals = torch.cat(globals_to_collect, axis=-1)



        updated_globals = self._global_model(collected_globals)

        graph.update({'graph_globals': updated_globals})

        return graph