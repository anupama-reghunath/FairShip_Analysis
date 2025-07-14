import torch
from heliumveto_GNN.shared.blocks.edge_block import EdgeBlock
from heliumveto_GNN.shared.blocks.node_block import NodeBlock
from heliumveto_GNN.shared.blocks.global_block import GlobalBlock

def test_blocks_run(fake_graph):
    n_dim = fake_graph.nodes.shape[1]
    e_dim = fake_graph.edges.shape[1]
    g_dim = fake_graph.graph_globals.shape[1]

    fake_graph = EdgeBlock(n_dim, e_dim, g_dim)(fake_graph)
    fake_graph = NodeBlock(n_dim, e_dim, g_dim)(fake_graph)
    fake_graph = GlobalBlock(n_dim, e_dim, g_dim)(fake_graph)

    # after updates the dims should stay the same
    assert fake_graph.nodes.shape[1]  == n_dim
    assert fake_graph.edges.shape[1]  == e_dim
    assert fake_graph.graph_globals.shape[1] == g_dim
    # and tensors remain finite
    assert torch.isfinite(fake_graph.nodes).all()
