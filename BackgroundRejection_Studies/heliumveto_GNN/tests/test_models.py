import torch
from heliumveto_GNN.binary.model.gnn_model import GNNBinary
from heliumveto_GNN.multiclass.model.gnn_model import GNNMulti

def _dims(g):
    return g.nodes.shape[1], g.edges.shape[1], g.graph_globals.shape[1]

def test_binary_forward(fake_graph):
    n_dim, e_dim, g_dim = _dims(fake_graph)
    model = GNNBinary(n_dim, e_dim, g_dim)
    out = model(fake_graph)
    # Accept scalar or length-1 vector
    assert out.numel() == 1
    assert torch.isfinite(out).all()

def test_multiclass_forward(fake_graph):
    n_dim, e_dim, g_dim = _dims(fake_graph)
    model = GNNMulti(n_dim, e_dim, g_dim)
    logits = model(fake_graph.unsqueeze(0)) if hasattr(fake_graph, "unsqueeze") else model(fake_graph)
    assert logits.shape == (1, 3)