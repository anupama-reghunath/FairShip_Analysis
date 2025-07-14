import torch
from torch_geometric.loader import DataLoader
from heliumveto_GNN.binary.model.gnn_model import GNNBinary

def test_train_one_step(fake_graph):
    fake_graph.y = torch.tensor([1.0])   # binary label
    loader = DataLoader([fake_graph], batch_size=1)

    n_dim, e_dim, g_dim = fake_graph.nodes.shape[1], fake_graph.edges.shape[1], fake_graph.graph_globals.shape[1]
    model = GNNBinary(n_dim, e_dim, g_dim)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for batch in loader:
        opt.zero_grad()
        out = model(batch)
        loss = loss_fn(out.unsqueeze(0), torch.ones(1))
        loss.backward()
        opt.step()
    # parameters updated?  compare a weight tensor before/after
    assert loss < 1.0