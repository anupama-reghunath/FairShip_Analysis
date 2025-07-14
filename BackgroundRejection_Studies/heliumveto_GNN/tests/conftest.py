# heliumveto_GNN/tests/conftest.py
import numpy as np
import pytest
import torch

from heliumveto_GNN.shared.util.geometry import XYZ
from heliumveto_GNN.shared.gnn.graphcoder import event_to_graph

@pytest.fixture(scope="session")
def fake_event():
    """Return a (854, 10) array with exactly 7 fired cells."""
    arr = np.zeros((854, 10), dtype=np.float32)
    arr[:7, 0] = 0.05          # column 0 = energy
    return arr

@pytest.fixture(scope="session")
def fake_graph(fake_event):
    """Convert the fake event into a torch_geometric Data graph."""
    g = event_to_graph(fake_event, XYZ)
    # annotate batch tensors the blocks expect
    g.batch = torch.zeros(g.nodes.size(0),  dtype=torch.long)
    g.batch_edges = torch.zeros(g.edges.size(0), dtype=torch.long)
    return g

@pytest.fixture(scope="session")
def empty_event():
    """854 cells but none fired (energy == 0)."""
    import numpy as np
    arr = np.zeros((854, 10), dtype=np.float32)
    return arr