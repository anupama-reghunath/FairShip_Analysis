import pytest
from heliumveto_GNN.shared.gnn.graphcoder import event_to_graph
from heliumveto_GNN.shared.util.geometry import XYZ

def test_empty_event_raises(empty_event):
    # The graph builder should complain
    with pytest.raises(ValueError, match="no SBT hits"):
        _ = event_to_graph(empty_event, XYZ)