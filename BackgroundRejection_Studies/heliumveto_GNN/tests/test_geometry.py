import numpy as np
from heliumveto_GNN.shared.util.geometry import phi_from_xy

def test_phi_basic():
    assert phi_from_xy(0, -1) == 0           # bottom centre
    assert np.isclose(phi_from_xy(1, 0), 90) # +x axis → 90°