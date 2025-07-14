# heliumveto_GNN/shared/gnn/feature_registry.py
import numpy as np

def energy_column(arr):           # arr shape (854,)
    return arr.reshape(-1, 1)     # (N,1)

def xyz_columns(xyz, mask):
    return xyz.T[mask]            # (N,3)

def phi_column(xyz, mask):
    from heliumveto_GNN.shared.util.geometry import phi_from_xy
    g = xyz.T[mask]
    phi = phi_from_xy(g[:,0], g[:,1]).reshape(-1,1)
    return phi

def delta_t_column(delta_t_slice):
    """delta_t_slice is (N_fired,) vector for the fired cells."""
    return delta_t_slice.reshape(-1, 1)