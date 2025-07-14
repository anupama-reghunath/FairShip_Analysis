import numpy as np

XYZ = np.load("/eos/experiment/ship/user/anupamar/NN_data/SBT_new_geo_XYZ.npy")                     # shape (3, 854) or similar

def phi_from_xy(x, y):
    """Azimuth in degrees with 0° at bottom centre."""
    import numpy as np
    phi = np.degrees(np.mod(np.arctan2(y, x), 2 * np.pi))
    return (phi + 90) % 360