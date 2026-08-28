"""Az/elevation <-> unit-vector helpers used to avoid azimuth wraparound issues."""
import numpy as np


def az_el_to_unit_vector(azimuth_deg, elevation_deg):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z])


def angular_separation(u_a, u_b):
    dot = np.clip(np.dot(u_a, u_b), -1.0, 1.0)
    return np.arccos(dot)
