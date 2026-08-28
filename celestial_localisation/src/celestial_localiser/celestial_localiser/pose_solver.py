"""Solves for (latitude, longitude, heading) by minimizing angular residuals
between observed and ephemeris-predicted celestial unit vectors.

Heading (yaw) accounts for the fact that the camera/robot azimuth reference
frame is not necessarily aligned with true north.
"""
import numpy as np
from scipy.optimize import least_squares

from celestial_localizer.coordinate_transforms import az_el_to_unit_vector


def _residuals(state, observations, timestamp, ephemeris):
    latitude, longitude, heading = state
    errors = []
    for obs in observations:
        predicted = ephemeris.predict(obs['object_id'], timestamp, latitude, longitude)
        if predicted is None:
            continue
        pred_az, pred_el = predicted
        obs_u = az_el_to_unit_vector(obs['azimuth'] + heading, obs['elevation'])
        pred_u = az_el_to_unit_vector(pred_az, pred_el)
        weight = obs.get('confidence', 1.0)
        errors.append(weight * np.linalg.norm(obs_u - pred_u))
    if not errors:
        # No usable observations; return zero residual (no correction pressure).
        return [0.0]
    return errors


def solve(observations, timestamp, ephemeris, initial_state, robust_loss='soft_l1'):
    """initial_state = (latitude_deg, longitude_deg, heading_deg)."""
    result = least_squares(
        _residuals,
        x0=initial_state,
        args=(observations, timestamp, ephemeris),
        loss=robust_loss,
        bounds=([-90.0, -180.0, -360.0], [90.0, 180.0, 360.0]),
    )
    return result
