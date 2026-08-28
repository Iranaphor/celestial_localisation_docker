"""Approximate covariance estimation from the solver residual.

This is a first-pass approximation: covariance scales with the final
residual cost and the number of usable observations. Replace with proper
Jacobian-based covariance propagation once measurement uncertainty modeling
is more mature (see project spec section 29).
"""
import numpy as np


def estimate_covariance(result, num_observations, base_variance_deg2=1.0):
    if num_observations == 0:
        scale = 1e6
    else:
        scale = max(1e-6, result.cost) / num_observations
    variance = base_variance_deg2 * (1.0 + scale)
    return np.diag([variance, variance, variance * 4.0])
