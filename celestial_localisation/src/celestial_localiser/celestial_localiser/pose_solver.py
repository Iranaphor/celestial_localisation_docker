"""Solve for latitude, longitude and heading from celestial observations."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from celestial_localiser.coordinate_transforms import az_el_to_unit_vector


@dataclass
class SolveResult:
    """Serializable solver outcome used by the ROS node and diagnostics."""

    x: np.ndarray
    cost: float
    status: int
    message: str
    nfev: int
    total_nfev: int
    optimality: float
    residuals: np.ndarray
    success: bool
    valid: bool
    failure_reason: str
    initial_state: tuple[float, float, float]
    selected_start: tuple[float, float, float]
    starts_tried: int
    boundary_solution: bool
    start_diagnostics: list[dict]

    @property
    def max_residual(self):
        if self.residuals.size == 0:
            return float('nan')
        return float(np.max(np.abs(self.residuals)))

    @property
    def rms_residual(self):
        if self.residuals.size == 0:
            return float('nan')
        return float(np.sqrt(np.mean(np.square(self.residuals))))


def _residuals(state, observations, timestamp, ephemeris):
    latitude, longitude, heading = state
    errors = []
    if hasattr(ephemeris, 'predict_many'):
        predictions = ephemeris.predict_many(
            [obs['object_id'] for obs in observations],
            timestamp,
            latitude,
            longitude,
        )
    else:
        predictions = [
            ephemeris.predict(obs['object_id'], timestamp, latitude, longitude)
            for obs in observations
        ]
    for obs, predicted in zip(observations, predictions):
        if predicted is None:
            continue
        pred_az, pred_el = predicted
        obs_u = az_el_to_unit_vector(obs['azimuth'] + heading, obs['elevation'])
        pred_u = az_el_to_unit_vector(pred_az, pred_el)
        weight = obs.get('confidence', 1.0)
        errors.append(weight * np.linalg.norm(obs_u - pred_u))
    if not errors:
        return []
    return errors


def _global_initial_states(initial_state):
    starts = [tuple(float(value) for value in initial_state)]
    for latitude in (-60.0, 0.0, 60.0):
        for longitude in (-135.0, -45.0, 45.0, 135.0):
            for heading in (0.0, 180.0):
                candidate = (latitude, longitude, heading)
                if candidate not in starts:
                    starts.append(candidate)
    return starts


def _boundary_solution(state, tolerance=1e-6):
    lower = np.asarray([-90.0, -180.0, -360.0])
    upper = np.asarray([90.0, 180.0, 360.0])
    return bool(
        np.any(np.isclose(state, lower, atol=tolerance, rtol=0.0))
        or np.any(np.isclose(state, upper, atol=tolerance, rtol=0.0))
    )


def _invalid_result(
    initial_state,
    reason,
    message,
    starts_tried=0,
    start_diagnostics=None,
):
    state = tuple(float(value) for value in initial_state)
    return SolveResult(
        x=np.asarray(state, dtype=np.float64),
        cost=float('inf'),
        status=-1,
        message=message,
        nfev=0,
        total_nfev=0,
        optimality=float('inf'),
        residuals=np.asarray([], dtype=np.float64),
        success=False,
        valid=False,
        failure_reason=reason,
        initial_state=state,
        selected_start=state,
        starts_tried=starts_tried,
        boundary_solution=False,
        start_diagnostics=list(start_diagnostics or []),
    )


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _start_diagnostic(start, result=None, error=None):
    diagnostic = {
        'initial_state': [float(value) for value in start],
        'final_state': None,
        'cost': None,
        'status': None,
        'message': '',
        'nfev': 0,
        'optimality': None,
        'residuals': [],
        'rms_residual': None,
        'max_residual': None,
        'success': False,
        'failure_reason': '',
        'exception': '',
        'selected': False,
    }
    if error is not None:
        diagnostic['failure_reason'] = 'optimizer_exception'
        diagnostic['exception'] = str(error)
        return diagnostic

    final_state = np.asarray(result.x, dtype=np.float64)
    residuals = np.asarray(result.fun, dtype=np.float64).reshape(-1)
    diagnostic.update({
        'final_state': [
            _finite_float(value) for value in final_state
        ] if np.all(np.isfinite(final_state)) else None,
        'cost': _finite_float(result.cost),
        'status': int(result.status),
        'message': str(result.message),
        'nfev': int(result.nfev),
        'optimality': _finite_float(result.optimality),
        'residuals': [
            _finite_float(value) for value in residuals
        ] if np.all(np.isfinite(residuals)) else [],
        'success': bool(result.success),
        'failure_reason': '' if result.success else 'optimizer_not_converged',
    })
    if diagnostic['residuals']:
        residual_array = np.asarray(diagnostic['residuals'], dtype=np.float64)
        diagnostic['rms_residual'] = float(
            np.sqrt(np.mean(np.square(residual_array)))
        )
        diagnostic['max_residual'] = float(np.max(np.abs(residual_array)))
    return diagnostic


def solve(
    observations,
    timestamp,
    ephemeris,
    initial_state,
    robust_loss='soft_l1',
    max_nfev=30,
    *,
    min_observations=3,
    global_search=False,
    max_starts=25,
    initial_states=None,
):
    """Solve from one or a deterministic bounded set of initial states."""
    if len(initial_state) != 3 or not np.all(np.isfinite(initial_state)):
        raise ValueError('initial_state must contain three finite values')
    if min_observations < 1:
        raise ValueError('min_observations must be positive')
    if max_starts < 1:
        raise ValueError('max_starts must be positive')

    original_initial_state = tuple(float(value) for value in initial_state)
    if len(observations) < min_observations:
        return _invalid_result(
            original_initial_state,
            'insufficient_observations',
            f'need at least {min_observations} usable observations; got {len(observations)}',
        )
    initial_state_array = np.asarray(initial_state, dtype=np.float64)
    if not _residuals(initial_state_array, observations, timestamp, ephemeris):
        return _invalid_result(
            original_initial_state,
            'no_supported_observations',
            'none of the observations produced an ephemeris prediction',
        )

    if initial_states is not None:
        starts = [tuple(float(value) for value in state) for state in initial_states]
    elif global_search:
        starts = _global_initial_states(original_initial_state)
    else:
        starts = [original_initial_state]
    if not starts:
        return _invalid_result(original_initial_state, 'no_initial_states', 'no initial states supplied')
    starts = starts[:max_starts]

    candidates = []
    start_diagnostics = []
    start_errors = []
    for start_index, start in enumerate(starts):
        try:
            result = least_squares(
                _residuals,
                x0=start,
                args=(observations, timestamp, ephemeris),
                loss=robust_loss,
                bounds=([-90.0, -180.0, -360.0], [90.0, 180.0, 360.0]),
                max_nfev=max_nfev,
            )
        except (FloatingPointError, RuntimeError, ValueError) as error:
            start_errors.append(str(error))
            start_diagnostics.append(_start_diagnostic(start, error=error))
            continue
        diagnostic = _start_diagnostic(start, result=result)
        start_diagnostics.append(diagnostic)
        if np.all(np.isfinite(result.x)) and np.isfinite(result.cost):
            candidates.append((result, start, start_index))

    if not candidates:
        return _invalid_result(
            original_initial_state,
            'optimizer_exception' if start_errors else 'non_finite_result',
            (
                f'all optimiser starts failed: {start_errors[0]}'
                if start_errors
                else 'all optimiser starts returned non-finite results'
            ),
            starts_tried=len(starts),
            start_diagnostics=start_diagnostics,
        )

    best_result, selected_start, selected_index = min(
        candidates,
        key=lambda candidate: candidate[0].cost,
    )
    start_diagnostics[selected_index]['selected'] = True
    valid = bool(best_result.success)
    return SolveResult(
        x=np.asarray(best_result.x, dtype=np.float64),
        cost=float(best_result.cost),
        status=int(best_result.status),
        message=str(best_result.message),
        nfev=int(best_result.nfev),
        total_nfev=sum(int(candidate[0].nfev) for candidate in candidates),
        optimality=float(best_result.optimality),
        residuals=np.asarray(best_result.fun, dtype=np.float64),
        success=bool(best_result.success),
        valid=valid,
        failure_reason='' if valid else 'optimizer_not_converged',
        initial_state=original_initial_state,
        selected_start=tuple(float(value) for value in selected_start),
        starts_tried=len(starts),
        boundary_solution=_boundary_solution(best_result.x),
        start_diagnostics=start_diagnostics,
    )
