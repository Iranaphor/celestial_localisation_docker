import math

import pytest

from celestial_localiser.pose_solver import solve


class SyntheticEphemeris:
    def predict(self, object_id, timestamp, latitude, longitude):
        index = int(object_id[1:])
        base_azimuth = (17.0 + index * 73.0) % 360.0
        base_elevation = 35.0 + index * 3.0
        azimuth = (
            base_azimuth
            + (0.18 + 0.035 * index) * latitude
            + (0.55 - 0.025 * index) * longitude
            + timestamp * 0.0001
        )
        elevation = (
            base_elevation
            + (0.42 - 0.025 * index) * latitude
            + (0.13 + 0.018 * index) * longitude
        )
        return azimuth, elevation


def observations_for(state, timestamp=12345.0, elevation_bias=0.0):
    latitude, longitude, heading = state
    ephemeris = SyntheticEphemeris()
    observations = []
    for index in range(8):
        object_id = f's{index}'
        azimuth, elevation = ephemeris.predict(
            object_id,
            timestamp,
            latitude,
            longitude,
        )
        observations.append({
            'object_id': object_id,
            'azimuth': azimuth - heading,
            'elevation': elevation + elevation_bias * ((index % 3) - 1),
            'confidence': 1.0,
        })
    return observations


@pytest.mark.parametrize(
    'true_state, initial_state',
    (
        ((23.0, -70.0, 17.0), (20.0, -68.0, 10.0)),
        ((-41.0, 179.0, -32.0), (51.5, -0.1, 0.0)),
        ((72.0, -179.0, 45.0), (-51.5, 0.1, 0.0)),
    ),
)
def test_global_multistart_recovers_exact_synthetic_directions(true_state, initial_state):
    result = solve(
        observations_for(true_state),
        12345.0,
        SyntheticEphemeris(),
        initial_state,
        max_nfev=500,
        global_search=True,
    )

    assert result.valid is True
    assert result.starts_tried > 1
    assert result.max_residual < 1e-6
    assert result.x[0] == pytest.approx(true_state[0], abs=1e-4)
    assert result.x[1] == pytest.approx(true_state[1], abs=1e-4)
    assert result.x[2] == pytest.approx(true_state[2], abs=1e-4)


def test_solver_reports_insufficient_observations_instead_of_returning_previous_pose():
    result = solve(
        observations_for((23.0, -70.0, 17.0))[:2],
        12345.0,
        SyntheticEphemeris(),
        (51.5, -0.1, 0.0),
    )

    assert result.valid is False
    assert result.failure_reason == 'insufficient_observations'
    assert result.x.tolist() == [51.5, -0.1, 0.0]
    assert result.nfev == 0


def test_controlled_angular_bias_is_visible_in_residual_diagnostics():
    unbiased = solve(
        observations_for((23.0, -70.0, 17.0)),
        12345.0,
        SyntheticEphemeris(),
        (22.0, -69.0, 16.0),
        max_nfev=500,
    )
    biased = solve(
        observations_for((23.0, -70.0, 17.0), elevation_bias=0.2),
        12345.0,
        SyntheticEphemeris(),
        (22.0, -69.0, 16.0),
        max_nfev=500,
    )

    assert unbiased.valid is True
    assert unbiased.max_residual < 1e-6
    assert biased.max_residual > math.radians(0.01)
    assert biased.valid is False
    assert biased.failure_reason == 'optimizer_not_converged'