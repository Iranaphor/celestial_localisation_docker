import math
import json
from pathlib import Path

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
    assert len(result.start_diagnostics) == result.starts_tried
    assert sum(item['selected'] for item in result.start_diagnostics) == 1
    assert all(item['final_state'] is not None for item in result.start_diagnostics)
    assert all(item['residuals'] for item in result.start_diagnostics)


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
    assert result.start_diagnostics == []


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


def test_lower_cost_unconverged_start_is_not_replaced_by_worse_converged_start():
    fixture_path = Path(__file__).parent / 'data' / 'reykjavik_solver_fixture.json'
    from celestial_localiser.ephemeris import EphemerisProvider

    capture = json.loads(fixture_path.read_text(encoding='utf-8'))
    result = solve(
        capture['observations'],
        capture['timestamp'],
        EphemerisProvider(),
        (51.5, -0.1, 0.0),
        max_nfev=30,
        global_search=True,
        max_starts=25,
    )

    finite = [
        diagnostic for diagnostic in result.start_diagnostics
        if diagnostic['cost'] is not None
    ]
    successful = [diagnostic for diagnostic in finite if diagnostic['success']]
    assert finite
    assert successful
    assert min(diagnostic['cost'] for diagnostic in finite) < min(
        diagnostic['cost'] for diagnostic in successful
    )
    selected = next(
        diagnostic for diagnostic in result.start_diagnostics
        if diagnostic['selected']
    )
    assert selected['cost'] == min(diagnostic['cost'] for diagnostic in finite)
    assert selected['success'] is False
    assert result.valid is False
    assert result.failure_reason == 'optimizer_not_converged'


def _fixture_observations(fixture, confidence_key):
    return [
        {
            'object_id': observation['object_id'],
            'azimuth': observation['azimuth'],
            'elevation': observation['elevation'],
            'confidence': observation[confidence_key],
        }
        for observation in fixture['observations']
    ]


def _unweighted_rms_degrees(result, observations):
    angles = []
    for residual, observation in zip(result.residuals, observations):
        chord = abs(float(residual)) / max(observation['confidence'], 1e-3)
        angles.append(math.degrees(2.0 * math.asin(min(1.0, chord / 2.0))))
    return math.sqrt(sum(angle * angle for angle in angles) / len(angles))


def _geographic_error_meters(state, truth):
    latitude_delta = math.radians(state[0] - truth['latitude'])
    longitude_delta = math.radians(
        (state[1] - truth['longitude'] + 180.0) % 360.0 - 180.0
    )
    haversine_a = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(math.radians(truth['latitude']))
        * math.cos(math.radians(state[0]))
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * 6_371_000.0 * math.atan2(
        math.sqrt(haversine_a),
        math.sqrt(max(0.0, 1.0 - haversine_a)),
    )


def test_melbourne_agreement_only_failure_remains_a_high_residual_regression():
    fixture_path = Path(__file__).parent / 'data' / 'melbourne_agreement_only_fixture.json'
    from celestial_localiser.ephemeris import EphemerisProvider

    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    settings = fixture['solver_settings']
    ephemeris = EphemerisProvider()

    baseline_observations = _fixture_observations(fixture, 'baseline_confidence')
    baseline = solve(
        baseline_observations,
        fixture['timestamp'],
        ephemeris,
        fixture['initial_state'],
        max_nfev=settings['max_nfev'],
        max_starts=settings['max_starts'],
        global_search=settings['global_search'],
    )
    assert baseline.valid is fixture['expected']['baseline']['valid']
    assert _geographic_error_meters(baseline.x, fixture['truth']) < 2_000.0
    assert _unweighted_rms_degrees(baseline, baseline_observations) < fixture[
        'expected'
    ]['baseline']['max_unweighted_residual_degrees']

    agreement_observations = _fixture_observations(
        fixture, 'agreement_only_confidence'
    )
    agreement_only = solve(
        agreement_observations,
        fixture['timestamp'],
        ephemeris,
        fixture['initial_state'],
        max_nfev=settings['max_nfev'],
        max_starts=settings['max_starts'],
        global_search=settings['global_search'],
    )
    expected = fixture['expected']['agreement_only']
    assert agreement_only.valid is expected['valid']
    assert agreement_only.starts_tried >= expected['minimum_start_count']
    selected_indices = [
        index for index, diagnostic in enumerate(agreement_only.start_diagnostics)
        if diagnostic['selected']
    ]
    assert selected_indices == [expected['selected_start_index']]
    finite_costs = [
        diagnostic['cost']
        for diagnostic in agreement_only.start_diagnostics
        if diagnostic['cost'] is not None
    ]
    assert agreement_only.cost == pytest.approx(min(finite_costs))
    assert _unweighted_rms_degrees(agreement_only, agreement_observations) > expected[
        'minimum_unweighted_rms_residual_degrees'
    ]
    assert all(
        diagnostic['final_state'] is not None
        and _geographic_error_meters(diagnostic['final_state'], fixture['truth'])
        > expected['minimum_geographic_error_meters_per_start']
        for diagnostic in agreement_only.start_diagnostics
    )