#!/usr/bin/env python3
"""Diagnose the saved New York face observations and solver solution basins."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'celestial_localisation' / 'src' / 'celestial_localiser'))

from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_localiser.pose_solver import _residuals, solve  # noqa: E402


def _wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _direction(azimuth, elevation):
    azimuth = math.radians(azimuth)
    elevation = math.radians(elevation)
    cosine = math.cos(elevation)
    return np.asarray((
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ), dtype=np.float64)


def _angular_error(first, second):
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def _geographic_error(latitude, longitude, scene):
    latitude_delta = math.radians(latitude - scene['latitude'])
    longitude_delta = math.radians(
        (longitude - scene['longitude'] + 180.0) % 360.0 - 180.0
    )
    haversine_a = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(math.radians(scene['latitude']))
        * math.cos(math.radians(latitude))
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * 6_371_000.0 * math.atan2(
        math.sqrt(haversine_a),
        math.sqrt(max(0.0, 1.0 - haversine_a)),
    )


def _objective(residuals):
    residuals = np.asarray(residuals, dtype=np.float64)
    return {
        'raw_half_squared': float(0.5 * np.sum(np.square(residuals))),
        'soft_l1': float(np.sum(np.sqrt(1.0 + np.square(residuals)) - 1.0)),
        'rms_residual': float(np.sqrt(np.mean(np.square(residuals))))
        if residuals.size else None,
        'max_residual': float(np.max(np.abs(residuals)))
        if residuals.size else None,
    }


def _state_payload(state, scene):
    return {
        'latitude': float(state[0]),
        'longitude': float(state[1]),
        'heading': float(state[2]),
        'geographic_error_meters': _geographic_error(
            float(state[0]), float(state[1]), scene
        ),
    }


def _diagnostic_start(item, scene):
    payload = dict(item)
    if payload.get('final_state') is not None:
        payload['geographic_error_meters'] = _geographic_error(
            payload['final_state'][0],
            payload['final_state'][1],
            scene,
        )
    return payload


def _signed_observation_errors(observations, scene, provider):
    rows = []
    for observation in observations:
        predicted = provider.predict(
            observation['object_id'],
            scene['timestamp'],
            scene['latitude'],
            scene['longitude'],
            scene['altitude'],
        )
        if predicted is None:
            continue
        predicted_azimuth, predicted_elevation = predicted
        observed_azimuth = (
            float(observation['azimuth']) + float(scene['heading'])
        ) % 360.0
        observed_elevation = float(observation['elevation'])
        cross_track = _wrap_degrees(observed_azimuth - predicted_azimuth) * math.cos(
            math.radians(predicted_elevation)
        )
        observed_direction = _direction(observed_azimuth, observed_elevation)
        predicted_direction = _direction(predicted_azimuth, predicted_elevation)
        rows.append({
            'object_id': observation['object_id'],
            'observed_azimuth_deg': observed_azimuth,
            'observed_elevation_deg': observed_elevation,
            'predicted_azimuth_deg': predicted_azimuth,
            'predicted_elevation_deg': predicted_elevation,
            'signed_azimuth_error_deg': _wrap_degrees(
                observed_azimuth - predicted_azimuth
            ),
            'cross_track_error_deg': cross_track,
            'signed_elevation_error_deg': observed_elevation - predicted_elevation,
            'angular_error_deg': _angular_error(
                observed_direction,
                predicted_direction,
            ),
            'brightness': observation.get('brightness'),
        })
    return rows


def _summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {'count': 0, 'median': None, 'p90': None, 'mean': None, 'max': None}
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': len(array),
        'median': float(np.median(array)),
        'p90': float(np.percentile(array, 90.0)),
        'mean': float(np.mean(array)),
        'max': float(np.max(array)),
    }


def _correlation(rows, feature):
    features = np.asarray([row[feature] for row in rows], dtype=np.float64)
    errors = np.abs(np.asarray([
        row['angular_error_deg'] for row in rows
    ], dtype=np.float64))
    if len(features) < 2 or np.std(features) == 0.0 or np.std(errors) == 0.0:
        return None
    return float(np.corrcoef(features, errors)[0, 1])


def _load_new_york(args):
    campaign = json.loads(args.campaign.read_text(encoding='utf-8'))
    replay_case = next(
        case for case in campaign['replay_cases']
        if case['scene_id'] == 'new_york_00_h045'
    )
    scene = replay_case['scene_definition']
    transfer = json.loads(args.transfer.read_text(encoding='utf-8'))
    transfer_scene = next(
        item for item in transfer['scenes']
        if item['scene']['scene_id'] == 'new_york_00_h045'
    )
    common_ids = transfer_scene['common_all_face_sizes_ids']
    observations = [
        observation for observation in replay_case['face_observations']
        if observation['object_id'] in common_ids
    ]
    return scene, common_ids, observations


def _solve_case(observations, scene, provider, initial_state, args):
    started = time.perf_counter()
    result = solve(
        observations,
        scene['timestamp'],
        provider,
        initial_state,
        max_nfev=args.max_nfev,
        global_search=True,
        max_starts=args.max_starts,
    )
    elapsed = time.perf_counter() - started
    state = [float(value) for value in result.x]
    residuals = _residuals(state, observations, scene['timestamp'], provider)
    return result, {
        'valid': result.valid,
        'failure_reason': result.failure_reason,
        'selected_state': _state_payload(state, scene),
        'cost': result.cost,
        'selected_objective': _objective(residuals),
        'nfev': result.nfev,
        'total_nfev': result.total_nfev,
        'runtime_seconds': elapsed,
        'start_diagnostics': [
            _diagnostic_start(item, scene)
            for item in result.start_diagnostics
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--campaign', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'face_panorama_campaign_20260929_nfev60_starts5.json',
    )
    parser.add_argument(
        '--transfer', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'face_resolution_transfer_20260929.json',
    )
    parser.add_argument('--max-nfev', type=int, default=60)
    parser.add_argument('--max-starts', type=int, default=5)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'new_york_observation_diagnostic_20260929.json',
    )
    args = parser.parse_args()

    scene, common_ids, observations = _load_new_york(args)
    provider = EphemerisProvider()
    initial_state = (51.5, -0.1, 0.0)
    truth_observation_errors = _signed_observation_errors(
        observations, scene, provider
    )
    truth_residuals = _residuals(
        (scene['latitude'], scene['longitude'], scene['heading']),
        observations,
        scene['timestamp'],
        provider,
    )
    full_result, full = _solve_case(
        observations, scene, provider, initial_state, args
    )

    leave_one_out = []
    for removed in observations:
        retained = [
            observation for observation in observations
            if observation['object_id'] != removed['object_id']
        ]
        _, result = _solve_case(retained, scene, provider, initial_state, args)
        leave_one_out.append({
            'removed_object_id': removed['object_id'],
            'retained_observation_count': len(retained),
            **result,
        })

    output = {
        'scene': scene,
        'source_campaign': args.campaign.name,
        'source_transfer_report': args.transfer.name,
        'resolution_observation_availability': {
            '512': 'exact face solver observations available from campaign replay_cases',
            '1024': 'not saved; transfer report contains IDs/errors only',
            '2048': 'not saved; transfer report contains IDs/errors only',
        },
        'common_all_face_sizes_ids': common_ids,
        'observation_count': len(observations),
        'solver_configuration': {
            'max_nfev': args.max_nfev,
            'max_starts': args.max_starts,
            'global_search': True,
            'initial_state': initial_state,
            'ground_truth_used_for_selection': False,
        },
        'signed_truth_observation_errors': truth_observation_errors,
        'signed_truth_error_summary': {
            'angular_error_deg': _summary(
                row['angular_error_deg'] for row in truth_observation_errors
            ),
            'cross_track_error_deg': _summary(
                abs(row['cross_track_error_deg']) for row in truth_observation_errors
            ),
            'elevation_error_deg': _summary(
                abs(row['signed_elevation_error_deg'])
                for row in truth_observation_errors
            ),
            'correlation_abs_error_with_brightness': _correlation(
                truth_observation_errors, 'brightness'
            ) if all(row['brightness'] is not None for row in truth_observation_errors) else None,
        },
        'objective_at_ground_truth': _objective(truth_residuals),
        'full_solve': full,
        'leave_one_out': leave_one_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'common_id_count': len(common_ids),
        'objective_at_ground_truth': output['objective_at_ground_truth'],
        'selected_objective': full['selected_objective'],
        'selected_geographic_error_meters': full['selected_state']['geographic_error_meters'],
        'leave_one_out_count': len(leave_one_out),
        'higher_resolution_observations_available': False,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
