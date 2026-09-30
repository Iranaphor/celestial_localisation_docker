#!/usr/bin/env python3
"""Evaluate ground-truth-free residual gates on saved face captures."""

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
sys.path.insert(0, str(ROOT / 'scripts'))

from evaluate_face_quality_rules import (  # noqa: E402
    QUALITY_ABLATIONS,
    _capture_measurements,
    _quality_weight,
)
from celestial_localiser.coordinate_transforms import az_el_to_unit_vector  # noqa: E402
from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_localiser.pose_solver import _residuals, solve  # noqa: E402

GATE_PROFILES = {
    'strict_residual': {
        'rms_angular_degrees': 0.10,
        'p90_angular_degrees': 0.10,
        'max_angular_degrees': 0.35,
    },
    'balanced_residual': {
        'rms_angular_degrees': 0.15,
        'p90_angular_degrees': 0.15,
        'max_angular_degrees': 0.75,
    },
    'lenient_residual': {
        'rms_angular_degrees': 0.25,
        'p90_angular_degrees': 0.25,
        'max_angular_degrees': 1.00,
    },
    'residual_plus_narrow_geometry': {
        'rms_angular_degrees': 0.15,
        'p90_angular_degrees': 0.10,
        'max_angular_degrees': 0.75,
        'narrow_span_degrees': 60.0,
        'narrow_span_max_condition': 80.0,
    },
}

FROZEN_DIAGNOSTIC_PROFILE = 'balanced_residual'


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


def _angular_residuals(observations, state, timestamp, ephemeris):
    weighted_chords = np.asarray(
        _residuals(state, observations, timestamp, ephemeris),
        dtype=np.float64,
    )
    confidence = np.asarray([
        max(float(observation.get('confidence', 1.0)), 1e-3)
        for observation in observations
    ], dtype=np.float64)
    confidence = confidence[:len(weighted_chords)]
    unweighted_chords = np.abs(weighted_chords) / confidence
    return np.degrees(2.0 * np.arcsin(np.clip(unweighted_chords / 2.0, 0.0, 1.0)))


def _geometry(observations):
    directions = np.asarray([
        az_el_to_unit_vector(observation['azimuth'], observation['elevation'])
        for observation in observations
    ], dtype=np.float64)
    if len(directions) < 2:
        return {
            'observation_count': len(directions),
            'max_pairwise_separation_degrees': 0.0,
            'covariance_eigenvalues': [],
            'covariance_condition': None,
        }
    dot_products = np.clip(directions @ directions.T, -1.0, 1.0)
    pairwise = np.degrees(np.arccos(dot_products))
    upper = pairwise[np.triu_indices(len(directions), k=1)]
    centered = directions - np.mean(directions, axis=0)
    covariance = centered.T @ centered / max(1, len(directions) - 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    positive = eigenvalues[eigenvalues > 1e-12]
    return {
        'observation_count': len(directions),
        'max_pairwise_separation_degrees': float(np.max(upper)),
        'covariance_eigenvalues': [float(value) for value in eigenvalues],
        'covariance_condition': (
            float(positive[0] / positive[-1]) if len(positive) >= 2 else None
        ),
    }


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


def _policy_observations(measurements, confidence_policy):
    observations = []
    for observation in measurements['observations']:
        item = dict(observation)
        if confidence_policy == 'agreement_only':
            item['confidence'] = _quality_weight(
                measurements['metrics'][item['object_id']],
                QUALITY_ABLATIONS['agreement_only'],
            )
        else:
            item['confidence'] = 1.0
        observations.append(item)
    return observations


def _solve_capture(capture_path, provider, confidence_policy):
    measurements = _capture_measurements(capture_path)
    observations = _policy_observations(measurements, confidence_policy)
    scene = measurements['scene']
    started = time.perf_counter()
    result = solve(
        observations,
        scene['timestamp'],
        provider,
        (51.5, -0.1, 0.0),
        max_nfev=60,
        max_starts=5,
        global_search=True,
    )
    elapsed = time.perf_counter() - started
    state = [float(value) for value in result.x]
    residuals = _angular_residuals(
        observations,
        result.x,
        scene['timestamp'],
        provider,
    )
    if len(residuals):
        residual_metrics = {
            'rms_angular_degrees': float(np.sqrt(np.mean(np.square(residuals)))),
            'median_angular_degrees': float(np.median(residuals)),
            'p90_angular_degrees': float(np.percentile(residuals, 90.0)),
            'max_angular_degrees': float(np.max(residuals)),
            'mad_angular_degrees': float(
                1.4826 * np.median(np.abs(residuals - np.median(residuals)))
            ),
            'residuals_angular_degrees': [float(value) for value in residuals],
        }
    else:
        residual_metrics = {
            'rms_angular_degrees': None,
            'median_angular_degrees': None,
            'p90_angular_degrees': None,
            'max_angular_degrees': None,
            'mad_angular_degrees': None,
            'residuals_angular_degrees': [],
        }
    start_index = next(
        (index for index, item in enumerate(result.start_diagnostics)
         if item.get('selected')),
        None,
    )
    start_diagnostics = []
    for item in result.start_diagnostics:
        diagnostic = dict(item)
        if diagnostic.get('final_state') is not None:
            diagnostic['geographic_error_meters'] = _geographic_error(
                diagnostic['final_state'][0],
                diagnostic['final_state'][1],
                scene,
            )
        start_diagnostics.append(diagnostic)
    return {
        'capture_id': measurements['capture_id'],
        'scene': scene,
        'valid': result.valid,
        'failure_reason': result.failure_reason,
        'selected_state': {
            'latitude': state[0],
            'longitude': state[1],
            'heading': state[2],
            'geographic_error_meters': (
                _geographic_error(state[0], state[1], scene)
                if result.valid else None
            ),
        },
        'selected_cost': result.cost,
        'selected_start_index': start_index,
        'start_count': len(result.start_diagnostics),
        'converged_start_count': sum(
            item.get('success', False) for item in result.start_diagnostics
        ),
        'start_diagnostics': start_diagnostics,
        'residual_metrics': residual_metrics,
        'geometry': _geometry(observations),
        'runtime_seconds': elapsed,
        'observation_count': len(observations),
    }


def _gate_decision(result, thresholds):
    if not result['valid']:
        return False, 'solver_not_valid'
    metrics = result['residual_metrics']
    residual_keys = (
        'rms_angular_degrees',
        'p90_angular_degrees',
        'max_angular_degrees',
    )
    failures = [
        key for key, limit in thresholds.items()
        if key in residual_keys
        if metrics[key] is None or metrics[key] > limit
    ]
    span = result['geometry']['max_pairwise_separation_degrees']
    condition = result['geometry']['covariance_condition']
    if (
        'narrow_span_degrees' in thresholds
        and span < thresholds['narrow_span_degrees']
        and condition is not None
        and condition > thresholds['narrow_span_max_condition']
    ):
        failures.append('narrow_sky_geometry')
    return not failures, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--captures', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'quality_heldout_captures_20260929',
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'fit_quality_gate_20260929.json',
    )
    parser.add_argument(
        '--confidence-policy',
        choices=('baseline', 'agreement_only'),
        default='baseline',
        help='measurement confidence policy to replay; defaults to unweighted baseline',
    )
    parser.add_argument('--reasonable-error-meters', type=float, default=5000.0)
    parser.add_argument('--gross-error-meters', type=float, default=10000.0)
    args = parser.parse_args()

    provider = EphemerisProvider()
    results = []
    for capture_path in sorted(args.captures.glob('*/capture.json')):
        results.append(_solve_capture(capture_path, provider, args.confidence_policy))

    gates = {}
    for gate_name, thresholds in GATE_PROFILES.items():
        rows = []
        for result in results:
            accepted, reason = _gate_decision(result, thresholds)
            geographic_error = result['selected_state']['geographic_error_meters']
            rows.append({
                'capture_id': result['capture_id'],
                'valid': result['valid'],
                'accepted': accepted,
                'reason': reason,
                'geographic_error_meters': geographic_error,
                'reasonable': (
                    geographic_error is not None
                    and geographic_error <= args.reasonable_error_meters
                ),
                'gross': (
                    geographic_error is not None
                    and geographic_error > args.gross_error_meters
                ),
            })
        gates[gate_name] = {
            'thresholds': thresholds,
            'rows': rows,
            'attempt_count': len(rows),
            'accepted_count': sum(row['accepted'] for row in rows),
            'rejected_count': sum(not row['accepted'] for row in rows),
            'valid_count': sum(row['valid'] for row in rows),
            'invalid_count': sum(not row['valid'] for row in rows),
            'valid_accepted_count': sum(row['valid'] and row['accepted'] for row in rows),
            'valid_rejected_count': sum(row['valid'] and not row['accepted'] for row in rows),
            'invalid_rejected_count': sum((not row['valid']) and not row['accepted'] for row in rows),
            'gross_accepted_count': sum(row['accepted'] and row['gross'] for row in rows),
            'gross_rejected_count': sum((not row['accepted']) and row['gross'] for row in rows),
            'reasonable_rejected_count': sum(
                (not row['accepted']) and row['reasonable'] for row in rows
            ),
            'reasonable_accepted_count': sum(row['accepted'] and row['reasonable'] for row in rows),
        }

    output = {
        'gate_role': 'diagnostic_only',
        'frozen_diagnostic_profile': FROZEN_DIAGNOSTIC_PROFILE,
        'production_gate_enabled': False,
        'independent_validation_required': True,
        'input_dataset': str(args.captures),
        'input_policy': args.confidence_policy,
        'convergence_is_precondition': True,
        'ground_truth_used_for_selection': False,
        'diagnostic_error_labels': {
            'reasonable_error_meters': args.reasonable_error_meters,
            'gross_error_meters': args.gross_error_meters,
        },
        'gate_profiles': GATE_PROFILES,
        'results': results,
        'gates': gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'capture_count': len(results),
        'gates': {
            name: {
                key: value for key, value in gate.items()
                if key.endswith('count') or key == 'attempt_count'
            }
            for name, gate in gates.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
