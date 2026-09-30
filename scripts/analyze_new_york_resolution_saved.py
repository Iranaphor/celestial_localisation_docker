#!/usr/bin/env python3
"""Analyze saved New York face-resolution observations without rendering."""

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

FIXED_SOLVER = {
    'max_nfev': 60,
    'max_starts': 5,
    'global_search': True,
}


def _direction(azimuth, elevation):
    azimuth = math.radians(float(azimuth))
    elevation = math.radians(float(elevation))
    cosine = math.cos(elevation)
    return np.asarray((
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ), dtype=np.float64)


def _azimuth_elevation(direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return (
        math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))),
    )


def _wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


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


def _signed_summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {
            'count': 0,
            'median': None,
            'p10': None,
            'p90': None,
            'mean': None,
            'min': None,
            'max': None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': len(array),
        'median': float(np.median(array)),
        'p10': float(np.percentile(array, 10.0)),
        'p90': float(np.percentile(array, 90.0)),
        'mean': float(np.mean(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
    }


def _error_summary(rows):
    return {
        'count': len(rows),
        'object_ids': sorted(row['object_id'] for row in rows),
        'angular_error_deg': _summary(
            row['angular_error_deg'] for row in rows
        ),
        'signed_azimuth_error_deg': _signed_summary(
            row['signed_azimuth_error_deg'] for row in rows
        ),
        'cross_track_error_deg': _signed_summary(
            row['cross_track_error_deg'] for row in rows
        ),
        'signed_elevation_error_deg': _signed_summary(
            row['signed_elevation_error_deg'] for row in rows
        ),
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


def _signed_truth_errors(observations, scene, provider):
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
        observed_direction = _direction(observed_azimuth, observed_elevation)
        predicted_direction = _direction(predicted_azimuth, predicted_elevation)
        signed_azimuth = _wrap_degrees(observed_azimuth - predicted_azimuth)
        rows.append({
            'object_id': observation['object_id'],
            'signed_azimuth_error_deg': signed_azimuth,
            'cross_track_error_deg': signed_azimuth * math.cos(
                math.radians(predicted_elevation)
            ),
            'signed_elevation_error_deg': observed_elevation - predicted_elevation,
            'angular_error_deg': _angular_error(observed_direction, predicted_direction),
        })
    return rows


def _correlation(rows, feature):
    features = np.asarray([row[feature] for row in rows], dtype=np.float64)
    errors = np.abs(np.asarray([
        row['angular_error_deg'] for row in rows
    ], dtype=np.float64))
    if len(features) < 2 or np.std(features) == 0.0 or np.std(errors) == 0.0:
        return None
    return float(np.corrcoef(features, errors)[0, 1])


def _load_report(path):
    report = json.loads(path.read_text(encoding='utf-8'))
    scenes = report.get('scenes', [])
    if len(scenes) != 1 or scenes[0]['scene']['scene_id'] != 'new_york_00_h045':
        raise ValueError('expected a single new_york_00_h045 transfer scene')
    return report, scenes[0]


def _observations_for_case(case, common_ids, scene):
    records = {
        record['object_id']: record
        for record in case['matched_star_records']
    }
    observations = []
    for object_id in common_ids:
        record = records[object_id]
        local_direction = _direction(
            record['face_azimuth_deg'] - scene['heading'],
            record['face_elevation_deg'],
        )
        azimuth, elevation = _azimuth_elevation(local_direction)
        observations.append({
            'object_id': object_id,
            'azimuth': azimuth,
            'elevation': elevation,
            'confidence': 1.0,
        })
    return observations


def _solve_case(observations, scene, provider, args):
    if len(observations) < 3:
        return {
            'valid': False,
            'failure_reason': 'insufficient_observations',
            'selected_state': None,
            'selected_objective': None,
            'ground_truth_objective': None,
            'start_diagnostics': [],
            'leave_one_out': [],
        }
    ground_truth_residuals = _residuals(
        (scene['latitude'], scene['longitude'], scene['heading']),
        observations,
        scene['timestamp'],
        provider,
    )
    started = time.perf_counter()
    result = solve(
        observations,
        scene['timestamp'],
        provider,
        (51.5, -0.1, 0.0),
        max_nfev=args.max_nfev,
        global_search=True,
        max_starts=args.max_starts,
    )
    elapsed = time.perf_counter() - started
    selected_state = _state_payload(result.x, scene)
    selected_residuals = _residuals(
        result.x,
        observations,
        scene['timestamp'],
        provider,
    )
    leave_one_out = []
    for removed in observations:
        retained = [
            observation for observation in observations
            if observation['object_id'] != removed['object_id']
        ]
        loo = solve(
            retained,
            scene['timestamp'],
            provider,
            (51.5, -0.1, 0.0),
            max_nfev=args.max_nfev,
            global_search=True,
            max_starts=args.max_starts,
        )
        leave_one_out.append({
            'removed_object_id': removed['object_id'],
            'valid': loo.valid,
            'failure_reason': loo.failure_reason,
            'selected_state': _state_payload(loo.x, scene),
            'objective': _objective(loo.residuals),
        })
    return {
        'valid': result.valid,
        'failure_reason': result.failure_reason,
        'selected_state': selected_state,
        'selected_objective': _objective(selected_residuals),
        'ground_truth_objective': _objective(ground_truth_residuals),
        'start_diagnostics': [
            _diagnostic_start(item, scene)
            for item in result.start_diagnostics
        ],
        'leave_one_out': leave_one_out,
        'runtime_seconds': elapsed,
        'observation_count': len(observations),
    }


def _overlap_candidates(case, canonical_records, radius):
    canonical_by_id = {
        record['object_id']: np.asarray(record['direction'], dtype=np.float64)
        for record in canonical_records
    }
    candidates = []
    for record in case['all_face_detections']:
        direction = np.asarray(record['direction'], dtype=np.float64)
        nearby = []
        for object_id, canonical_direction in canonical_by_id.items():
            separation = _angular_error(direction, canonical_direction)
            if separation <= radius:
                nearby.append((separation, object_id))
        if not nearby:
            continue
        separation, object_id = min(nearby)
        candidates.append({
            'object_id': object_id,
            'face_name': record['face_name'],
            'pixel_x': record['pixel_x'],
            'pixel_y': record['pixel_y'],
            'brightness': record['brightness'],
            'distance_to_canonical_degrees': separation,
        })
    by_id = {}
    for candidate in candidates:
        by_id.setdefault(candidate['object_id'], []).append(candidate)
    for values in by_id.values():
        values.sort(key=lambda item: item['distance_to_canonical_degrees'])
    return by_id


def _candidate_error_rows(case, canonical_records, scene, provider, radius):
    canonical_by_id = {
        record['object_id']: np.asarray(record['direction'], dtype=np.float64)
        for record in canonical_records
    }
    predicted_by_id = {}
    matched_by_id = {
        record['object_id']: record
        for record in case['matched_star_records']
    }
    for object_id in canonical_by_id:
        predicted_by_id[object_id] = provider.predict(
            object_id,
            scene['timestamp'],
            scene['latitude'],
            scene['longitude'],
            scene['altitude'],
        )
    candidates_by_id = {}
    for record in case['all_face_detections']:
        direction = np.asarray(record['direction'], dtype=np.float64)
        nearby = [
            (_angular_error(direction, canonical_direction), object_id)
            for object_id, canonical_direction in canonical_by_id.items()
        ]
        nearby = [item for item in nearby if item[0] <= radius]
        if not nearby:
            continue
        distance, object_id = min(nearby)
        predicted = predicted_by_id[object_id]
        if predicted is None:
            continue
        predicted_azimuth, predicted_elevation = predicted
        observed_azimuth = float(record['azimuth_deg'])
        observed_elevation = float(record['elevation_deg'])
        signed_azimuth = _wrap_degrees(observed_azimuth - predicted_azimuth)
        matched = matched_by_id.get(object_id)
        candidates_by_id.setdefault(object_id, []).append({
            'face_name': record['face_name'],
            'pixel_x': record['pixel_x'],
            'pixel_y': record['pixel_y'],
            'brightness': record['brightness'],
            'direction': record['direction'],
            'selected_match': bool(
                matched is not None
                and record['face_name'] == matched['face_name']
                and abs(record['pixel_x'] - matched['pixel_x']) < 1e-6
                and abs(record['pixel_y'] - matched['pixel_y']) < 1e-6
            ),
            'distance_to_canonical_degrees': distance,
            'signed_azimuth_error_deg': signed_azimuth,
            'cross_track_error_deg': signed_azimuth * math.cos(
                math.radians(predicted_elevation)
            ),
            'signed_elevation_error_deg': (
                observed_elevation - predicted_elevation
            ),
            'angular_error_deg': _angular_error(
                direction,
                _direction(predicted_azimuth, predicted_elevation),
            ),
        })
    for values in candidates_by_id.values():
        values.sort(key=lambda item: item['distance_to_canonical_degrees'])
    return candidates_by_id


def _matching_diagnostics(
    case,
    canonical_records,
    scene,
    provider,
    max_distance,
    ambiguity_margin,
    duplicate_radius,
    candidate_errors,
):
    object_ids = sorted(record['object_id'] for record in canonical_records)
    canonical_by_id = {
        record['object_id']: np.asarray(record['direction'], dtype=np.float64)
        for record in canonical_records
    }
    raw_detections = case['all_face_detections']
    retained_indices = []
    retained_directions = []
    duplicate_of = {}
    ordered_detections = sorted(
        enumerate(raw_detections),
        key=lambda item: (-item[1].get('brightness', 0.0), item[0]),
    )
    for source_index, detection in ordered_detections:
        direction = np.asarray(detection['direction'], dtype=np.float64)
        duplicate_index = next(
            (
                retained_index
                for retained_index, retained_direction in zip(
                    retained_indices, retained_directions
                )
                if _angular_error(direction, retained_direction) <= duplicate_radius
            ),
            None,
        )
        if duplicate_index is None:
            retained_indices.append(source_index)
            retained_directions.append(direction)
        else:
            duplicate_of[source_index] = duplicate_index
    retained_index_set = set(retained_indices)
    matched_by_id = {
        record['object_id']: record
        for record in case['matched_star_records']
    }
    candidate_error_by_key = {
        (
            row['face_name'],
            row['pixel_x'],
            row['pixel_y'],
        ): row
        for rows in candidate_errors.values()
        for row in rows
    }

    def reference(index):
        detection = raw_detections[index]
        return {
            'face_name': detection['face_name'],
            'pixel_x': detection['pixel_x'],
            'pixel_y': detection['pixel_y'],
            'brightness': detection['brightness'],
        }

    diagnostics = {}
    for object_id in object_ids:
        canonical_direction = canonical_by_id[object_id]
        raw_candidates = []
        for index, detection in enumerate(raw_detections):
            distance = _angular_error(
                canonical_direction,
                np.asarray(detection['direction'], dtype=np.float64),
            )
            if distance <= max_distance:
                raw_candidates.append((distance, index))
        raw_candidates.sort()
        retained_candidates = [
            item for item in raw_candidates if item[1] in retained_index_set
        ]
        matched = matched_by_id.get(object_id)
        selected_index = next(
            (
                index for _, index in raw_candidates
                if matched is not None
                and raw_detections[index]['face_name'] == matched['face_name']
                and abs(raw_detections[index]['pixel_x'] - matched['pixel_x']) < 1e-6
                and abs(raw_detections[index]['pixel_y'] - matched['pixel_y']) < 1e-6
            ),
            None,
        )
        panorama_margin = (
            retained_candidates[1][0] - retained_candidates[0][0]
            if len(retained_candidates) > 1 else None
        )
        panorama_ambiguous = (
            panorama_margin is not None
            and panorama_margin < ambiguity_margin
        )
        predicted = provider.predict(
            object_id,
            scene['timestamp'],
            scene['latitude'],
            scene['longitude'],
            scene['altitude'],
        )
        panorama_truth = None
        if predicted is not None:
            predicted_azimuth, predicted_elevation = predicted
            canonical_azimuth, canonical_elevation = _azimuth_elevation(
                canonical_direction
            )
            signed_azimuth = _wrap_degrees(
                canonical_azimuth - predicted_azimuth
            )
            panorama_truth = {
                'signed_azimuth_error_deg': signed_azimuth,
                'cross_track_error_deg': signed_azimuth * math.cos(
                    math.radians(predicted_elevation)
                ),
                'signed_elevation_error_deg': (
                    canonical_elevation - predicted_elevation
                ),
                'angular_error_deg': _angular_error(
                    canonical_direction,
                    _direction(predicted_azimuth, predicted_elevation),
                ),
            }

        candidate_payloads = []
        face_ambiguous_object_ids = set()
        for rank, (distance, index) in enumerate(retained_candidates):
            detection = raw_detections[index]
            face_distances = sorted(
                (
                    _angular_error(
                        canonical_by_id[candidate_id],
                        np.asarray(detection['direction'], dtype=np.float64),
                    ),
                    candidate_id,
                )
                for candidate_id in object_ids
            )
            face_distances = [
                item for item in face_distances if item[0] <= max_distance
            ]
            face_margin = (
                face_distances[1][0] - face_distances[0][0]
                if len(face_distances) > 1 else None
            )
            face_ambiguous = (
                face_margin is not None and face_margin < ambiguity_margin
            )
            if face_ambiguous:
                face_ambiguous_object_ids.update(
                    candidate_id for _, candidate_id in face_distances
                )
            key = (
                detection['face_name'],
                detection['pixel_x'],
                detection['pixel_y'],
            )
            error = candidate_error_by_key.get(key, {})
            candidate_payloads.append({
                **reference(index),
                'deduplicated': True,
                'duplicate_of': None,
                'panorama_distance_degrees': distance,
                'panorama_rank_after_deduplication': rank,
                'selected_match': index == selected_index,
                'face_nearest_object_id': (
                    face_distances[0][1] if face_distances else None
                ),
                'face_nearest_distance_degrees': (
                    face_distances[0][0] if face_distances else None
                ),
                'face_second_distance_degrees': (
                    face_distances[1][0]
                    if len(face_distances) > 1 else None
                ),
                'face_ambiguity_margin_degrees': face_margin,
                'face_ambiguous': face_ambiguous,
                'mutual_nearest': bool(
                    face_distances and face_distances[0][1] == object_id
                ),
                'signed_azimuth_error_deg': error.get(
                    'signed_azimuth_error_deg'
                ),
                'cross_track_error_deg': error.get('cross_track_error_deg'),
                'signed_elevation_error_deg': error.get(
                    'signed_elevation_error_deg'
                ),
                'angular_error_deg': error.get('angular_error_deg'),
            })
        raw_payloads = []
        for distance, index in raw_candidates:
            if index in retained_index_set:
                continue
            duplicate_index = duplicate_of.get(index)
            raw_payloads.append({
                **reference(index),
                'deduplicated': False,
                'duplicate_of': (
                    reference(duplicate_index)
                    if duplicate_index is not None else None
                ),
                'panorama_distance_degrees': distance,
                'selected_match': index == selected_index,
            })
        selected_rank = next(
            (
                payload['panorama_rank_after_deduplication']
                for payload in candidate_payloads
                if payload['selected_match']
            ),
            None,
        )
        diagnostics[object_id] = {
            'raw_candidate_count': len(raw_candidates),
            'retained_candidate_count': len(retained_candidates),
            'duplicate_discarded_candidate_count': len(raw_payloads),
            'panorama_ambiguity_margin_degrees': panorama_margin,
            'panorama_ambiguous': panorama_ambiguous,
            'face_ambiguous_object_ids': sorted(face_ambiguous_object_ids),
            'selected_rank_after_deduplication': selected_rank,
            'selection_rule': (
                'nearest_panorama_candidate_after_duplicate_deduplication'
                if selected_rank == 0 and not panorama_ambiguous
                else 'not_selected_by_saved_match'
            ),
            'panorama_truth_bias': panorama_truth,
            'retained_candidates': candidate_payloads,
            'discarded_duplicate_candidates': raw_payloads,
        }
    return diagnostics


def _measurement_subsets(case, signed_errors, candidate_rows, blend_radius):
    matched_by_id = {
        record['object_id']: record
        for record in case['matched_star_records']
    }
    rows_by_class = {
        'clean_isolated': [],
        'overlapping_faces': [],
        'suspected_blended': [],
        'unresolved': [],
    }
    rows_by_flag = {
        'clean_isolated': [],
        'overlapping_faces': [],
        'suspected_blended': [],
    }
    details = {}
    for error in signed_errors:
        object_id = error['object_id']
        matched = matched_by_id[object_id]
        candidates = candidate_rows.get(object_id, [])
        candidate_faces = sorted({item['face_name'] for item in candidates})
        same_face_neighbors = []
        for candidate in candidates:
            if candidate['face_name'] != matched['face_name']:
                continue
            if candidate['selected_match']:
                continue
            if candidate['distance_to_canonical_degrees'] <= blend_radius:
                same_face_neighbors.append(
                    candidate['distance_to_canonical_degrees']
                )
        if same_face_neighbors:
            classification = 'suspected_blended'
        elif len(candidate_faces) >= 2:
            classification = 'overlapping_faces'
        elif len(candidates) == 1:
            classification = 'clean_isolated'
        else:
            classification = 'unresolved'
        row = {
            **error,
            'classification': classification,
            'candidate_count': len(candidates),
            'candidate_faces': candidate_faces,
            'same_face_neighbor_count': len(same_face_neighbors),
            'nearest_same_face_neighbor_degrees': (
                min(same_face_neighbors) if same_face_neighbors else None
            ),
        }
        details[object_id] = row
        rows_by_class[classification].append(row)
        if classification == 'clean_isolated':
            rows_by_flag['clean_isolated'].append(row)
        if len(candidate_faces) >= 2:
            rows_by_flag['overlapping_faces'].append(row)
        if same_face_neighbors:
            rows_by_flag['suspected_blended'].append(row)
    return {
        'by_class': {
            classification: _error_summary(rows)
            for classification, rows in rows_by_class.items()
        },
        'by_flag': {
            flag: _error_summary(rows)
            for flag, rows in rows_by_flag.items()
        },
        'by_object_id': details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--transfer', type=Path, required=True,
        help='saved transfer report with per-resolution records',
    )
    parser.add_argument(
        '--output', type=Path, required=True,
    )
    parser.add_argument('--max-nfev', type=int, default=60)
    parser.add_argument('--max-starts', type=int, default=5)
    parser.add_argument('--overlap-radius-degrees', type=float, default=2.0)
    parser.add_argument('--blend-neighbor-radius-degrees', type=float, default=0.50)
    parser.add_argument('--ambiguity-margin-degrees', type=float, default=0.25)
    parser.add_argument('--duplicate-radius-degrees', type=float, default=0.15)
    args = parser.parse_args()

    report, scene_report = _load_report(args.transfer)
    scene = scene_report['scene']
    common_ids = scene_report['common_all_face_sizes_ids']
    provider = EphemerisProvider()
    results = {}
    overlap = {}
    overlap_errors = {}
    for case in scene_report['cases']:
        observations = _observations_for_case(case, common_ids, scene)
        size = str(case['face_size'])
        signed_errors = _signed_truth_errors(observations, scene, provider)
        candidate_errors = _candidate_error_rows(
            case,
            report['scenes'][0]['canonical_star_records'],
            scene,
            provider,
            args.overlap_radius_degrees,
        )
        matching_diagnostics = _matching_diagnostics(
            case,
            report['scenes'][0]['canonical_star_records'],
            scene,
            provider,
            args.overlap_radius_degrees,
            args.ambiguity_margin_degrees,
            args.duplicate_radius_degrees,
            candidate_errors,
        )
        results[size] = {
            'matched_id_count': case['matched_id_count'],
            'dropped_canonical_ids': case['dropped_canonical_ids'],
            'ambiguous_canonical_ids': case['ambiguous_canonical_ids'],
            'signed_truth_errors': signed_errors,
            'signed_truth_summary': {
                'angular_error_deg': _summary(
                    row['angular_error_deg'] for row in signed_errors
                ),
                'cross_track_error_deg': _summary(
                    abs(row['cross_track_error_deg']) for row in signed_errors
                ),
                'elevation_error_deg': _summary(
                    abs(row['signed_elevation_error_deg']) for row in signed_errors
                ),
            },
            'measurement_subsets': _measurement_subsets(
                case,
                signed_errors,
                candidate_errors,
                args.blend_neighbor_radius_degrees,
            ),
            'matching_diagnostics': matching_diagnostics,
            'solver': _solve_case(observations, scene, provider, args),
        }
        overlap[size] = _overlap_candidates(
            case,
            report['scenes'][0]['canonical_star_records'],
            args.overlap_radius_degrees,
        )
        overlap_errors[size] = candidate_errors

    shared_bias = {}
    for object_id in common_ids:
        per_size = {}
        for size, result in results.items():
            row = next(
                row for row in result['signed_truth_errors']
                if row['object_id'] == object_id
            )
            per_size[size] = row
        shared_bias[object_id] = per_size

    output = {
        'scene': scene,
        'source_transfer_report': args.transfer.name,
        'common_all_face_sizes_ids': common_ids,
        'solver_configuration': {
            'max_nfev': args.max_nfev,
            'max_starts': args.max_starts,
            'global_search': True,
            'ground_truth_used_for_selection': False,
        },
        'matching_configuration': {
            'max_distance_degrees': args.overlap_radius_degrees,
            'ambiguity_margin_degrees': args.ambiguity_margin_degrees,
            'duplicate_radius_degrees': args.duplicate_radius_degrees,
            'ground_truth_used_for_selection': False,
        },
        'geographic_error_scope': {
            'observation_count': len(common_ids),
            'object_ids': common_ids,
            'uses_all_common_ids': True,
            'subset_classifications_change_solver_input': False,
        },
        'resolution_observations': results,
        'overlap_face_candidates': overlap,
        'overlap_face_error_diagnostics': overlap_errors,
        'diagnostic_subset_definition': {
            'ground_truth_used_for_selection': False,
            'candidate_radius_degrees': args.overlap_radius_degrees,
            'blend_neighbor_radius_degrees': args.blend_neighbor_radius_degrees,
            'classes': {
                'clean_isolated': (
                    'one canonical candidate within the candidate radius and '
                    'no same-face neighbor within the blend radius'
                ),
                'overlapping_faces': (
                    'candidates on at least two faces, without a closer '
                        'same-face candidate near the canonical direction'
                ),
                'suspected_blended': (
                        'a second same-face candidate within the blend radius of '
                        'the canonical direction; '
                    'diagnostic label, not a production rejection rule'
                ),
                'unresolved': 'does not match the other diagnostic classes',
            },
        },
        'shared_bias_by_object_id': shared_bias,
        'higher_resolution_observations_saved': True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'common_id_count': len(common_ids),
        'face_sizes': sorted(results),
        'hip_26241_candidates': {
            size: overlap[size].get('HIP_26241', [])
            for size in overlap
        },
        'selected_geographic_error_meters': {
            size: results[size]['solver']['selected_state']['geographic_error_meters']
            for size in results
            if results[size]['solver']['selected_state'] is not None
        },
        'measurement_subset_counts': {
            size: {
                classification: summary['count']
                for classification, summary in results[size][
                    'measurement_subsets'
                ]['by_class'].items()
            }
            for size in results
        },
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
