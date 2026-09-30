#!/usr/bin/env python3
"""Diagnose independent Tetra3 identification on saved perspective faces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tetra3

ROOT = Path(__file__).resolve().parents[1]
for source_directory in (
    ROOT / 'celestial_localisation' / 'src' / 'celestial_simulation',
    ROOT / 'celestial_localisation' / 'src' / 'celestial_detector',
    ROOT / 'celestial_localisation' / 'src' / 'celestial_localiser',
):
    sys.path.insert(0, str(source_directory))

from celestial_detector.face_direction import face_pixel_to_direction  # noqa: E402
from celestial_detector.star_detector import detect_stars  # noqa: E402
from celestial_detector.star_identifier import StarIdentifier  # noqa: E402
from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_localiser.pose_solver import solve  # noqa: E402
from celestial_simulation.projection import cube_faces  # noqa: E402


def _direction_to_azimuth_elevation(direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return (
        math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))),
    )


def _angular_error(first, second):
    return math.degrees(math.acos(float(np.clip(
        np.dot(first, second), -1.0, 1.0
    ))))


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


def _rotate_heading(direction, heading_degrees):
    angle = math.radians(float(heading_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    direction = np.asarray(direction, dtype=np.float64)
    return np.asarray((
        cosine * direction[0] - sine * direction[1],
        sine * direction[0] + cosine * direction[1],
        direction[2],
    ), dtype=np.float64)


def _truth_error(object_id, direction, scene, provider):
    predicted = provider.predict(
        object_id,
        scene['timestamp'],
        scene['latitude'],
        scene['longitude'],
        scene['altitude'],
    )
    if predicted is None:
        return None
    predicted_azimuth, predicted_elevation = predicted
    observed_azimuth, observed_elevation = _direction_to_azimuth_elevation(direction)
    signed_azimuth = (
        observed_azimuth - predicted_azimuth + 180.0
    ) % 360.0 - 180.0
    return {
        'signed_azimuth_error_deg': signed_azimuth,
        'signed_elevation_error_deg': observed_elevation - predicted_elevation,
        'angular_error_deg': _angular_error(
            direction,
            _direction_from_angles(predicted_azimuth, predicted_elevation),
        ),
    }


def _direction_from_angles(azimuth, elevation):
    azimuth = math.radians(float(azimuth))
    elevation = math.radians(float(elevation))
    cosine = math.cos(elevation)
    return np.asarray((
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ), dtype=np.float64)


def _quality_tuple(assignment):
    return (
        int(assignment['match_count']),
        -float(assignment['match_probability']),
        -float(assignment['match_rmse_arcseconds']),
    )


def _detect_face_detections(capture_directory, face_size, scene, threshold, max_candidates):
    detections_by_face = {}
    face_views = {face.name: face for face in cube_faces()}
    for face_name, face in face_views.items():
        image_path = (
            capture_directory
            / f'face_{face_size}'
            / f'{face_name}.png'
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f'failed to read {image_path}')
        detections = detect_stars(
            image,
            threshold_sigma=threshold,
            minimum_elevation_degrees=-90.0,
            max_candidates=max_candidates,
        )
        rows = []
        for source_index, detection in enumerate(detections):
            local_direction = face_pixel_to_direction(
                face,
                detection['pixel_x'],
                detection['pixel_y'],
                image.shape[1],
                image.shape[0],
                95.0,
            )
            rows.append({
                'source_index': source_index,
                'face_name': face_name,
                'pixel_x': float(detection['pixel_x']),
                'pixel_y': float(detection['pixel_y']),
                'brightness': float(detection['brightness']),
                'local_direction': local_direction,
                'global_direction': _rotate_heading(
                    local_direction, scene['heading']
                ),
            })
        detections_by_face[face_name] = rows
    return detections_by_face


def _identify_face_crops(detections, crop_fov, solver, crop_size):
    if not detections:
        return {'hypothesis_count': 0, 'assignments': []}
    directions = np.asarray([
        detection['local_direction'] for detection in detections
    ], dtype=np.float64)
    identifier = StarIdentifier(
        solver=solver,
        fov_degrees=crop_fov,
        fov_max_error_degrees=5.0,
        tile_size=crop_size,
        match_radius=0.02,
        match_threshold=0.001,
        min_matches=4,
        pattern_checking_stars=8,
    )
    best_by_source = {}
    hypothesis_count = 0
    for center_index in range(len(detections)):
        tile_indices, tile_centroids = identifier._make_tile(
            directions,
            detections,
            center_index,
        )
        if len(tile_indices) < 4:
            continue
        brightness_order = sorted(
            range(len(tile_indices)),
            key=lambda index: detections[tile_indices[index]].get(
                'brightness', 0.0
            ),
            reverse=True,
        )
        ordered_indices = [tile_indices[index] for index in brightness_order]
        ordered_centroids = tile_centroids[brightness_order]
        try:
            result = identifier._solve(ordered_centroids)
        except (RuntimeError, ValueError, TypeError):
            result = None
        if result is None:
            continue
        match_count = int(result.get('Matches') or 0)
        if match_count < 4:
            continue
        probability = result.get('Prob')
        probability = float(probability) if probability is not None else 1.0
        if probability > 0.001:
            continue
        rmse = result.get('RMSE')
        rmse = float(rmse) if rmse is not None else float('inf')
        matched_centroids = result.get('matched_centroids')
        matched_catalogue_ids = result.get('matched_catID')
        if matched_centroids is None or matched_catalogue_ids is None:
            continue
        hypothesis_count += 1
        used_positions = set()
        for match_index, matched_centroid in enumerate(matched_centroids):
            if match_index >= len(matched_catalogue_ids):
                break
            distances = np.linalg.norm(
                ordered_centroids - np.asarray(matched_centroid, dtype=np.float64),
                axis=1,
            )
            for used_position in used_positions:
                distances[used_position] = float('inf')
            source_position = int(np.argmin(distances))
            if not np.isfinite(distances[source_position]):
                continue
            if distances[source_position] > max(2.0, crop_size * 0.02):
                continue
            used_positions.add(source_position)
            source_index = ordered_indices[source_position]
            assignment = {
                **detections[source_index],
                'object_id': f'HIP_{int(matched_catalogue_ids[match_index])}',
                'match_count': match_count,
                'match_probability': probability,
                'match_rmse_arcseconds': rmse,
                'crop_center_source_index': center_index,
                'crop_detection_count': len(tile_indices),
            }
            previous = best_by_source.get(source_index)
            if previous is None or _quality_tuple(assignment) > _quality_tuple(previous):
                best_by_source[source_index] = assignment
    return {
        'hypothesis_count': hypothesis_count,
        'assignments': list(best_by_source.values()),
    }


def _load_case(report, scene_report, face_size, capture_directory):
    case = next(
        case for case in scene_report['cases']
        if case['face_size'] == face_size
    )
    return case, Path(capture_directory or scene_report['capture_directory'])


def _selected_record(case, object_id):
    return next(
        record for record in case['matched_star_records']
        if record['object_id'] == object_id
    )


def _source_matches_record(assignment, record):
    return (
        assignment['face_name'] == record['face_name']
        and abs(assignment['pixel_x'] - record['pixel_x']) < 1.0
        and abs(assignment['pixel_y'] - record['pixel_y']) < 1.0
    )


def _annotate_assignments(assignments, case, scene, common_ids, provider):
    selected_records = {
        record['object_id']: record
        for record in case['matched_star_records']
    }
    canonical_by_id = {
        record['object_id']: np.asarray(record['canonical_direction'], dtype=np.float64)
        for record in case['matched_star_records']
    }
    annotated = []
    for assignment in assignments:
        item = dict(assignment)
        item['local_direction'] = item['local_direction'].tolist()
        item['global_direction'] = item['global_direction'].tolist()
        object_id = item['object_id']
        selected = selected_records.get(object_id)
        item['panorama_selected_source'] = (
            _source_matches_record(assignment, selected)
            if selected is not None else False
        )
        item['panorama_distance_degrees'] = (
            _angular_error(
                np.asarray(assignment['global_direction'], dtype=np.float64),
                canonical_by_id[object_id],
            )
            if object_id in canonical_by_id else None
        )
        item['truth_error'] = _truth_error(
            object_id,
            np.asarray(assignment['global_direction'], dtype=np.float64),
            scene,
            provider,
        )
        item['common_id'] = object_id in common_ids
        annotated.append(item)
    return annotated


def _choose_common_assignments(assignments, common_ids):
    candidates = [
        assignment for assignment in assignments
        if assignment['object_id'] in common_ids
    ]
    candidates.sort(key=lambda assignment: (
        -int(assignment['match_count']),
        float(assignment['match_probability']),
        float(assignment['match_rmse_arcseconds']),
        assignment['object_id'],
        assignment['face_name'],
        assignment['pixel_x'],
        assignment['pixel_y'],
    ))
    selected = {}
    used_sources = set()
    for assignment in candidates:
        source = (
            assignment['face_name'],
            assignment['pixel_x'],
            assignment['pixel_y'],
        )
        if assignment['object_id'] in selected or source in used_sources:
            continue
        selected[assignment['object_id']] = assignment
        used_sources.add(source)
    return selected


def _observation(object_id, global_direction, scene):
    local_direction = _rotate_heading(global_direction, -scene['heading'])
    azimuth, elevation = _direction_to_azimuth_elevation(local_direction)
    return {
        'object_id': object_id,
        'azimuth': azimuth,
        'elevation': elevation,
        'confidence': 1.0,
    }


def _baseline_observations(case, common_ids, scene):
    observations = []
    for object_id in common_ids:
        record = _selected_record(case, object_id)
        local_azimuth = (record['face_azimuth_deg'] - scene['heading']) % 360.0
        observations.append({
            'object_id': object_id,
            'azimuth': local_azimuth,
            'elevation': record['face_elevation_deg'],
            'confidence': 1.0,
        })
    return observations


def _solve_variant(observations, scene, provider, args):
    if len(observations) < 3:
        return {
            'valid': False,
            'failure_reason': 'insufficient_observations',
            'observation_count': len(observations),
            'geographic_error_meters': None,
            'cost': None,
            'rms_residual': None,
        }
    started = time.perf_counter()
    result = solve(
        observations,
        scene['timestamp'],
        provider,
        (51.5, -0.1, 0.0),
        max_nfev=args.max_nfev,
        max_starts=args.max_starts,
        global_search=True,
    )
    return {
        'valid': result.valid,
        'failure_reason': result.failure_reason,
        'observation_count': len(observations),
        'latitude': float(result.x[0]),
        'longitude': float(result.x[1]),
        'heading': float(result.x[2]),
        'geographic_error_meters': _geographic_error(
            float(result.x[0]), float(result.x[1]), scene
        ) if result.valid else None,
        'cost': float(result.cost),
        'rms_residual': float(result.rms_residual),
        'runtime_seconds': time.perf_counter() - started,
    }


def _variant_observations(case, common_ids, selected_assignments, scene, only_ids=None):
    baseline_by_id = {
        observation['object_id']: observation
        for observation in _baseline_observations(case, common_ids, scene)
    }
    observations = []
    for object_id in common_ids:
        assignment = selected_assignments.get(object_id)
        if only_ids is not None and object_id not in only_ids:
            assignment = None
        if assignment is None:
            observations.append(baseline_by_id[object_id])
        else:
            observations.append(_observation(
                object_id,
                np.asarray(assignment['global_direction'], dtype=np.float64),
                scene,
            ))
    return observations


def _run_scene(report, scene_report, args, provider, solver):
    scene = scene_report['scene']
    common_ids = scene_report['common_all_face_sizes_ids']
    results = {}
    for face_size in args.face_sizes:
        case, capture_directory = _load_case(
            report, scene_report, face_size, args.capture_directory
        )
        detections_by_face = _detect_face_detections(
            capture_directory,
            face_size,
            scene,
            args.star_threshold,
            args.max_candidates,
        )
        face_results = {}
        all_assignments = []
        for face_name, detections in detections_by_face.items():
            independent = _identify_face_crops(
                detections,
                args.crop_fov,
                solver,
                args.crop_size,
            )
            face_results[face_name] = {
                'detection_count': len(detections),
                'hypothesis_count': independent['hypothesis_count'],
                'assignment_count': len(independent['assignments']),
            }
            all_assignments.extend(independent['assignments'])
        annotated_assignments = _annotate_assignments(
            all_assignments,
            case,
            scene,
            common_ids,
            provider,
        )
        selected_common = _choose_common_assignments(
            annotated_assignments,
            common_ids,
        )
        baseline = _baseline_observations(case, common_ids, scene)
        variants = {
            'panorama_transferred': _solve_variant(
                baseline, scene, provider, args
            ),
            'hip26241_replacement': _solve_variant(
                _variant_observations(
                    case, common_ids, selected_common, scene,
                    only_ids={'HIP_26241'},
                ) if 'HIP_26241' in selected_common else baseline,
                scene,
                provider,
                args,
            ),
            'independent_hybrid': _solve_variant(
                _variant_observations(
                    case, common_ids, selected_common, scene
                ),
                scene,
                provider,
                args,
            ),
            'independent_only_common_ids': _solve_variant(
                [
                    _observation(
                        object_id,
                        np.asarray(assignment['global_direction'], dtype=np.float64),
                        scene,
                    )
                    for object_id, assignment in selected_common.items()
                ],
                scene,
                provider,
                args,
            ),
        }
        hip_assignments = [
            assignment for assignment in annotated_assignments
            if assignment['object_id'] == 'HIP_26241'
        ]
        results[str(face_size)] = {
            'face_identification': face_results,
            'independent_assignment_count': len(annotated_assignments),
            'independent_common_assignment_count': len(selected_common),
            'independent_assignments': annotated_assignments,
            'selected_common_assignments': selected_common,
            'hip26241_independent_assignments': hip_assignments,
            'solver_variants': variants,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-directory', type=Path, default=None)
    parser.add_argument('--face-sizes', type=int, nargs='+', default=[512, 1024, 2048])
    parser.add_argument('--crop-fov', type=float, default=30.0)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--star-threshold', type=float, default=5.0)
    parser.add_argument('--max-candidates', type=int, default=64)
    parser.add_argument('--max-nfev', type=int, default=60)
    parser.add_argument('--max-starts', type=int, default=5)
    args = parser.parse_args()

    report = json.loads(args.transfer.read_text(encoding='utf-8'))
    provider = EphemerisProvider()
    solver = tetra3.Tetra3(load_database='default_database')
    scene_reports = report.get('scenes', [])
    if len(scene_reports) != 1:
        raise ValueError('expected the single saved New York resolution scene')
    scene_report = scene_reports[0]
    started = time.perf_counter()
    results = _run_scene(report, scene_report, args, provider, solver)
    output = {
        'diagnostic_only': True,
        'ground_truth_used_for_identification_or_selection': False,
        'posthoc_truth_annotations': True,
        'source_transfer_report': args.transfer.name,
        'scene': scene_report['scene'],
        'capture_directory': str(
            args.capture_directory or scene_report['capture_directory']
        ),
        'method': {
            'independent_identifier': 'Tetra3 on face-local overlapping virtual perspective crops',
            'face_fov_degrees': 95.0,
            'crop_fov_degrees': args.crop_fov,
            'crop_size': args.crop_size,
            'star_threshold_sigma': args.star_threshold,
            'max_candidates': args.max_candidates,
            'match_radius': 0.02,
            'match_threshold': 0.001,
            'minimum_matches': 4,
            'ground_truth_used_for_selection': False,
            'production_changed': False,
        },
        'common_ids_used_for_controlled_solver_comparison': scene_report[
            'common_all_face_sizes_ids'
        ],
        'results': results,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'crop_fov_degrees': args.crop_fov,
        'results': {
            size: {
                'independent_assignment_count': value['independent_assignment_count'],
                'independent_common_assignment_count': value[
                    'independent_common_assignment_count'
                ],
                'hip26241': [
                    {
                        key: assignment.get(key)
                        for key in (
                            'face_name',
                            'pixel_x',
                            'pixel_y',
                            'match_count',
                            'match_probability',
                            'panorama_selected_source',
                            'panorama_distance_degrees',
                            'truth_error',
                        )
                    }
                    for assignment in value['hip26241_independent_assignments']
                ],
                'solver_variants': {
                    name: {
                        key: variant.get(key)
                        for key in (
                            'valid',
                            'observation_count',
                            'geographic_error_meters',
                            'cost',
                            'rms_residual',
                        )
                    }
                    for name, variant in value['solver_variants'].items()
                },
            }
            for size, value in results.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
