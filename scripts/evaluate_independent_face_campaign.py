#!/usr/bin/env python3
"""Evaluate fixed independent face identification on saved capture scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'celestial_localisation' / 'src' / 'celestial_detector'))
sys.path.insert(0, str(ROOT / 'celestial_localisation' / 'src' / 'celestial_localiser'))

from diagnose_independent_face_identification import (  # noqa: E402
    _choose_common_assignments,
    _direction_to_azimuth_elevation,
    _identify_face_crops,
    _observation,
    _rotate_heading,
    _solve_variant,
    _truth_error,
)
from celestial_detector.face_direction import match_face_detections  # noqa: E402
from celestial_detector.face_direction import face_pixel_to_direction  # noqa: E402
from celestial_detector.star_detector import detect_stars  # noqa: E402
from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_simulation.projection import cube_faces  # noqa: E402


def _load_capture(capture_path):
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    scene = capture['scene_definition']
    settings = capture['resolved_settings']['compare_configuration']
    return capture, scene, settings


def _panorama_detections(capture, scene):
    detections = []
    for observation in capture['panorama_observations']:
        direction = _rotate_heading(
            _direction_from_observation(observation),
            scene['heading'],
        )
        detections.append({
            'object_id': observation['object_id'],
            'direction': direction,
            'brightness': observation.get('confidence', 1.0),
        })
    return detections


def _direction_from_observation(observation):
    azimuth = np.radians(float(observation['azimuth']))
    elevation = np.radians(float(observation['elevation']))
    cosine = np.cos(elevation)
    return np.asarray((
        np.cos(azimuth) * cosine,
        np.sin(azimuth) * cosine,
        np.sin(elevation),
    ), dtype=np.float64)


def _flatten_detections(detections_by_face):
    return [
        detection
        for detections in detections_by_face.values()
        for detection in detections
    ]


def _detect_flat_capture_faces(capture_directory, face_size, scene, threshold, max_candidates):
    detections_by_face = {}
    face_views = {face.name: face for face in cube_faces()}
    for face_name, face in face_views.items():
        image_path = capture_directory / f'face_{face_name}.png'
        image = __import__('cv2').imread(
            str(image_path),
            __import__('cv2').IMREAD_GRAYSCALE,
        )
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
                'direction': _rotate_heading(
                    local_direction, scene['heading']
                ),
            })
        detections_by_face[face_name] = rows
    return detections_by_face


def _saved_face_observations(capture):
    return [dict(observation) for observation in capture['face_observations']]


def _same_source(assignment, face_detection):
    return (
        assignment['face_name'] == face_detection['face_name']
        and abs(assignment['pixel_x'] - face_detection['pixel_x']) < 1.0
        and abs(assignment['pixel_y'] - face_detection['pixel_y']) < 1.0
    )


def _assignment_summary(
    assignments,
    matches,
    saved_source_by_id,
    panorama_ids,
    scene,
    provider,
):
    match_ids = set(matches)
    same_source_count = 0
    alternate_source_count = 0
    novel_id_count = 0
    truth_errors = []
    rows = []
    for assignment in assignments:
        object_id = assignment['object_id']
        matching = matches.get(object_id)
        same_saved_source = (
            saved_source_by_id.get(object_id) == assignment['face_name']
        )
        same_rerun_source = (
            matching is not None
            and _same_source(assignment, matching['face_detection'])
        )
        if object_id not in match_ids:
            novel_id_count += 1
        elif same_saved_source:
            same_source_count += 1
        else:
            alternate_source_count += 1
        truth = _truth_error(
            object_id,
            np.asarray(assignment['global_direction'], dtype=np.float64),
            scene,
            provider,
        )
        if truth is not None:
            truth_errors.append(truth['angular_error_deg'])
        rows.append({
            'object_id': object_id,
            'face_name': assignment['face_name'],
            'pixel_x': assignment['pixel_x'],
            'pixel_y': assignment['pixel_y'],
            'match_count': assignment['match_count'],
            'match_probability': assignment['match_probability'],
            'panorama_match_exists': object_id in match_ids,
            'same_source_as_saved_baseline': same_saved_source,
            'same_source_as_rerun_panorama_match': same_rerun_source,
            'alternate_source_for_panorama_id': (
                object_id in match_ids and not same_saved_source
            ),
            'alternate_source_for_saved_baseline': not same_saved_source,
            'novel_relative_to_panorama': object_id not in panorama_ids,
            'truth_error': truth,
        })
    return {
        'assignment_count': len(assignments),
        'panorama_id_count': len(panorama_ids),
        'panorama_match_id_count': len(match_ids),
        'same_source_count': same_source_count,
        'alternate_source_count': alternate_source_count,
        'novel_id_count': novel_id_count,
        'posthoc_truth_angular_error_deg': {
            'count': len(truth_errors),
            'within_0_1_degree_count': sum(error <= 0.1 for error in truth_errors),
            'median': float(np.median(truth_errors)) if truth_errors else None,
            'p90': float(np.percentile(truth_errors, 90.0)) if truth_errors else None,
        },
        'rows': rows,
    }


def _selected_assignment_summary(
    selected_assignments,
    matches,
    saved_source_by_id,
    scene,
    provider,
):
    same_source_count = 0
    alternate_source_count = 0
    truth_errors = []
    rows = []
    for object_id, assignment in selected_assignments.items():
        matching = matches.get(object_id)
        same_source = (
            saved_source_by_id.get(object_id) == assignment['face_name']
        )
        same_rerun_source = (
            matching is not None
            and _same_source(assignment, matching['face_detection'])
        )
        if same_source:
            same_source_count += 1
        else:
            alternate_source_count += 1
        truth = _truth_error(
            object_id,
            np.asarray(assignment['global_direction'], dtype=np.float64),
            scene,
            provider,
        )
        if truth is not None:
            truth_errors.append(truth['angular_error_deg'])
        rows.append({
            'object_id': object_id,
            'face_name': assignment['face_name'],
            'pixel_x': assignment['pixel_x'],
            'pixel_y': assignment['pixel_y'],
            'match_count': assignment['match_count'],
            'match_probability': assignment['match_probability'],
            'same_source_as_saved_baseline': same_source,
            'same_source_as_rerun_panorama_match': same_rerun_source,
            'truth_error': truth,
        })
    return {
        'selected_count': len(selected_assignments),
        'same_source_count': same_source_count,
        'alternate_source_count': alternate_source_count,
        'posthoc_truth_angular_error_deg': {
            'count': len(truth_errors),
            'within_0_1_degree_count': sum(error <= 0.1 for error in truth_errors),
            'median': float(np.median(truth_errors)) if truth_errors else None,
            'p90': float(np.percentile(truth_errors, 90.0)) if truth_errors else None,
        },
        'rows': rows,
    }


def _hybrid_observations(base_observations, selected_assignments, scene):
    observations = []
    for base_observation in base_observations:
        object_id = base_observation['object_id']
        assignment = selected_assignments.get(object_id)
        if assignment is None:
            observations.append(dict(base_observation))
            continue
        else:
            local_direction = _rotate_heading(
                np.asarray(assignment['global_direction'], dtype=np.float64),
                -scene['heading'],
            )
        azimuth, elevation = _direction_to_azimuth_elevation(local_direction)
        observations.append({
            'object_id': object_id,
            'azimuth': azimuth,
            'elevation': elevation,
            'confidence': 1.0,
        })
    return observations


def _run_capture(capture_path, args, provider, solver):
    capture, scene, settings = _load_capture(capture_path)
    face_size = int(settings['face_size'])
    capture_directory = capture_path.parent
    detections_by_face = _detect_flat_capture_faces(
        capture_directory,
        face_size,
        scene,
        args.star_threshold,
        args.max_candidates,
    )
    all_detections = _flatten_detections(detections_by_face)
    independent_assignments = []
    face_summary = {}
    for face_name, detections in detections_by_face.items():
        result = _identify_face_crops(
            detections,
            args.crop_fov,
            solver,
            args.crop_size,
        )
        face_summary[face_name] = {
            'detection_count': len(detections),
            'hypothesis_count': result['hypothesis_count'],
            'assignment_count': len(result['assignments']),
        }
        independent_assignments.extend(result['assignments'])

    matches = match_face_detections(
        _panorama_detections(capture, scene),
        all_detections,
        (int(settings['panorama_width']), int(settings['panorama_height'])),
        max_distance_degrees=float(settings['face_match_radius']),
        ambiguity_margin_degrees=float(settings['face_match_ambiguity_margin']),
        duplicate_radius_degrees=float(settings['face_duplicate_radius']),
    )['matches']
    panorama_ids = set(
        observation['object_id']
        for observation in capture['panorama_observations']
    )
    baseline_observations = _saved_face_observations(capture)
    saved_source_by_id = {
        row['object_id']: row['source_face']
        for row in capture.get('rows', [])
        if row.get('source_face')
    }
    saved_face_ids = {
        observation['object_id'] for observation in baseline_observations
    }
    selected_assignments = _choose_common_assignments(
        independent_assignments,
        sorted(saved_face_ids),
    )
    assignment_summary = _assignment_summary(
        independent_assignments,
        matches,
        saved_source_by_id,
        panorama_ids,
        scene,
        provider,
    )
    selected_summary = _selected_assignment_summary(
        selected_assignments,
        matches,
        saved_source_by_id,
        scene,
        provider,
    )
    independent_only = [
        _observation(
            object_id,
            np.asarray(assignment['global_direction'], dtype=np.float64),
            scene,
        )
        for object_id, assignment in selected_assignments.items()
    ]
    variants = {
        'panorama_transferred': _solve_variant(
            baseline_observations, scene, provider, args
        ),
        'independent_hybrid': _solve_variant(
            _hybrid_observations(
                baseline_observations,
                selected_assignments,
                scene,
            ),
            scene,
            provider,
            args,
        ),
        'independent_only_matched_ids': _solve_variant(
            independent_only,
            scene,
            provider,
            args,
        ),
    }
    return {
        'scene_id': scene['scene_id'],
        'observation_counts': {
            'panorama_ids': len(panorama_ids),
            'saved_face_observations': len(baseline_observations),
            'rerun_panorama_matches': len(matches),
            'independent_assignments': len(independent_assignments),
            'independent_selected_matched_ids': len(selected_assignments),
        },
        'face_identification': face_summary,
        'matching': {
            'panorama_match_ids': sorted(matches),
            'saved_face_ids': sorted(saved_face_ids),
            'saved_source_faces': saved_source_by_id,
            'independent_selected_ids': sorted(selected_assignments),
        },
        'solver_input_provenance': {
            'panorama_transferred': 'saved capture face_observations',
            'independent_hybrid': (
                'saved capture face_observations with independent replacements'
            ),
        },
        'identity_comparison': assignment_summary,
        'selected_identity_comparison': selected_summary,
        'solver_variants': variants,
    }


def _summary(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {
            'count': 0,
            'median': None,
            'p90': None,
            'mean': None,
            'max': None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        'count': len(array),
        'median': float(np.median(array)),
        'p90': float(np.percentile(array, 90.0)),
        'mean': float(np.mean(array)),
        'max': float(np.max(array)),
    }


def _aggregate_results(results):
    variant_summary = {}
    for variant_name in (
        'panorama_transferred',
        'independent_hybrid',
        'independent_only_matched_ids',
    ):
        values = [result['solver_variants'][variant_name] for result in results]
        valid = [
            value for value in values
            if value['valid'] and value['geographic_error_meters'] is not None
        ]
        variant_summary[variant_name] = {
            'attempt_count': len(values),
            'valid_count': len(valid),
            'invalid_count': len(values) - len(valid),
            'observation_count': _summary(
                value['observation_count'] for value in values
            ),
            'geographic_error_meters': _summary(
                value['geographic_error_meters'] for value in valid
            ),
        }

    paired = []
    for result in results:
        baseline = result['solver_variants']['panorama_transferred']
        hybrid = result['solver_variants']['independent_hybrid']
        if not (
            baseline['valid']
            and hybrid['valid']
            and baseline['geographic_error_meters'] is not None
            and hybrid['geographic_error_meters'] is not None
        ):
            continue
        paired.append(
            hybrid['geographic_error_meters']
            - baseline['geographic_error_meters']
        )

    selected_count = 0
    same_source_count = 0
    alternate_source_count = 0
    truth_errors = []
    for result in results:
        comparison = result['selected_identity_comparison']
        selected_count += comparison['selected_count']
        same_source_count += comparison['same_source_count']
        alternate_source_count += comparison['alternate_source_count']
        truth_summary = comparison['posthoc_truth_angular_error_deg']
        if truth_summary['count']:
            truth_errors.extend(
                row['truth_error']['angular_error_deg']
                for row in comparison['rows']
                if row['truth_error'] is not None
            )

    return {
        'solver_variants': variant_summary,
        'paired_valid_baseline_vs_independent_hybrid': {
            'paired_count': len(paired),
            'improved_count': sum(value < 0.0 for value in paired),
            'worsened_count': sum(value > 0.0 for value in paired),
            'unchanged_count': sum(value == 0.0 for value in paired),
            'delta_independent_minus_baseline_meters': _summary(paired),
        },
        'selected_correspondences': {
            'selected_count': selected_count,
            'same_source_count': same_source_count,
            'alternate_source_count': alternate_source_count,
            'alternate_source_fraction': (
                alternate_source_count / selected_count
                if selected_count else None
            ),
            'posthoc_truth_angular_error_deg': _summary(truth_errors),
            'ground_truth_used_for_selection': False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-fov', type=float, default=45.0)
    parser.add_argument('--crop-size', type=int, default=512)
    parser.add_argument('--star-threshold', type=float, default=5.0)
    parser.add_argument('--max-candidates', type=int, default=64)
    parser.add_argument('--max-nfev', type=int, default=60)
    parser.add_argument('--max-starts', type=int, default=5)
    args = parser.parse_args()

    provider = EphemerisProvider()
    solver = __import__('tetra3').Tetra3(load_database='default_database')
    results = []
    started = time.perf_counter()
    for capture_path in sorted(args.captures.glob('*/capture.json')):
        results.append(_run_capture(capture_path, args, provider, solver))
    output = {
        'diagnostic_only': True,
        'ground_truth_used_for_identification_or_selection': False,
        'posthoc_truth_annotations': True,
        'production_changed': False,
        'capture_directory': str(args.captures),
        'fixed_settings': {
            'crop_fov_degrees': args.crop_fov,
            'crop_size': args.crop_size,
            'star_threshold_sigma': args.star_threshold,
            'max_candidates': args.max_candidates,
            'face_fov_degrees': 95.0,
            'face_match_radius_degrees': 2.0,
            'face_match_ambiguity_margin_degrees': 0.25,
            'face_duplicate_radius_degrees': 0.15,
        },
        'aggregate': _aggregate_results(results),
        'results': results,
        'elapsed_seconds': time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'capture_count': len(results),
        'summary': [
            {
                'scene_id': result['scene_id'],
                'saved_face_observations': result['observation_counts'][
                    'saved_face_observations'
                ],
                'rerun_panorama_matches': result['observation_counts'][
                    'rerun_panorama_matches'
                ],
                'independent_selected': result['observation_counts'][
                    'independent_selected_matched_ids'
                ],
                'same_source': result['identity_comparison']['same_source_count'],
                'alternate_source': result['identity_comparison']['alternate_source_count'],
                'independent_hybrid_error_m': result['solver_variants'][
                    'independent_hybrid'
                ]['geographic_error_meters'],
                'panorama_error_m': result['solver_variants'][
                    'panorama_transferred'
                ]['geographic_error_meters'],
            }
            for result in results
        ],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
