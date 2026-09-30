#!/usr/bin/env python3
"""Evaluate ground-truth-free face measurement quality rules on saved captures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for source_directory in (
    ROOT / 'celestial_localisation' / 'src' / 'celestial_simulation',
    ROOT / 'celestial_localisation' / 'src' / 'celestial_detector',
    ROOT / 'celestial_localisation' / 'src' / 'celestial_localiser',
):
    sys.path.insert(0, str(source_directory))

from celestial_detector.face_direction import (  # noqa: E402
    angular_separation_degrees,
    face_pixel_to_direction,
    match_face_detections,
)
from celestial_detector.star_detector import detect_stars  # noqa: E402
from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_localiser.pose_solver import solve  # noqa: E402
from celestial_simulation.projection import cube_faces  # noqa: E402

SOLVER_SETTINGS = {
    'max_nfev': 60,
    'max_starts': 5,
    'global_search': True,
}

POLICIES = (
    'baseline',
    'quality_weighted',
    'agreement_only',
    'weighted_without_boundary',
    'weighted_without_agreement',
    'weighted_without_overlap',
    'weighted_without_crowding',
    'weighted_without_saturation',
    'quality_rejected',
)
QUALITY_ABLATIONS = {
    'quality_weighted': frozenset(),
    'agreement_only': frozenset(('boundary', 'overlap', 'crowding', 'saturation')),
    'weighted_without_boundary': frozenset(('boundary',)),
    'weighted_without_agreement': frozenset(('agreement',)),
    'weighted_without_overlap': frozenset(('overlap',)),
    'weighted_without_crowding': frozenset(('crowding',)),
    'weighted_without_saturation': frozenset(('saturation',)),
}


def _direction_from_azimuth_elevation(azimuth, elevation):
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


def _crop_metrics(image, pixel_x, pixel_y, radius=6):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    x_center = int(round(float(pixel_x)))
    y_center = int(round(float(pixel_y)))
    x_min = max(0, x_center - radius)
    x_max = min(width, x_center + radius + 1)
    y_min = max(0, y_center - radius)
    y_max = min(height, y_center + radius + 1)
    patch = gray[y_min:y_max, x_min:x_max].astype(np.float64)
    if patch.size == 0:
        return {
            'saturation_fraction': 0.0,
            'component_count': 0,
            'component_area_fraction': 0.0,
            'peak_to_background': 0.0,
        }
    border = np.concatenate((
        patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1],
    ))
    background = float(np.median(border))
    peak = float(np.max(patch))
    threshold = background + 0.5 * max(0.0, peak - background)
    mask = (patch >= threshold).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    component_areas = [
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, component_count)
        if stats[index, cv2.CC_STAT_AREA] >= 2
    ]
    return {
        'saturation_fraction': float(np.mean(patch >= 250.0)),
        'component_count': len(component_areas),
        'component_area_fraction': (
            float(max(component_areas) / patch.size) if component_areas else 0.0
        ),
        'peak_to_background': float((peak - background) / (np.std(border) + 1e-6)),
    }


def _quality_metrics(face_detection, image_by_face, canonical_direction, all_detections):
    face_name = face_detection['face_name']
    image = image_by_face[face_name]
    height, width = image.shape[:2]
    pixel_x = float(face_detection['pixel_x'])
    pixel_y = float(face_detection['pixel_y'])
    center_x = pixel_x + 0.5
    center_y = pixel_y + 0.5
    boundary_margin = min(
        center_x / width,
        1.0 - center_x / width,
        center_y / height,
        1.0 - center_y / height,
    )
    other_detections = [
        detection for detection in all_detections
        if not (
            detection['face_name'] == face_name
            and abs(detection['pixel_x'] - pixel_x) < 1.0
            and abs(detection['pixel_y'] - pixel_y) < 1.0
        )
    ]
    nearest_neighbor = min(
        (
            angular_separation_degrees(face_detection['direction'], detection['direction'])
            for detection in other_detections
        ),
        default=1.0,
    )
    overlap_candidates = [
        detection for detection in all_detections
        if angular_separation_degrees(
            canonical_direction,
            detection['direction'],
        ) <= 0.20
    ]
    overlap_faces = {detection['face_name'] for detection in overlap_candidates}
    overlap_spread = max(
        (
            angular_separation_degrees(
                face_detection['direction'],
                detection['direction'],
            )
            for detection in overlap_candidates
        ),
        default=0.0,
    )
    crop = _crop_metrics(image, pixel_x, pixel_y)
    match_distance = angular_separation_degrees(
        face_detection['direction'], canonical_direction
    )
    boundary_score = min(1.0, boundary_margin / 0.05)
    agreement_score = math.exp(-match_distance / 0.15)
    overlap_score = math.exp(-overlap_spread / 0.05) if len(overlap_faces) > 1 else 1.0
    crowding_score = min(1.0, nearest_neighbor / 0.20)
    saturation_score = max(0.1, 1.0 - crop['saturation_fraction'])
    quality_components = {
        'boundary': boundary_score,
        'agreement': agreement_score,
        'overlap': overlap_score,
        'crowding': crowding_score,
        'saturation': saturation_score,
    }
    quality_score = float(np.prod(list(quality_components.values())))
    return {
        'face_name': face_name,
        'pixel_x': pixel_x,
        'pixel_y': pixel_y,
        'boundary_margin_normalized': boundary_margin,
        'match_distance_degrees': match_distance,
        'nearest_neighbor_degrees': nearest_neighbor,
        'overlap_candidate_count': len(overlap_candidates),
        'overlap_face_count': len(overlap_faces),
        'overlap_spread_degrees': overlap_spread,
        **crop,
        'quality_components': quality_components,
        'quality_score': quality_score,
    }


def _load_capture(capture_path):
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    scene = capture['scene_definition']
    settings = capture['resolved_settings']['compare_configuration']
    images = {}
    for face in cube_faces():
        path = capture_path.parent / capture['image_files'][face.name]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'failed to read {path}')
        images[face.name] = image
    heading_rotation = Rotation.from_euler('z', scene['heading'], degrees=True)
    panorama = []
    for observation in capture['panorama_observations']:
        panorama.append({
            **observation,
            'direction': heading_rotation.apply(
                _direction_from_azimuth_elevation(
                    observation['azimuth'], observation['elevation']
                )
            ),
        })
    return capture, scene, settings, images, panorama


def _capture_measurements(capture_path):
    capture, scene, settings, images, panorama = _load_capture(capture_path)
    all_detections = []
    for face in cube_faces():
        gray = cv2.cvtColor(images[face.name], cv2.COLOR_BGR2GRAY)
        detections = detect_stars(
            gray,
            threshold_sigma=float(settings.get('star_threshold', 5.0)),
            minimum_elevation_degrees=-90.0,
            max_candidates=int(settings.get('max_candidates', 64)),
        )
        heading_rotation = Rotation.from_euler('z', scene['heading'], degrees=True)
        for detection in detections:
            local = face_pixel_to_direction(
                face,
                detection['pixel_x'],
                detection['pixel_y'],
                images[face.name].shape[1],
                images[face.name].shape[0],
                float(settings.get('face_fov', 95.0)),
            )
            all_detections.append({
                **detection,
                'face_name': face.name,
                'direction': heading_rotation.apply(local),
            })
    matching = match_face_detections(
        panorama,
        all_detections,
        (2048, 1024),
        max_distance_degrees=2.0,
        ambiguity_margin_degrees=0.25,
        duplicate_radius_degrees=0.15,
    )
    canonical_by_id = {
        detection['object_id']: detection['direction']
        for detection in panorama
    }
    metrics = {}
    for object_id, match in matching['matches'].items():
        metrics[object_id] = _quality_metrics(
            match['face_detection'],
            images,
            canonical_by_id[object_id],
            all_detections,
        )
    observations = []
    inverse_heading = Rotation.from_euler('z', -scene['heading'], degrees=True)
    for object_id, match in matching['matches'].items():
        local = inverse_heading.apply(match['face_detection']['direction'])
        azimuth, elevation = _azimuth_elevation(local)
        observations.append({
            'object_id': object_id,
            'azimuth': azimuth,
            'elevation': elevation,
            'confidence': 1.0,
        })
    return {
        'capture_id': scene['scene_id'],
        'capture': capture,
        'scene': scene,
        'panorama': panorama,
        'all_detections': all_detections,
        'matching': matching,
        'metrics': metrics,
        'observations': observations,
    }


def _solve(observations, scene, provider):
    if len(observations) < 3:
        return {
            'valid': False,
            'failure_reason': 'insufficient_observations',
            'geographic_error_meters': None,
            'rms_residual': None,
            'runtime_seconds': 0.0,
            'observation_count': len(observations),
        }
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
    return {
        'valid': result.valid,
        'failure_reason': result.failure_reason,
        'geographic_error_meters': _geographic_error(
            float(result.x[0]), float(result.x[1]), scene
        ),
        'rms_residual': result.rms_residual,
        'runtime_seconds': time.perf_counter() - started,
        'observation_count': len(observations),
    }


def _quality_weight(metrics, disabled_signals= frozenset()):
    components = metrics['quality_components']
    active = [
        value for name, value in components.items()
        if name not in disabled_signals
    ]
    return max(0.05, min(1.0, float(np.prod(active))))


def _quality_keep(metrics):
    return (
        metrics['boundary_margin_normalized'] >= 0.02
        and metrics['match_distance_degrees'] <= 0.15
        and (
            metrics['overlap_face_count'] < 2
            or metrics['overlap_spread_degrees'] <= 0.05
        )
        and metrics['nearest_neighbor_degrees'] >= 0.08
    )


def _truth_angular_errors(observations, scene, provider):
    errors = []
    for observation in observations:
        expected = provider.predict(
            observation['object_id'],
            scene['timestamp'],
            scene['latitude'],
            scene['longitude'],
            scene['altitude'],
        )
        if expected is None:
            continue
        observed = _direction_from_azimuth_elevation(
            observation['azimuth'] + scene['heading'], observation['elevation']
        )
        errors.append(_angular_error(observed, _direction_from_azimuth_elevation(*expected)))
    return errors


def _evaluate_capture(measurements, provider):
    base_by_id = {
        observation['object_id']: observation
        for observation in measurements['observations']
    }
    policies = {}
    for policy in POLICIES:
        observations = []
        for object_id, observation in base_by_id.items():
            metric = measurements['metrics'][object_id]
            if policy == 'quality_rejected' and not _quality_keep(metric):
                continue
            item = dict(observation)
            disabled_signals = QUALITY_ABLATIONS.get(policy)
            if disabled_signals is not None:
                item['confidence'] = _quality_weight(metric, disabled_signals)
            observations.append(item)
        solve_result = _solve(observations, measurements['scene'], provider)
        policies[policy] = {
            **solve_result,
            'dropped_ids': sorted(set(base_by_id) - {
                observation['object_id'] for observation in observations
            }),
            'quality_weights': {
                observation['object_id']: observation['confidence']
                for observation in observations
                if policy == 'quality_weighted'
            },
            'truth_angular_error_deg': _summary(
                _truth_angular_errors(observations, measurements['scene'], provider)
            ),
        }
    baseline_error = policies['baseline']['geographic_error_meters']
    for policy_result in policies.values():
        error = policy_result['geographic_error_meters']
        policy_result['delta_vs_baseline_meters'] = (
            error - baseline_error
            if error is not None and baseline_error is not None
            else None
        )
    return policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--captures', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'face_failure_captures_20260929',
    )
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output'
        / 'face_quality_rules_20260929.json',
    )
    args = parser.parse_args()
    provider = EphemerisProvider()
    captures = []
    for capture_path in sorted(args.captures.glob('*/capture.json')):
        measurements = _capture_measurements(capture_path)
        captures.append({
            'capture_id': measurements['capture_id'],
            'matching': {
                'panorama_id_count': measurements['matching']['panorama_id_count'],
                'face_detection_count': measurements['matching']['face_detection_count'],
                'deduplicated_face_detection_count': measurements['matching'][
                    'deduplicated_face_detection_count'
                ],
                'lost_object_ids': measurements['matching']['lost_object_ids'],
                'ambiguous_object_ids': measurements['matching'][
                    'ambiguous_object_ids'
                ],
                'matched_object_ids': sorted(measurements['metrics']),
            },
            'measurement_metrics': measurements['metrics'],
            'policies': _evaluate_capture(measurements, provider),
        })
    output = {
        'ground_truth_used_for_selection': False,
        'quality_rules': {
            'quality_weighted': 'confidence is an observable quality score from boundary, agreement, overlap, crowding and crop metrics',
            'ablations': {
                policy: f'disables {sorted(QUALITY_ABLATIONS[policy])}'
                for policy in QUALITY_ABLATIONS
                if policy != 'quality_weighted'
            },
            'quality_rejected': 'reject boundary<0.02, match distance>0.15 deg, divergent overlap>0.05 deg, or nearest detection<0.08 deg',
        },
        'solver_settings': {'max_nfev': 60, 'max_starts': 5, 'global_search': True},
        'captures': captures,
        'aggregate': {
            policy: {
                'valid_count': sum(
                    capture['policies'][policy]['valid']
                    for capture in captures
                ),
                'capture_count': len(captures),
                'dropped_observations': sum(
                    len(capture['policies'][policy]['dropped_ids'])
                    for capture in captures
                ),
                'geographic_error_meters': _summary(
                    capture['policies'][policy]['geographic_error_meters']
                    for capture in captures
                ),
                'rms_residual': _summary(
                    capture['policies'][policy]['rms_residual']
                    for capture in captures
                ),
                'runtime_seconds': _summary(
                    capture['policies'][policy]['runtime_seconds']
                    for capture in captures
                ),
                'truth_angular_error_deg': _summary(
                    capture['policies'][policy]['truth_angular_error_deg']['median']
                    for capture in captures
                ),
                'delta_vs_baseline_meters': _summary(
                    capture['policies'][policy]['delta_vs_baseline_meters']
                    for capture in captures
                ),
            }
            for policy in POLICIES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'capture_count': len(captures),
        'aggregate': output['aggregate'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
