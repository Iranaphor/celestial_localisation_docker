#!/usr/bin/env python3
"""Compare Astropy, original-face and panorama star directions for one scene."""

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

from celestial_detector.angular_projection import pixel_to_az_el
from celestial_detector.face_direction import (
    face_pixel_to_direction,
    match_face_detections,
)
from celestial_detector.star_detector import detect_stars
from celestial_detector.star_identifier import StarIdentifier, identify_stars
from celestial_localiser.ephemeris import EphemerisProvider
from celestial_localiser.pose_solver import solve
from celestial_simulation.projection import compose_equirectangular, cube_faces
from celestial_simulation.stellarium_browser import StellariumBrowserBridge


def direction_from_azimuth_elevation(azimuth_degrees, elevation_degrees):
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    cosine = math.cos(elevation)
    return np.asarray((
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ), dtype=np.float64)


def angular_error_degrees(first, second):
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def azimuth_elevation_from_direction(direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return (
        math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))),
    )


def summarize(errors):
    if not errors:
        return {'count': 0, 'median_deg': None, 'p90_deg': None, 'max_deg': None}
    return {
        'count': len(errors),
        'median_deg': float(np.median(errors)),
        'p90_deg': float(np.percentile(errors, 90.0)),
        'max_deg': float(np.max(errors)),
    }


def create_bridge(args):
    return StellariumBrowserBridge(
        engine_js=args.engine_js,
        engine_wasm=args.engine_wasm,
        data_root=args.data_root,
        browser_executable=args.browser_executable,
        face_size=args.face_size,
        timeout_seconds=args.timeout,
        show_object_labels=False,
        enable_landscape=False,
    )


def _save_render_capture(capture_dir, faces, panorama):
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    image_files = {}
    for face in cube_faces():
        image_path = capture_dir / f'face_{face.name}.png'
        if not cv2.imwrite(str(image_path), faces[face.name]):
            raise OSError(f'failed to write {image_path}')
        image_files[face.name] = image_path.name
    panorama_path = capture_dir / 'panorama.png'
    if not cv2.imwrite(str(panorama_path), panorama):
        raise OSError(f'failed to write {panorama_path}')
    image_files['panorama'] = panorama_path.name
    return image_files


def run(args, bridge=None, identifier=None, ephemeris=None, capture_dir=None):
    owns_bridge = bridge is None
    if bridge is None:
        bridge = create_bridge(args)

    scene_started = time.perf_counter()
    render_started = time.perf_counter()
    try:
        faces = bridge.render_faces(
            latitude=args.latitude,
            longitude=args.longitude,
            altitude=args.altitude,
            timestamp_ms=args.timestamp * 1000.0,
            field_of_view_degrees=args.face_fov,
            yaw_degrees=args.heading,
        )
    finally:
        if owns_bridge:
            bridge.close()
    render_seconds = time.perf_counter() - render_started

    panorama = compose_equirectangular(
        faces,
        args.panorama_width,
        args.panorama_height,
        args.face_fov,
    )
    capture_image_files = None
    if capture_dir:
        capture_image_files = _save_render_capture(
            capture_dir,
            faces,
            panorama,
        )
    if identifier is None:
        identifier = StarIdentifier(tile_size=args.star_tile_size)
    if ephemeris is None:
        ephemeris = EphemerisProvider()
    face_detection_started = time.perf_counter()
    face_detections = []
    for face in cube_faces():
        image = faces[face.name]
        detections = detect_stars(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            threshold_sigma=args.star_threshold,
            minimum_elevation_degrees=-90.0,
            max_candidates=args.max_candidates,
        )
        for detection in detections:
            direction = face_pixel_to_direction(
                face,
                detection['pixel_x'],
                detection['pixel_y'],
                args.face_size,
                args.face_size,
                args.face_fov,
            )
            direction = Rotation.from_euler(
                'z', args.heading, degrees=True
            ).apply(direction)
            face_detections.append({
                'face': face.name,
                'direction': direction,
                'brightness': detection['brightness'],
            })
    face_detection_seconds = time.perf_counter() - face_detection_started

    panorama_detection_started = time.perf_counter()
    panorama_detections = detect_stars(
        cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY),
        threshold_sigma=args.star_threshold,
        minimum_elevation_degrees=args.min_elevation,
        max_candidates=args.max_candidates,
    )
    panorama_detections = identify_stars(
        panorama_detections,
        (args.panorama_width, args.panorama_height),
        identifier,
    )
    panorama_by_id = {}
    panorama_matches = []
    panorama_detection_by_id = {}
    expected_by_id = {}
    for detection in panorama_detections:
        object_id = detection.get('object_id', 'UNKNOWN')
        if object_id == 'UNKNOWN':
            continue
        azimuth, elevation = pixel_to_az_el(
            detection['pixel_x'],
            detection['pixel_y'],
            args.panorama_width,
            args.panorama_height,
        )
        panorama_direction = direction_from_azimuth_elevation(
            azimuth + args.heading,
            elevation,
        )
        panorama_match = {
            **detection,
            'direction': panorama_direction,
        }
        panorama_matches.append(panorama_match)
        previous = panorama_detection_by_id.get(object_id)
        if previous is None or detection.get('brightness', 0.0) > previous.get('brightness', 0.0):
            panorama_by_id[object_id] = panorama_direction
            panorama_detection_by_id[object_id] = panorama_match
        expected = ephemeris.predict(
            object_id,
            args.timestamp,
            args.latitude,
            args.longitude,
            args.altitude,
        )
        if expected is not None:
            expected_by_id[object_id] = direction_from_azimuth_elevation(*expected)
    panorama_detection_seconds = time.perf_counter() - panorama_detection_started

    matching_started = time.perf_counter()
    matching = match_face_detections(
        panorama_matches,
        face_detections,
        (args.panorama_width, args.panorama_height),
        max_distance_degrees=args.face_match_radius,
        ambiguity_margin_degrees=args.face_match_ambiguity_margin,
        duplicate_radius_degrees=args.face_duplicate_radius,
    )
    matching_seconds = time.perf_counter() - matching_started
    face_errors = []
    panorama_errors = []
    face_by_id = {
        object_id: {
            'face': match['face_detection'].get('face', match['face_detection'].get('face_name', 'unknown')),
            'direction': match['face_detection']['direction'],
            'error': match['angular_distance_degrees'],
        }
        for object_id, match in matching['matches'].items()
    }

    common_ids = sorted(set(face_by_id) & set(panorama_by_id) & set(expected_by_id))
    rows = []
    for object_id in common_ids:
        expected = expected_by_id[object_id]
        face_direction_value = face_by_id[object_id]['direction']
        panorama_direction = panorama_by_id[object_id]
        face_error = angular_error_degrees(face_direction_value, expected)
        panorama_error = angular_error_degrees(panorama_direction, expected)
        face_errors.append(face_error)
        panorama_errors.append(panorama_error)
        rows.append({
            'object_id': object_id,
            'source_face': face_by_id[object_id]['face'],
            'face_error_deg': face_error,
            'panorama_error_deg': panorama_error,
        })

    def solve_direction_set(direction_by_id):
        direction_started = time.perf_counter()
        observations = []
        inverse_heading = Rotation.from_euler('z', -args.heading, degrees=True)
        for object_id in common_ids:
            local_direction = inverse_heading.apply(direction_by_id[object_id])
            azimuth, elevation = azimuth_elevation_from_direction(local_direction)
            observations.append({
                'object_id': object_id,
                'azimuth': azimuth,
                'elevation': elevation,
                'confidence': 1.0,
            })
        solve_started = time.perf_counter()
        result = solve(
            observations,
            args.timestamp,
            ephemeris,
            (
                args.initial_latitude,
                args.initial_longitude,
                args.initial_heading,
            ),
            max_nfev=args.max_nfev,
            global_search=args.global_search,
            max_starts=args.max_starts,
        )
        return {
            'valid': result.valid,
            'failure_reason': result.failure_reason,
            'latitude': float(result.x[0]),
            'longitude': float(result.x[1]),
            'heading': float(result.x[2]),
            'cost': result.cost,
            'rms_residual': result.rms_residual,
            'max_residual': result.max_residual,
            'selected_nfev': result.nfev,
            'total_nfev': result.total_nfev,
            'starts_tried': result.starts_tried,
            'direction_build_seconds': solve_started - direction_started,
            'runtime_seconds': time.perf_counter() - solve_started,
            'total_runtime_seconds': time.perf_counter() - direction_started,
            'observations': observations,
            'start_diagnostics': result.start_diagnostics,
            'message': result.message,
        }

    solver_inputs = {
        'face': {
            object_id: face_by_id[object_id]['direction']
            for object_id in common_ids
        },
        'panorama': {
            object_id: panorama_by_id[object_id]
            for object_id in common_ids
        },
    }
    solver_results = {
        representation: solve_direction_set(direction_by_id)
        for representation, direction_by_id in solver_inputs.items()
    }

    def geographic_error(result):
        if not all(math.isfinite(result[key]) for key in ('latitude', 'longitude')):
            return None
        latitude_delta = math.radians(result['latitude'] - args.latitude)
        longitude_delta = math.radians(
            (result['longitude'] - args.longitude + 180.0) % 360.0 - 180.0
        )
        haversine_a = (
            math.sin(latitude_delta / 2.0) ** 2
            + math.cos(math.radians(args.latitude))
            * math.cos(math.radians(result['latitude']))
            * math.sin(longitude_delta / 2.0) ** 2
        )
        return 2.0 * 6_371_000.0 * math.atan2(
            math.sqrt(haversine_a),
            math.sqrt(max(0.0, 1.0 - haversine_a)),
        )

    for result in solver_results.values():
        result['geographic_error_meters'] = geographic_error(result)

    scene_runtime_seconds = time.perf_counter() - scene_started

    return {
        'configuration': vars(args),
        'capture_image_files': capture_image_files,
        'counts': {
            'face_matched_to_panorama_ids': len(face_by_id),
            'panorama_identified': matching['panorama_id_count'],
            'common_ids': len(common_ids),
            'face_detections': matching['face_detection_count'],
            'deduplicated_face_detections': matching['deduplicated_face_detection_count'],
            'face_stars_lost_during_matching': len(matching['lost_object_ids']),
            'face_stars_rejected_as_ambiguous': len(matching['ambiguous_object_ids']),
        },
        'star_ids': {
            'panorama_identified': sorted(panorama_by_id),
            'face_matched': sorted(face_by_id),
            'common': common_ids,
        },
        'pairing': {
            'method': 'mutual nearest original-face centroid to observed panorama direction',
            'uses_ground_truth_for_pairing': False,
            'face_match_radius_degrees': args.face_match_radius,
            'ambiguity_margin_degrees': args.face_match_ambiguity_margin,
            'duplicate_radius_degrees': args.face_duplicate_radius,
            'lost_object_ids': matching['lost_object_ids'],
            'ambiguous_object_ids': matching['ambiguous_object_ids'],
        },
        'timing': {
            'render_seconds': render_seconds,
            'face_detection_seconds': face_detection_seconds,
            'panorama_detection_seconds': panorama_detection_seconds,
            'matching_seconds': matching_seconds,
            'scene_total_seconds': scene_runtime_seconds,
            'render_shared_by_both': True,
            'panorama_path_seconds': (
                render_seconds
                + panorama_detection_seconds
                + solver_results['panorama']['total_runtime_seconds']
            ),
            'face_path_seconds': (
                render_seconds
                + panorama_detection_seconds
                + face_detection_seconds
                + matching_seconds
                + solver_results['face']['total_runtime_seconds']
            ),
        },
        'solver_configuration': {
            'initial_latitude': args.initial_latitude,
            'initial_longitude': args.initial_longitude,
            'initial_heading': args.initial_heading,
            'robust_loss': 'soft_l1',
            'min_observations': 3,
            'max_nfev': args.max_nfev,
            'global_search': args.global_search,
            'max_starts': args.max_starts,
        },
        'paired_solver_results': solver_results,
        'face_errors_deg': summarize(face_errors),
        'panorama_errors_deg': summarize(panorama_errors),
        'rows': rows,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latitude', type=float, default=23.0)
    parser.add_argument('--longitude', type=float, default=-70.0)
    parser.add_argument('--altitude', type=float, default=0.0)
    parser.add_argument('--heading', type=float, default=17.0)
    parser.add_argument('--timestamp', type=float, default=1700000000.0)
    parser.add_argument('--face-size', type=int, default=512)
    parser.add_argument('--panorama-width', type=int, default=2048)
    parser.add_argument('--panorama-height', type=int, default=1024)
    parser.add_argument('--face-fov', type=float, default=95.0)
    parser.add_argument('--star-threshold', type=float, default=5.0)
    parser.add_argument('--min-elevation', type=float, default=25.0)
    parser.add_argument('--max-candidates', type=int, default=64)
    parser.add_argument('--star-tile-size', type=int, default=512)
    parser.add_argument('--face-match-radius', type=float, default=2.0)
    parser.add_argument('--face-match-ambiguity-margin', type=float, default=0.25)
    parser.add_argument('--face-duplicate-radius', type=float, default=0.15)
    parser.add_argument('--initial-latitude', type=float, default=51.5)
    parser.add_argument('--initial-longitude', type=float, default=-0.1)
    parser.add_argument('--initial-heading', type=float, default=0.0)
    parser.add_argument('--max-nfev', type=int, default=30)
    parser.add_argument('--max-starts', type=int, default=5)
    parser.add_argument('--global-search', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--engine-js', default='/opt/stellarium/stellarium-web-engine.js')
    parser.add_argument('--engine-wasm', default='/opt/stellarium/stellarium-web-engine.wasm')
    parser.add_argument('--data-root', default='/opt/stellarium/data')
    parser.add_argument('--browser-executable', default='')
    parser.add_argument('--timeout', type=float, default=60.0)
    return parser


def main():
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
