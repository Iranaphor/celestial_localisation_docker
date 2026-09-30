#!/usr/bin/env python3
"""Compare face resolutions using one canonical 512-panorama ID set."""

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

from celestial_detector.angular_projection import pixel_to_az_el  # noqa: E402
from celestial_detector.face_direction import (  # noqa: E402
    angular_separation_degrees,
    face_pixel_to_direction,
    match_face_detections,
)
from celestial_detector.star_detector import detect_stars  # noqa: E402
from celestial_detector.star_identifier import StarIdentifier, identify_stars  # noqa: E402
from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_localiser.pose_solver import solve  # noqa: E402
from celestial_simulation.projection import compose_equirectangular, cube_faces  # noqa: E402
from celestial_simulation.stellarium_browser import StellariumBrowserBridge  # noqa: E402

FIXED_SOLVER = {
    'max_nfev': 60,
    'max_starts': 5,
    'global_search': True,
}


def _direction_from_azimuth_elevation(azimuth_degrees, elevation_degrees):
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    cosine = math.cos(elevation)
    return np.asarray((
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ), dtype=np.float64)


def _azimuth_elevation_from_direction(direction):
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    return (
        math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))),
    )


def _angular_summary(values):
    values = [float(value) for value in values]
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


def _geographic_error(result, scene):
    if not all(math.isfinite(result[key]) for key in ('latitude', 'longitude')):
        return None
    latitude_delta = math.radians(result['latitude'] - scene['latitude'])
    longitude_delta = math.radians(
        (result['longitude'] - scene['longitude'] + 180.0) % 360.0 - 180.0
    )
    haversine_a = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(math.radians(scene['latitude']))
        * math.cos(math.radians(result['latitude']))
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * 6_371_000.0 * math.atan2(
        math.sqrt(haversine_a),
        math.sqrt(max(0.0, 1.0 - haversine_a)),
    )


def _create_bridge(args, face_size):
    return StellariumBrowserBridge(
        engine_js=args.engine_js,
        engine_wasm=args.engine_wasm,
        data_root=args.data_root,
        browser_executable=args.browser_executable,
        face_size=face_size,
        timeout_seconds=args.timeout,
        show_object_labels=False,
        enable_landscape=False,
    )


def _render_faces(scene, args, face_size):
    bridge = _create_bridge(args, face_size)
    started = time.perf_counter()
    try:
        faces = bridge.render_faces(
            latitude=scene['latitude'],
            longitude=scene['longitude'],
            altitude=scene['altitude'],
            timestamp_ms=scene['timestamp'] * 1000.0,
            field_of_view_degrees=args.face_fov,
            yaw_degrees=scene['heading'],
        )
    finally:
        bridge.close()
    return faces, time.perf_counter() - started


def _canonical_panorama_detections(faces, scene, args, identifier):
    panorama = compose_equirectangular(
        faces,
        args.panorama_width,
        args.panorama_height,
        args.face_fov,
    )
    detections = detect_stars(
        cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY),
        threshold_sigma=args.star_threshold,
        minimum_elevation_degrees=args.min_elevation,
        max_candidates=args.max_candidates,
    )
    detections = identify_stars(
        detections,
        (args.panorama_width, args.panorama_height),
        identifier,
    )
    heading_rotation = Rotation.from_euler('z', scene['heading'], degrees=True)
    identified = []
    for detection in detections:
        object_id = str(detection.get('object_id', '')).strip()
        if not object_id or object_id.upper() == 'UNKNOWN':
            continue
        azimuth, elevation = pixel_to_az_el(
            detection['pixel_x'],
            detection['pixel_y'],
            args.panorama_width,
            args.panorama_height,
        )
        identified.append({
            **detection,
            'direction': heading_rotation.apply(
                _direction_from_azimuth_elevation(azimuth, elevation)
            ),
        })
    return identified, panorama


def _face_detections(faces, scene, args, face_size):
    heading_rotation = Rotation.from_euler('z', scene['heading'], degrees=True)
    detections = []
    for face in cube_faces():
        image = faces[face.name]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stars = detect_stars(
            gray,
            threshold_sigma=args.star_threshold,
            minimum_elevation_degrees=-90.0,
            max_candidates=args.max_candidates,
        )
        for star in stars:
            direction = face_pixel_to_direction(
                face,
                star['pixel_x'],
                star['pixel_y'],
                image.shape[1],
                image.shape[0],
                args.face_fov,
            )
            detections.append({
                'face_name': face.name,
                'direction': heading_rotation.apply(direction),
                'brightness': star['brightness'],
                'pixel_x': star['pixel_x'],
                'pixel_y': star['pixel_y'],
            })
    return detections


def _write_crop(image, pixel_x, pixel_y, path, radius=12):
    center_x = int(round(float(pixel_x)))
    center_y = int(round(float(pixel_y)))
    height, width = image.shape[:2]
    x_min = max(0, center_x - radius)
    x_max = min(width, center_x + radius + 1)
    y_min = max(0, center_y - radius)
    y_max = min(height, center_y + radius + 1)
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0 or not cv2.imwrite(str(path), crop):
        raise OSError(f'failed to write crop {path}')


def _solve_observations(observations, scene, ephemeris, args):
    if len(observations) < 3:
        return {
            'valid': False,
            'failure_reason': 'insufficient_observations',
            'latitude': None,
            'longitude': None,
            'heading': None,
            'geographic_error_meters': None,
            'rms_residual': None,
            'runtime_seconds': 0.0,
            'observation_count': len(observations),
        }
    started = time.perf_counter()
    result = solve(
        observations,
        scene['timestamp'],
        ephemeris,
        (args.initial_latitude, args.initial_longitude, args.initial_heading),
        max_nfev=FIXED_SOLVER['max_nfev'],
        global_search=FIXED_SOLVER['global_search'],
        max_starts=FIXED_SOLVER['max_starts'],
    )
    payload = {
        'valid': result.valid,
        'failure_reason': result.failure_reason,
        'latitude': float(result.x[0]),
        'longitude': float(result.x[1]),
        'heading': float(result.x[2]),
        'geographic_error_meters': _geographic_error({
            'latitude': float(result.x[0]),
            'longitude': float(result.x[1]),
        }, scene),
        'rms_residual': result.rms_residual,
        'runtime_seconds': time.perf_counter() - started,
        'observation_count': len(observations),
    }
    return payload


def compare_scene(scene, args, identifier, ephemeris, capture_dir=None):
    canonical_faces, canonical_render_seconds = _render_faces(scene, args, 512)
    canonical_ids, canonical_panorama = _canonical_panorama_detections(
        canonical_faces,
        scene,
        args,
        identifier,
    )
    canonical_id_set = {detection['object_id'] for detection in canonical_ids}
    capture_path = Path(capture_dir) if capture_dir else None
    if capture_path:
        capture_path.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(capture_path / 'canonical_panorama.png'), canonical_panorama):
            raise OSError(f'failed to write {capture_path / "canonical_panorama.png"}')
        canonical_faces_path = capture_path / 'canonical_faces'
        canonical_faces_path.mkdir(exist_ok=True)
        for face in cube_faces():
            if not cv2.imwrite(str(canonical_faces_path / f'{face.name}.png'), canonical_faces[face.name]):
                raise OSError(f'failed to write canonical face {face.name}')
    expected_by_id = {}
    for detection in canonical_ids:
        expected = ephemeris.predict(
            detection['object_id'],
            scene['timestamp'],
            scene['latitude'],
            scene['longitude'],
            scene['altitude'],
        )
        if expected is not None:
            expected_by_id[detection['object_id']] = _direction_from_azimuth_elevation(*expected)

    cases = []
    matched_by_size = {}
    for face_size in args.face_sizes:
        if face_size == 512:
            faces = canonical_faces
            render_seconds = canonical_render_seconds
        else:
            faces, render_seconds = _render_faces(scene, args, face_size)
        face_detections = _face_detections(faces, scene, args, face_size)
        matching = match_face_detections(
            canonical_ids,
            face_detections,
            (args.panorama_width, args.panorama_height),
            max_distance_degrees=args.face_match_radius,
            ambiguity_margin_degrees=args.face_match_ambiguity_margin,
            duplicate_radius_degrees=args.face_duplicate_radius,
        )
        matched_by_size[face_size] = matching['matches']
        matched_ids = set(matching['matches'])
        rows = []
        matched_star_records = []
        all_face_detections = []
        if capture_path:
            size_path = capture_path / f'face_{face_size}'
            size_path.mkdir(exist_ok=True)
            for face in cube_faces():
                if not cv2.imwrite(str(size_path / f'{face.name}.png'), faces[face.name]):
                    raise OSError(f'failed to write face {face.name} at size {face_size}')
        for detection in face_detections:
            all_face_detections.append({
                'face_name': detection['face_name'],
                'pixel_x': detection['pixel_x'],
                'pixel_y': detection['pixel_y'],
                'brightness': detection['brightness'],
                'direction': [float(value) for value in detection['direction']],
                'azimuth_deg': _azimuth_elevation_from_direction(detection['direction'])[0],
                'elevation_deg': _azimuth_elevation_from_direction(detection['direction'])[1],
            })
        for object_id, match in matching['matches'].items():
            detection = match['face_detection']
            record = {
                'object_id': object_id,
                'face_name': detection['face_name'],
                'pixel_x': detection['pixel_x'],
                'pixel_y': detection['pixel_y'],
                'brightness': detection['brightness'],
                'match_distance_degrees': match['angular_distance_degrees'],
                'face_direction': [float(value) for value in detection['direction']],
                'face_azimuth_deg': _azimuth_elevation_from_direction(detection['direction'])[0],
                'face_elevation_deg': _azimuth_elevation_from_direction(detection['direction'])[1],
                'canonical_direction': [
                    float(value) for value in next(
                        item['direction'] for item in canonical_ids
                        if item['object_id'] == object_id
                    )
                ],
            }
            if capture_path:
                crop_directory = capture_path / f'crops_{face_size}'
                crop_directory.mkdir(exist_ok=True)
                crop_path = crop_directory / f'{object_id}_{detection["face_name"]}.png'
                _write_crop(
                    faces[detection['face_name']],
                    detection['pixel_x'],
                    detection['pixel_y'],
                    crop_path,
                )
                record['crop_file'] = str(crop_path.relative_to(capture_path))
            matched_star_records.append(record)
            expected = expected_by_id.get(object_id)
            if expected is None:
                continue
            rows.append({
                'object_id': object_id,
                'angular_error_degrees': angular_separation_degrees(detection['direction'], expected),
            })
        cases.append({
            'face_size': face_size,
            'render_seconds': render_seconds,
            'canonical_panorama_id_count': len(canonical_id_set),
            'matched_id_count': len(matched_ids),
            'matched_ids': sorted(matched_ids),
            'dropped_canonical_ids': sorted(canonical_id_set - matched_ids),
            'ambiguous_canonical_ids': matching['ambiguous_object_ids'],
            'new_ids_relative_to_canonical': [],
            'angular_error_all_matched_ids_deg': _angular_summary(
                row['angular_error_degrees'] for row in rows
            ),
            'rows': rows,
            'all_face_detections': all_face_detections,
            'matched_star_records': matched_star_records,
        })

    common_ids = sorted(set.intersection(*(
        set(matched_by_size[face_size]) for face_size in args.face_sizes
    )))
    for case in cases:
        face_size = case['face_size']
        matches = matched_by_size[face_size]
        rows_by_id = {
            row['object_id']: row for row in case['rows']
        }
        common_rows = [rows_by_id[object_id] for object_id in common_ids]
        inverse_heading = Rotation.from_euler('z', -scene['heading'], degrees=True)
        observations = []
        for object_id in common_ids:
            local_direction = inverse_heading.apply(
                matches[object_id]['face_detection']['direction']
            )
            azimuth, elevation = _azimuth_elevation_from_direction(local_direction)
            observations.append({
                'object_id': object_id,
                'azimuth': azimuth,
                'elevation': elevation,
                'confidence': 1.0,
            })
        case['common_all_face_sizes_ids'] = common_ids
        case['common_all_face_sizes_count'] = len(common_ids)
        case['angular_error_common_all_sizes_deg'] = _angular_summary(
            row['angular_error_degrees'] for row in common_rows
        )
        case['solver_common_all_sizes'] = _solve_observations(
            observations,
            scene,
            ephemeris,
            args,
        )
    return {
        'scene': scene,
        'face_sizes': args.face_sizes,
        'canonical_id_count': len(canonical_id_set),
        'canonical_ids': sorted(canonical_id_set),
        'canonical_star_records': [
            {
                'object_id': detection['object_id'],
                'pixel_x': detection['pixel_x'],
                'pixel_y': detection['pixel_y'],
                'brightness': detection.get('brightness'),
                'direction': [float(value) for value in detection['direction']],
                'azimuth_deg': _azimuth_elevation_from_direction(detection['direction'])[0],
                'elevation_deg': _azimuth_elevation_from_direction(detection['direction'])[1],
            }
            for detection in canonical_ids
        ],
        'canonical_render_seconds': canonical_render_seconds,
        'common_all_face_sizes_ids': common_ids,
        'capture_directory': str(capture_path) if capture_path else None,
        'cases': cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--scene-id', nargs='+', required=True)
    parser.add_argument('--face-sizes', type=int, nargs='+', default=[512, 1024, 2048])
    parser.add_argument('--face-fov', type=float, default=95.0)
    parser.add_argument('--panorama-width', type=int, default=2048)
    parser.add_argument('--panorama-height', type=int, default=1024)
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
    parser.add_argument('--engine-js', default='/opt/stellarium/stellarium-web-engine.js')
    parser.add_argument('--engine-wasm', default='/opt/stellarium/stellarium-web-engine.wasm')
    parser.add_argument('--data-root', default='/opt/stellarium/data')
    parser.add_argument('--browser-executable', default='')
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path)
    args = parser.parse_args()

    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    scenes_by_id = {scene['scene_id']: scene for scene in scenes}
    missing = [scene_id for scene_id in args.scene_id if scene_id not in scenes_by_id]
    if missing:
        raise ValueError(f'scene IDs are not in the manifest: {missing}')
    if len(set(args.face_sizes)) != len(args.face_sizes):
        raise ValueError('face sizes must be unique')
    if any(face_size < 32 for face_size in args.face_sizes):
        raise ValueError('face sizes must be at least 32 pixels')

    identifier = StarIdentifier(tile_size=args.star_tile_size)
    ephemeris = EphemerisProvider()
    report = {
        'configuration': {
            'face_sizes': args.face_sizes,
            'solver': FIXED_SOLVER,
            'panorama_width': args.panorama_width,
            'panorama_height': args.panorama_height,
            'pairing_uses_ground_truth': False,
            'identification_source': '512-face composed panorama only',
        },
        'scenes': [],
    }
    for scene_id in args.scene_id:
        print(f'running {scene_id}', flush=True)
        report['scenes'].append(
            compare_scene(
                scenes_by_id[scene_id],
                args,
                identifier,
                ephemeris,
                capture_dir=(args.capture_dir / scene_id if args.capture_dir else None),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'scene_count': len(report['scenes']),
        'face_sizes': args.face_sizes,
        'common_all_face_sizes_counts': [
            len(scene['common_all_face_sizes_ids']) for scene in report['scenes']
        ],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
