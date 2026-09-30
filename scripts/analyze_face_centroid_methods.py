#!/usr/bin/env python3
"""Compare original-face centroid estimators on saved rendered captures."""

from __future__ import annotations

import argparse
import csv
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

METHODS = ('dao', 'weighted_radius_3', 'weighted_radius_5', 'quadratic_peak')
FIXED_SOLVER_SETTINGS = {
    'max_nfev': 60,
    'max_starts': 5,
    'global_search': True,
}
CSV_FIELDS = (
    'capture_id',
    'method',
    'object_id',
    'source_face',
    'pixel_x',
    'pixel_y',
    'normalized_x',
    'normalized_y',
    'brightness',
    'center_distance_normalized',
    'center_distance_degrees',
    'match_distance_degrees',
    'angular_error_degrees',
    'cross_track_error_degrees',
    'elevation_error_degrees',
)


def _direction_from_azimuth_elevation(azimuth_degrees, elevation_degrees):
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    cosine = math.cos(elevation)
    return np.asarray((
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ), dtype=np.float64)


def _wrapped_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _face_rotation(heading):
    return Rotation.from_euler('z', float(heading), degrees=True)


def _patch_bounds(gray, center_x, center_y, radius):
    height, width = gray.shape[:2]
    x_center = int(round(float(center_x)))
    y_center = int(round(float(center_y)))
    x_min = max(0, x_center - radius)
    x_max = min(width, x_center + radius + 1)
    y_min = max(0, y_center - radius)
    y_max = min(height, y_center + radius + 1)
    return x_min, x_max, y_min, y_max


def _weighted_centroid(gray, detection, radius):
    x_min, x_max, y_min, y_max = _patch_bounds(
        gray,
        detection['pixel_x'],
        detection['pixel_y'],
        radius,
    )
    patch = gray[y_min:y_max, x_min:x_max].astype(np.float64)
    if patch.size == 0:
        return detection['pixel_x'], detection['pixel_y']
    border = np.concatenate((
        patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1],
    ))
    weights = np.maximum(patch - float(np.median(border)), 0.0)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return detection['pixel_x'], detection['pixel_y']
    coordinates_x, coordinates_y = np.meshgrid(
        np.arange(x_min, x_max, dtype=np.float64),
        np.arange(y_min, y_max, dtype=np.float64),
    )
    return (
        float(np.sum(coordinates_x * weights) / total),
        float(np.sum(coordinates_y * weights) / total),
    )


def _quadratic_peak_centroid(gray, detection):
    x_min, x_max, y_min, y_max = _patch_bounds(gray, detection['pixel_x'], detection['pixel_y'], 2)
    patch = gray[y_min:y_max, x_min:x_max].astype(np.float64)
    if patch.size == 0:
        return detection['pixel_x'], detection['pixel_y']
    peak_y, peak_x = np.unravel_index(int(np.argmax(patch)), patch.shape)
    peak_x += x_min
    peak_y += y_min
    refined_x = float(peak_x)
    refined_y = float(peak_y)
    if 0 < peak_x < gray.shape[1] - 1:
        left, center, right = gray[peak_y, peak_x - 1:peak_x + 2].astype(np.float64)
        denominator = left - 2.0 * center + right
        if abs(float(denominator)) > 1e-12:
            refined_x += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    if 0 < peak_y < gray.shape[0] - 1:
        upper, center, lower = gray[peak_y - 1:peak_y + 2, peak_x].astype(np.float64)
        denominator = upper - 2.0 * center + lower
        if abs(float(denominator)) > 1e-12:
            refined_y += float(np.clip(0.5 * (upper - lower) / denominator, -0.5, 0.5))
    return refined_x, refined_y


def _centroid(gray, detection, method):
    if method == 'dao':
        return float(detection['pixel_x']), float(detection['pixel_y'])
    if method == 'weighted_radius_3':
        return _weighted_centroid(gray, detection, 3)
    if method == 'weighted_radius_5':
        return _weighted_centroid(gray, detection, 5)
    if method == 'quadratic_peak':
        return _quadratic_peak_centroid(gray, detection)
    raise ValueError(f'unknown centroid method: {method}')


def _capture_settings(capture):
    return capture['resolved_settings']['compare_configuration']


def _panorama_detections(capture):
    scene = capture['scene_definition']
    heading_rotation = _face_rotation(scene['heading'])
    detections = []
    for observation in capture['panorama_observations']:
        local_direction = _direction_from_azimuth_elevation(
            observation['azimuth'],
            observation['elevation'],
        )
        detections.append({
            **observation,
            'direction': heading_rotation.apply(local_direction),
        })
    return detections


def _face_detections(capture, method):
    scene = capture['scene_definition']
    settings = _capture_settings(capture)
    heading_rotation = _face_rotation(scene['heading'])
    detections = []
    for face in cube_faces():
        image_path = Path(capture['_capture_directory']) / capture['image_files'][face.name]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f'failed to read {image_path}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raw_detections = detect_stars(
            gray,
            threshold_sigma=float(settings.get('star_threshold', 5.0)),
            minimum_elevation_degrees=-90.0,
            max_candidates=int(settings.get('max_candidates', 64)),
        )
        for detection in raw_detections:
            pixel_x, pixel_y = _centroid(gray, detection, method)
            local_direction = face_pixel_to_direction(
                face,
                pixel_x,
                pixel_y,
                image.shape[1],
                image.shape[0],
                float(settings.get('face_fov', 95.0)),
            )
            detections.append({
                'face_name': face.name,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'brightness': float(detection['brightness']),
                'direction': heading_rotation.apply(local_direction),
                'image_width': image.shape[1],
                'image_height': image.shape[0],
            })
    return detections


def _component_errors(observed, expected):
    observed_azimuth = math.degrees(math.atan2(observed[1], observed[0])) % 360.0
    observed_elevation = math.degrees(math.asin(float(np.clip(observed[2], -1.0, 1.0))))
    expected_azimuth = math.degrees(math.atan2(expected[1], expected[0])) % 360.0
    expected_elevation = math.degrees(math.asin(float(np.clip(expected[2], -1.0, 1.0))))
    azimuth_error = _wrapped_degrees(observed_azimuth - expected_azimuth)
    return (
        azimuth_error * math.cos(math.radians(expected_elevation)),
        observed_elevation - expected_elevation,
    )


def _geographic_error_meters(latitude, longitude, scene):
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


def _summarize(values):
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if not len(values):
        return {'count': 0, 'median': None, 'p90': None, 'mean': None, 'max': None}
    return {
        'count': int(len(values)),
        'median': float(np.median(values)),
        'p90': float(np.percentile(values, 90.0)),
        'mean': float(np.mean(values)),
        'max': float(np.max(values)),
    }


def _correlation(rows, feature):
    features = np.asarray([row[feature] for row in rows], dtype=np.float64)
    errors = np.asarray([abs(row['angular_error_degrees']) for row in rows], dtype=np.float64)
    if len(features) < 2 or np.std(features) == 0.0 or np.std(errors) == 0.0:
        return None
    return float(np.corrcoef(features, errors)[0, 1])


def _quartile_summary(rows, feature):
    values = np.asarray([row[feature] for row in rows], dtype=np.float64)
    if not len(values):
        return []
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    summaries = []
    for index in range(4):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 3:
            selected = [row for row in rows if lower <= row[feature] <= upper]
        else:
            selected = [row for row in rows if lower <= row[feature] < upper]
        summaries.append({
            'quartile': index + 1,
            'lower': float(lower),
            'upper': float(upper),
            'angular_error_degrees': _summarize(
                row['angular_error_degrees'] for row in selected
            ),
        })
    return summaries


def analyze_capture(capture_path, provider):
    capture = json.loads(capture_path.read_text(encoding='utf-8'))
    capture['_capture_directory'] = str(capture_path.parent)
    scene = capture['scene_definition']
    panorama_detections = _panorama_detections(capture)
    expected_by_id = {}
    for detection in panorama_detections:
        expected = provider.predict(
            detection['object_id'],
            scene['timestamp'],
            scene['latitude'],
            scene['longitude'],
            scene['altitude'],
        )
        if expected is not None:
            expected_by_id[detection['object_id']] = _direction_from_azimuth_elevation(*expected)

    method_rows = {}
    method_summaries = {}
    for method in METHODS:
        matching = match_face_detections(
            panorama_detections,
            _face_detections(capture, method),
            (2048, 1024),
            max_distance_degrees=2.0,
            ambiguity_margin_degrees=0.25,
            duplicate_radius_degrees=0.15,
        )
        rows = []
        for object_id, match in matching['matches'].items():
            expected = expected_by_id.get(object_id)
            if expected is None:
                continue
            detection = match['face_detection']
            observed = detection['direction']
            cross_track, elevation = _component_errors(observed, expected)
            face_center = next(
                face for face in cube_faces() if face.name == detection['face_name']
            )
            face_center_direction = _face_rotation(scene['heading']).apply(face_center.center)
            center_distance = angular_separation_degrees(observed, face_center_direction)
            width = detection['image_width']
            height = detection['image_height']
            center_x = float(detection['pixel_x']) + 0.5
            center_y = float(detection['pixel_y']) + 0.5
            radial = math.sqrt(
                ((center_x - width / 2.0) / (width / 2.0)) ** 2
                + ((center_y - height / 2.0) / (height / 2.0)) ** 2
            ) / math.sqrt(2.0)
            row = {
                'capture_id': scene['scene_id'],
                'method': method,
                'object_id': object_id,
                'source_face': detection['face_name'],
                'pixel_x': detection['pixel_x'],
                'pixel_y': detection['pixel_y'],
                'normalized_x': center_x / width,
                'normalized_y': center_y / height,
                'brightness': detection['brightness'],
                'center_distance_normalized': radial,
                'center_distance_degrees': center_distance,
                'match_distance_degrees': match['angular_distance_degrees'],
                'angular_error_degrees': angular_separation_degrees(observed, expected),
                'cross_track_error_degrees': cross_track,
                'elevation_error_degrees': elevation,
            }
            rows.append(row)
        method_rows[method] = rows
        observations = []
        inverse_heading = _face_rotation(-scene['heading'])
        settings = _capture_settings(capture)
        face_size = int(settings.get('face_size', 512))
        face_fov = float(settings.get('face_fov', 95.0))
        for row in rows:
            face = next(face for face in cube_faces() if face.name == row['source_face'])
            local_direction = inverse_heading.apply(
                face_pixel_to_direction(
                    face,
                    row['pixel_x'],
                    row['pixel_y'],
                    face_size,
                    face_size,
                    face_fov,
                )
            )
            azimuth = math.degrees(math.atan2(local_direction[1], local_direction[0])) % 360.0
            elevation = math.degrees(
                math.asin(float(np.clip(local_direction[2], -1.0, 1.0)))
            )
            observations.append({
                'object_id': row['object_id'],
                'azimuth': azimuth,
                'elevation': elevation,
                'confidence': 1.0,
            })
        started = time.perf_counter()
        result = solve(
            observations,
            scene['timestamp'],
            provider,
            (51.5, -0.1, 0.0),
            max_nfev=FIXED_SOLVER_SETTINGS['max_nfev'],
            global_search=FIXED_SOLVER_SETTINGS['global_search'],
            max_starts=FIXED_SOLVER_SETTINGS['max_starts'],
        )
        method_summaries[method] = {
            'angular_error_degrees': _summarize(row['angular_error_degrees'] for row in rows),
            'cross_track_error_degrees': _summarize(abs(row['cross_track_error_degrees']) for row in rows),
            'elevation_error_degrees': _summarize(abs(row['elevation_error_degrees']) for row in rows),
            'correlation_abs_error_with_brightness': _correlation(rows, 'brightness'),
            'correlation_abs_error_with_center_distance': _correlation(rows, 'center_distance_degrees'),
            'correlation_abs_error_with_normalized_x': _correlation(rows, 'normalized_x'),
            'correlation_abs_error_with_normalized_y': _correlation(rows, 'normalized_y'),
            'brightness_quartiles': _quartile_summary(rows, 'brightness'),
            'center_distance_quartiles': _quartile_summary(rows, 'center_distance_degrees'),
            'fixed_solver': {
                **FIXED_SOLVER_SETTINGS,
                'valid': result.valid,
                'failure_reason': result.failure_reason,
                'latitude': float(result.x[0]),
                'longitude': float(result.x[1]),
                'heading': float(result.x[2]),
                'geographic_error_meters': _geographic_error_meters(
                    float(result.x[0]),
                    float(result.x[1]),
                    scene,
                ),
                'rms_residual': result.rms_residual,
                'runtime_seconds': time.perf_counter() - started,
                'observation_count': len(observations),
            },
        }
    return method_rows, method_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--captures',
        type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output' / 'face_failure_captures_20260929',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output' / 'face_centroid_methods_20260929.json',
    )
    parser.add_argument(
        '--csv-output',
        type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output' / 'face_centroid_methods_20260929.csv',
    )
    args = parser.parse_args()
    provider = EphemerisProvider()
    all_rows = []
    captures = {}
    for capture_path in sorted(args.captures.glob('*/capture.json')):
        method_rows, method_summaries = analyze_capture(capture_path, provider)
        captures[capture_path.parent.name] = method_summaries
        for rows in method_rows.values():
            all_rows.extend(rows)
    output = {
        'captures': sorted(captures),
        'methods': list(METHODS),
        'pairing': {
            'uses_ground_truth_for_pairing': False,
            'method': 'mutual nearest face centroid to observed panorama directions',
        },
        'analysis': captures,
        'overall': {
            method: {
                'angular_error_degrees': _summarize(
                    row['angular_error_degrees']
                    for row in all_rows if row['method'] == method
                ),
                'cross_track_error_degrees': _summarize(
                    abs(row['cross_track_error_degrees'])
                    for row in all_rows if row['method'] == method
                ),
                'elevation_error_degrees': _summarize(
                    abs(row['elevation_error_degrees'])
                    for row in all_rows if row['method'] == method
                ),
                'fixed_solver': {
                    'valid_count': sum(
                        capture[method]['fixed_solver']['valid']
                        for capture in captures.values()
                    ),
                    'capture_count': len(captures),
                    'geographic_error_meters': _summarize(
                        capture[method]['fixed_solver']['geographic_error_meters']
                        for capture in captures.values()
                    ),
                    'runtime_seconds': _summarize(
                        capture[method]['fixed_solver']['runtime_seconds']
                        for capture in captures.values()
                    ),
                },
            }
            for method in METHODS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({
        'captures': len(captures),
        'rows': len(all_rows),
        'output': str(args.output),
        'csv_output': str(args.csv_output),
        'overall': output['overall'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
