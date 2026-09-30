#!/usr/bin/env python3
"""Compare original-face resolutions with fixed IDs and solver settings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_face_and_panorama import (  # noqa: E402
    EphemerisProvider,
    StarIdentifier,
    build_parser as build_scene_parser,
    create_bridge,
    run as run_scene,
)
from celestial_localiser.pose_solver import solve  # noqa: E402


FIXED_SOLVER = {
    'max_nfev': 60,
    'max_starts': 5,
    'global_search': True,
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


def _solve_same_ids(result, object_ids, scene, ephemeris):
    if not object_ids:
        return {
            'valid': False,
            'failure_reason': 'no_common_ids',
            'latitude': None,
            'longitude': None,
            'heading': None,
            'geographic_error_meters': None,
            'rms_residual': None,
            'max_residual': None,
            'runtime_seconds': 0.0,
            'observation_count': 0,
        }
    observations_by_id = {
        observation['object_id']: observation
        for observation in result['paired_solver_results']['face']['observations']
    }
    observations = [observations_by_id[object_id] for object_id in object_ids]
    started = time.perf_counter()
    solve_result = solve(
        observations,
        scene['timestamp'],
        ephemeris,
        (
            result['solver_configuration']['initial_latitude'],
            result['solver_configuration']['initial_longitude'],
            result['solver_configuration']['initial_heading'],
        ),
        max_nfev=FIXED_SOLVER['max_nfev'],
        global_search=FIXED_SOLVER['global_search'],
        max_starts=FIXED_SOLVER['max_starts'],
    )
    payload = {
        'valid': solve_result.valid,
        'failure_reason': solve_result.failure_reason,
        'latitude': float(solve_result.x[0]),
        'longitude': float(solve_result.x[1]),
        'heading': float(solve_result.x[2]),
        'geographic_error_meters': _geographic_error({
            'latitude': float(solve_result.x[0]),
            'longitude': float(solve_result.x[1]),
        }, scene),
        'rms_residual': solve_result.rms_residual,
        'max_residual': solve_result.max_residual,
        'runtime_seconds': time.perf_counter() - started,
        'observation_count': len(observations),
    }
    return payload


def _scene_args(args, scene, face_size):
    values = vars(args).copy()
    values.pop('output', None)
    values.pop('scene_id', None)
    values['face_size'] = face_size
    values['panorama_width'] = 2048
    values['panorama_height'] = 1024
    values.update(scene)
    return argparse.Namespace(**values)


def compare_scene(scene, face_sizes, args, identifier, ephemeris):
    results = {}
    for face_size in face_sizes:
        scene_args = _scene_args(args, scene, face_size)
        bridge = create_bridge(scene_args)
        try:
            results[str(face_size)] = run_scene(
                scene_args,
                bridge=bridge,
                identifier=identifier,
                ephemeris=ephemeris,
            )
        finally:
            bridge.close()

    common_ids = sorted(set.intersection(*(
        set(result['star_ids']['common']) for result in results.values()
    )))
    baseline_size = str(face_sizes[0])
    baseline_ids = set(results[baseline_size]['star_ids']['common'])
    baseline_rows_by_id = {
        row['object_id']: row for row in results[baseline_size]['rows']
    }
    cases = []
    for face_size in face_sizes:
        label = str(face_size)
        result = results[label]
        rows_by_id = {row['object_id']: row for row in result['rows']}
        common_rows = [rows_by_id[object_id] for object_id in common_ids]
        common_with_512_ids = sorted(
            baseline_ids & set(result['star_ids']['common'])
        )
        common_with_512_rows = [
            rows_by_id[object_id] for object_id in common_with_512_ids
        ]
        same_id_solver = _solve_same_ids(result, common_ids, scene, ephemeris)
        same_512_solver = _solve_same_ids(
            result,
            common_with_512_ids,
            scene,
            ephemeris,
        )
        cases.append({
            'face_size': face_size,
            'counts': result['counts'],
            'star_ids': result['star_ids'],
            'dropped_from_512_common_ids': sorted(
                baseline_ids - set(result['star_ids']['common'])
            ),
            'new_relative_to_512_common_ids': sorted(
                set(result['star_ids']['common']) - baseline_ids
            ),
            'common_all_face_sizes_count': len(common_ids),
            'common_with_512_ids': common_with_512_ids,
            'common_with_512_count': len(common_with_512_ids),
            'angular_error_all_ids_deg': result['face_errors_deg'],
            'angular_error_same_ids_deg': _summary(
                row['face_error_deg'] for row in common_rows
            ),
            'angular_error_with_512_ids_deg': _summary(
                row['face_error_deg'] for row in common_with_512_rows
            ),
            'baseline_512_angular_error_with_same_ids_deg': _summary(
                baseline_rows_by_id[object_id]['face_error_deg']
                for object_id in common_with_512_ids
            ),
            'geographic_error_all_ids_meters': result['paired_solver_results']['face']['geographic_error_meters'],
            'geographic_error_same_ids_meters': same_id_solver['geographic_error_meters'],
            'geographic_error_with_512_ids_meters': same_512_solver['geographic_error_meters'],
            'same_id_solver': same_id_solver,
            'same_512_id_solver': same_512_solver,
            'runtime': {
                'render_seconds': result['timing']['render_seconds'],
                'face_path_seconds': result['timing']['face_path_seconds'],
                'face_solver_seconds': result['paired_solver_results']['face']['runtime_seconds'],
            },
        })

    return {
        'scene': scene,
        'face_sizes': face_sizes,
        'common_all_face_sizes_ids': common_ids,
        'cases': cases,
    }


def main():
    parser = build_scene_parser()
    parser.description = __doc__
    parser.add_argument(
        '--scenes',
        type=Path,
        default=ROOT / 'scripts' / 'fixed_face_panorama_scenes.json',
    )
    parser.add_argument(
        '--scene-id',
        nargs='+',
        default=['london_00_h000', 'new_york_00_h045', 'quito_00_h030'],
    )
    parser.add_argument('--face-sizes', type=int, nargs='+', default=[512, 1024, 2048])
    parser.add_argument('--output', type=Path, required=True)
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
            'panorama_width': 2048,
            'panorama_height': 1024,
            'pairing_uses_ground_truth': False,
        },
        'scenes': [],
    }
    for scene_id in args.scene_id:
        print(f'running {scene_id}', flush=True)
        report['scenes'].append(
            compare_scene(
                scenes_by_id[scene_id],
                args.face_sizes,
                args,
                identifier,
                ephemeris,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'output': str(args.output),
        'scene_count': len(report['scenes']),
        'face_sizes': args.face_sizes,
        'all_size_common_id_counts': [
            len(scene['common_all_face_sizes_ids']) for scene in report['scenes']
        ],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()