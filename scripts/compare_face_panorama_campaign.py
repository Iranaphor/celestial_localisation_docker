#!/usr/bin/env python3
"""Run a controlled paired original-face versus panorama campaign."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_face_and_panorama import (  # noqa: E402
    StarIdentifier,
    EphemerisProvider,
    build_parser as build_scene_parser,
    create_bridge,
    run as run_scene,
)


CSV_FIELDS = (
    'scene_id',
    'latitude',
    'longitude',
    'altitude',
    'timestamp',
    'heading',
    'panorama_identified',
    'face_detections',
    'face_matches',
    'face_stars_lost_during_matching',
    'face_stars_rejected_as_ambiguous',
    'common_star_ids',
    'lost_star_ids',
    'ambiguous_star_ids',
    'common_ids',
    'face_valid',
    'face_failure_reason',
    'face_geographic_error_meters',
    'face_observation_error_median_deg',
    'face_observation_error_p90_deg',
    'face_solver_rms_residual',
    'face_solver_runtime_seconds',
    'face_total_runtime_seconds',
    'face_path_runtime_seconds',
    'panorama_valid',
    'panorama_failure_reason',
    'panorama_geographic_error_meters',
    'panorama_observation_error_median_deg',
    'panorama_observation_error_p90_deg',
    'panorama_solver_rms_residual',
    'panorama_solver_runtime_seconds',
    'panorama_total_runtime_seconds',
    'panorama_path_runtime_seconds',
    'scene_runtime_seconds',
    'scene_error',
)


def _summary(values):
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {
            'count': 0,
            'median': None,
            'p90': None,
            'mean': None,
            'min': None,
            'max': None,
        }
    array = np.asarray(numeric, dtype=np.float64)
    return {
        'count': len(numeric),
        'median': float(np.median(array)),
        'p90': float(np.percentile(array, 90.0)),
        'mean': float(np.mean(array)),
        'min': float(np.min(array)),
        'max': float(np.max(array)),
    }


def _solver_row(scene, result, representation):
    solver = result['paired_solver_results'][representation]
    errors = result[f'{representation}_errors_deg']
    return {
        'scene_id': scene['scene_id'],
        'latitude': scene['latitude'],
        'longitude': scene['longitude'],
        'altitude': scene['altitude'],
        'timestamp': scene['timestamp'],
        'heading': scene['heading'],
        'panorama_identified': result['counts']['panorama_identified'],
        'face_detections': result['counts']['face_detections'],
        'face_matches': result['counts']['face_matched_to_panorama_ids'],
        'face_stars_lost_during_matching': result['counts']['face_stars_lost_during_matching'],
        'face_stars_rejected_as_ambiguous': result['counts']['face_stars_rejected_as_ambiguous'],
        'common_star_ids': json.dumps(sorted(row['object_id'] for row in result['rows'])),
        'lost_star_ids': json.dumps(result['pairing']['lost_object_ids']),
        'ambiguous_star_ids': json.dumps(result['pairing']['ambiguous_object_ids']),
        'common_ids': result['counts']['common_ids'],
        f'{representation}_valid': bool(solver['valid']),
        f'{representation}_failure_reason': solver['failure_reason'],
        f'{representation}_geographic_error_meters': solver['geographic_error_meters'],
        f'{representation}_observation_error_median_deg': errors['median_deg'],
        f'{representation}_observation_error_p90_deg': errors['p90_deg'],
        f'{representation}_solver_rms_residual': solver['rms_residual'],
        f'{representation}_solver_runtime_seconds': solver['runtime_seconds'],
        f'{representation}_total_runtime_seconds': solver['total_runtime_seconds'],
        f'{representation}_path_runtime_seconds': result['timing'][
            f'{representation}_path_seconds'
        ],
        'scene_runtime_seconds': result['timing']['scene_total_seconds'],
        'scene_error': '',
    }


def _failed_row(scene, error, elapsed_seconds):
    row = {field: '' for field in CSV_FIELDS}
    row.update({
        'scene_id': scene['scene_id'],
        'latitude': scene['latitude'],
        'longitude': scene['longitude'],
        'altitude': scene['altitude'],
        'timestamp': scene['timestamp'],
        'heading': scene['heading'],
        'scene_runtime_seconds': elapsed_seconds,
        'scene_error': str(error),
        'face_valid': False,
        'panorama_valid': False,
        'face_failure_reason': 'scene_error',
        'panorama_failure_reason': 'scene_error',
    })
    return row


def _campaign_report(scenes, results, rows, args, elapsed_seconds):
    scene_by_id = {scene['scene_id']: scene for scene in scenes}
    face_results = [
        result['paired_solver_results']['face']
        for result in results
        if result is not None
    ]
    panorama_results = [
        result['paired_solver_results']['panorama']
        for result in results
        if result is not None
    ]
    face_valid = [result for result in face_results if result['valid']]
    panorama_valid = [result for result in panorama_results if result['valid']]
    paired_valid = [
        result for result in results
        if result is not None
        and result['paired_solver_results']['face']['valid']
        and result['paired_solver_results']['panorama']['valid']
    ]

    face_errors = []
    panorama_errors = []
    for result in results:
        if result is None:
            continue
        face_errors.extend(
            row['face_error_deg'] for row in result['rows']
        )
        panorama_errors.extend(
            row['panorama_error_deg'] for row in result['rows']
        )

    def representation_summary(representation, valid_results):
        return {
            'rendered_scene_count': len(results),
            'valid_solve_count': len(valid_results),
            'failure_count': len(results) - len(valid_results),
            'failure_reasons': dict(Counter(
                result['failure_reason'] or 'valid'
                for result in (
                    item['paired_solver_results'][representation]
                    for item in results
                )
            )),
            'geographic_error_meters': _summary(
                result['geographic_error_meters'] for result in valid_results
            ),
            'observation_angular_error_degrees': _summary(
                face_errors if representation == 'face' else panorama_errors
            ),
            'solver_rms_vector_residual': _summary(
                result['rms_residual'] for result in valid_results
            ),
            'solver_runtime_seconds': _summary(
                result['runtime_seconds'] for result in valid_results
            ),
            'direction_plus_solver_runtime_seconds': _summary(
                result['total_runtime_seconds'] for result in valid_results
            ),
            'end_to_end_path_runtime_seconds': _summary(
                item['timing'][f'{representation}_path_seconds']
                for item in results
            ),
        }

    paired_face_errors = [
        result['paired_solver_results']['face']['geographic_error_meters']
        for result in paired_valid
    ]
    paired_panorama_errors = [
        result['paired_solver_results']['panorama']['geographic_error_meters']
        for result in paired_valid
    ]
    paired_improvements = [
        panorama - face
        for face, panorama in zip(paired_face_errors, paired_panorama_errors)
    ]

    def paired_geographic_summary(representation):
        return _summary(
            result['paired_solver_results'][representation]['geographic_error_meters']
            for result in paired_valid
        )

    lost_counts = [
        row['face_stars_lost_during_matching']
        for row in rows
        if row.get('scene_error', '') == ''
    ]
    ambiguous_counts = [
        row['face_stars_rejected_as_ambiguous']
        for row in rows
        if row.get('scene_error', '') == ''
    ]

    return {
        'manifest': Path(args.scenes).name,
        'scene_count_requested': len(scenes),
        'scene_count_completed': len(results),
        'scene_count_failed_to_render': sum(
            1 for row in rows if row.get('scene_error', '')
        ),
        'elapsed_seconds': elapsed_seconds,
        'configuration': {
            key: value for key, value in vars(args).items()
            if key not in {'output', 'csv_output'}
        },
        'pairing': {
            'method': 'mutual nearest original-face centroid to observed panorama direction',
            'uses_ground_truth_for_pairing': False,
            'star_ids_and_solver_inputs_paired': True,
            'ambiguity_rejection_enabled': True,
        },
        'panorama_baseline': representation_summary('panorama', panorama_valid),
        'original_face': representation_summary('face', face_valid),
        'paired_geographic_error_meters': {
            'face': paired_geographic_summary('face'),
            'panorama': paired_geographic_summary('panorama'),
        },
        'paired_geographic_improvement_meters': _summary(paired_improvements),
        'paired_scene_count_for_improvement': len(paired_valid),
        'replay_cases': [
            {
                'scene_id': result['scene_id'],
                'capture_type': 'campaign_observation_archive',
                'scene_definition': scene,
                'latitude': scene['latitude'],
                'longitude': scene['longitude'],
                'altitude': scene['altitude'],
                'timestamp': scene['timestamp'],
                'heading': scene['heading'],
                'initial_state': [
                    result['configuration']['initial_latitude'],
                    result['configuration']['initial_longitude'],
                    result['configuration']['initial_heading'],
                ],
                'solver_configuration': result['solver_configuration'],
                'face_observations': result['paired_solver_results']['face']['observations'],
                'panorama_observations': result['paired_solver_results']['panorama']['observations'],
            }
            for result in results
            for scene in (scene_by_id[result['scene_id']],)
        ],
        'stars_lost_during_matching': {
            'per_scene_count': _summary(lost_counts),
            'total': int(sum(lost_counts)),
            'ambiguous_per_scene_count': _summary(ambiguous_counts),
            'ambiguous_total': int(sum(ambiguous_counts)),
        },
        'shared_render_timing': _summary(
            result['timing']['render_seconds'] for result in results
        ),
        'scenes': rows,
    }


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = build_scene_parser()
    parser.description = __doc__
    parser.add_argument(
        '--scenes',
        default=str(Path(__file__).with_name('fixed_face_panorama_scenes.json')),
    )
    parser.add_argument('--output', default='face_panorama_campaign.json')
    parser.add_argument('--csv-output', default='face_panorama_campaign.csv')
    args = parser.parse_args()

    scenes_path = Path(args.scenes)
    with scenes_path.open('r', encoding='utf-8') as stream:
        scenes = json.load(stream)
    if not 20 <= len(scenes) <= 30:
        raise ValueError('the fixed-scene campaign must contain between 20 and 30 scenes')
    if len({scene['scene_id'] for scene in scenes}) != len(scenes):
        raise ValueError('fixed-scene scene_id values must be unique')

    bridge = create_bridge(args)
    identifier = StarIdentifier(tile_size=args.star_tile_size)
    ephemeris = EphemerisProvider()
    results = []
    rows = []
    campaign_started = time.perf_counter()
    try:
        for index, scene in enumerate(scenes, start=1):
            scene_args = argparse.Namespace(**vars(args))
            for key, value in scene.items():
                setattr(scene_args, key, value)
            scene_started = time.perf_counter()
            try:
                result = run_scene(
                    scene_args,
                    bridge=bridge,
                    identifier=identifier,
                    ephemeris=ephemeris,
                )
                result['scene_id'] = scene['scene_id']
                results.append(result)
                rows.append(_solver_row(scene, result, 'face') | {
                    key: value for key, value in _solver_row(scene, result, 'panorama').items()
                    if key.startswith('panorama_') or key == 'scene_error'
                })
                print(
                    f"[{index}/{len(scenes)}] {scene['scene_id']}: "
                    f"common={result['counts']['common_ids']} "
                    f"face={result['paired_solver_results']['face']['valid']} "
                    f"panorama={result['paired_solver_results']['panorama']['valid']}",
                    flush=True,
                )
            except Exception as error:
                elapsed = time.perf_counter() - scene_started
                rows.append(_failed_row(scene, error, elapsed))
                print(
                    f"[{index}/{len(scenes)}] {scene['scene_id']}: scene error: {error}",
                    flush=True,
                )
                bridge.close()
                bridge = create_bridge(args)
    finally:
        bridge.close()

    elapsed_seconds = time.perf_counter() - campaign_started
    report = _campaign_report(scenes, results, rows, args, elapsed_seconds)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    _write_csv(Path(args.csv_output), rows)
    print(json.dumps({
        'output': str(output_path),
        'csv_output': str(args.csv_output),
        'scene_count_completed': report['scene_count_completed'],
        'scene_count_failed_to_render': report['scene_count_failed_to_render'],
        'paired_scene_count_for_improvement': report['paired_scene_count_for_improvement'],
        'face_median_error_meters': report['original_face']['geographic_error_meters']['median'],
        'panorama_median_error_meters': report['panorama_baseline']['geographic_error_meters']['median'],
        'lost_star_total': report['stars_lost_during_matching']['total'],
        'ambiguous_star_total': report['stars_lost_during_matching']['ambiguous_total'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
