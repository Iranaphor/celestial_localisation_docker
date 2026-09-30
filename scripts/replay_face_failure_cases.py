#!/usr/bin/env python3
"""Replay saved original-face captures without rendering new scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'celestial_localisation' / 'src' / 'celestial_localiser'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_pose_solver import CountingEphemeris  # noqa: E402
from celestial_localiser.ephemeris import EphemerisProvider  # noqa: E402
from celestial_localiser.pose_solver import solve  # noqa: E402


def geographic_error_meters(latitude, longitude, true_latitude, true_longitude):
    latitude_delta = math.radians(latitude - true_latitude)
    longitude_delta = math.radians(
        (longitude - true_longitude + 180.0) % 360.0 - 180.0
    )
    haversine_a = (
        math.sin(latitude_delta / 2.0) ** 2
        + math.cos(math.radians(true_latitude))
        * math.cos(math.radians(latitude))
        * math.sin(longitude_delta / 2.0) ** 2
    )
    return 2.0 * 6_371_000.0 * math.atan2(
        math.sqrt(haversine_a),
        math.sqrt(max(0.0, 1.0 - haversine_a)),
    )


def _annotate_start_diagnostics(diagnostics, scene):
    annotated = []
    for diagnostic in diagnostics:
        item = dict(diagnostic)
        final_state = item.get('final_state')
        if final_state is not None and all(value is not None for value in final_state):
            item['geographic_error_meters'] = geographic_error_meters(
                final_state[0],
                final_state[1],
                scene['latitude'],
                scene['longitude'],
            )
        else:
            item['geographic_error_meters'] = None
        annotated.append(item)
    return annotated


def _heading_delta(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _start_analysis(diagnostics):
    finite = [
        (index, diagnostic)
        for index, diagnostic in enumerate(diagnostics)
        if diagnostic.get('final_state') is not None
        and all(value is not None for value in diagnostic['final_state'])
        and diagnostic.get('cost') is not None
    ]
    groups = []
    for index, diagnostic in finite:
        state = diagnostic['final_state']
        group = next(
            (
                candidate for candidate in groups
                if abs(state[0] - candidate['representative_final_state'][0]) <= 0.01
                and abs(state[1] - candidate['representative_final_state'][1]) <= 0.01
                and _heading_delta(
                    state[2], candidate['representative_final_state'][2]
                ) <= 0.01
            ),
            None,
        )
        if group is None:
            group = {
                'representative_final_state': list(state),
                'start_indices': [],
                'costs': [],
                'geographic_errors_meters': [],
            }
            groups.append(group)
        group['start_indices'].append(index)
        group['costs'].append(diagnostic['cost'])
        group['geographic_errors_meters'].append(
            diagnostic.get('geographic_error_meters')
        )

    successful = [
        item for item in finite if item[1].get('success', False)
    ]

    def describe(item):
        if item is None:
            return None
        index, diagnostic = item
        return {
            'start_index': index,
            'cost': diagnostic['cost'],
            'final_state': diagnostic['final_state'],
            'geographic_error_meters': diagnostic.get(
                'geographic_error_meters'
            ),
            'success': diagnostic.get('success', False),
            'selected': diagnostic.get('selected', False),
        }

    lowest_cost_finite = min(finite, key=lambda item: item[1]['cost']) if finite else None
    lowest_cost_successful = (
        min(successful, key=lambda item: item[1]['cost'])
        if successful else None
    )
    selected = next(
        (item for item in finite if item[1].get('selected', False)),
        None,
    )
    return {
        'attempted_start_count': len(diagnostics),
        'finite_start_count': len(finite),
        'converged_start_count': sum(
            diagnostic.get('success', False)
            for _, diagnostic in finite
        ),
        'distinct_solution_count': len(groups),
        'solution_groups': groups,
        'lowest_cost_finite_start': describe(lowest_cost_finite),
        'lowest_cost_successful_start': describe(lowest_cost_successful),
        'selected_start': describe(selected),
    }


def _replay_case(case, scene, provider, args):
    observations = case['face_observations']
    initial_state = tuple(float(value) for value in case['initial_state'])
    solver_configuration = case.get('solver_configuration', {})
    robust_loss = solver_configuration.get('robust_loss', 'soft_l1')
    min_observations = int(solver_configuration.get('min_observations', 3))
    configurations = []
    for max_starts in args.max_starts:
        for max_nfev in args.max_nfev:
            ephemeris = CountingEphemeris(provider)
            started = time.perf_counter()
            result = solve(
                observations,
                float(case['timestamp']),
                ephemeris,
                initial_state,
                robust_loss=robust_loss,
                max_nfev=max_nfev,
                min_observations=min_observations,
                global_search=args.global_search,
                max_starts=max_starts,
            )
            elapsed = time.perf_counter() - started
            selected_geographic_error = None
            if result.start_diagnostics and all(value is not None for value in result.x):
                selected_geographic_error = geographic_error_meters(
                    result.x[0],
                    result.x[1],
                    scene['latitude'],
                    scene['longitude'],
                )
            annotated_diagnostics = _annotate_start_diagnostics(
                result.start_diagnostics,
                scene,
            )
            configurations.append({
                'max_nfev': max_nfev,
                'max_starts': max_starts,
                'global_search': args.global_search,
                'robust_loss': robust_loss,
                'min_observations': min_observations,
                'valid': result.valid,
                'failure_reason': result.failure_reason,
                'latitude': float(result.x[0]),
                'longitude': float(result.x[1]),
                'heading': float(result.x[2]),
                'geographic_error_meters': selected_geographic_error,
                'cost': result.cost,
                'rms_residual': result.rms_residual,
                'max_residual': result.max_residual,
                'selected_nfev': result.nfev,
                'total_nfev': result.total_nfev,
                'starts_tried': result.starts_tried,
                'start_diagnostics': annotated_diagnostics,
                'start_analysis': _start_analysis(annotated_diagnostics),
                'ephemeris_calls': ephemeris.calls,
                'ephemeris_batch_calls': ephemeris.batch_calls,
                'ephemeris_seconds': ephemeris.seconds,
                'runtime_seconds': elapsed,
                'message': result.message,
            })
    return configurations


def _load_capture_cases(captures_path):
    manifest_path = captures_path / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'capture manifest is missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cases = []
    for manifest_scene in manifest.get('scenes', []):
        capture_path = captures_path / manifest_scene['capture_file']
        capture = json.loads(capture_path.read_text(encoding='utf-8'))
        scene = capture['scene_definition']
        solver_configuration = capture['resolved_settings']['solver_configuration']
        cases.append({
            'scene_id': scene['scene_id'],
            'capture_type': capture['capture_type'],
            'historical_reference': capture.get('historical_reference', {}),
            'scene': scene,
            'face_observations': capture['face_observations'],
            'timestamp': scene['timestamp'],
            'initial_state': [
                solver_configuration['initial_latitude'],
                solver_configuration['initial_longitude'],
                solver_configuration['initial_heading'],
            ],
            'solver_configuration': solver_configuration,
        })
    return cases


def run(args):
    if args.captures:
        replay_cases = _load_capture_cases(args.captures)
        scene_rows = {case['scene_id']: case['scene'] for case in replay_cases}
        capture_type = 'new_capture_of_same_scene_definition'
    else:
        campaign = json.loads(args.campaign.read_text(encoding='utf-8'))
        replay_cases = campaign.get('replay_cases')
        if not replay_cases:
            raise ValueError(
                'campaign artifact has no replay_cases; exact face-failure replay is '
                'unavailable because the historical campaign did not persist solver observations'
            )
        scene_rows = {
            row['scene_id']: row
            for row in campaign.get('scenes', [])
        }
        replay_cases = [
            case for case in replay_cases
            if scene_rows.get(case['scene_id'], {}).get('face_failure_reason')
        ]
        capture_type = 'campaign_replay_of_recorded_observations'
    provider = EphemerisProvider(args.star_database_path)
    output_cases = []
    for case in replay_cases:
        scene = scene_rows[case['scene_id']]
        configurations = _replay_case(case, scene, provider, args)
        output_cases.append({
            'scene_id': case['scene_id'],
            'capture_type': case.get('capture_type', capture_type),
            'historical_reference': case.get('historical_reference', {}),
            'original_failure_reason': scene.get('face_failure_reason'),
            'observations': len(case['face_observations']),
            'scene': {
                'latitude': scene['latitude'],
                'longitude': scene['longitude'],
                'altitude': scene['altitude'],
                'timestamp': scene['timestamp'],
                'heading': scene['heading'],
            },
            'initial_state': case['initial_state'],
            'configurations': configurations,
        })
    return {
        'source_campaign': args.campaign.name if args.campaign else None,
        'source_captures': args.captures.name if args.captures else None,
        'capture_type': capture_type,
        'failure_case_count': len(output_cases),
        'max_nfev': args.max_nfev,
        'max_starts': args.max_starts,
        'global_search': args.global_search,
        'cases': output_cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--campaign', type=Path)
    source.add_argument('--captures', type=Path)
    parser.add_argument('--star-database-path', default='')
    parser.add_argument('--max-nfev', type=int, nargs='+', default=[30, 60, 120, 240])
    parser.add_argument('--max-starts', type=int, nargs='+', default=[5, 13, 25])
    parser.add_argument('--global-search', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
