#!/usr/bin/env python3
"""Summarize a saved independent-face campaign without rerunning identification."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


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


def _error_rates(errors, attempt_count):
    errors = [float(value) for value in errors if value is not None]
    result = {
        'valid_count': len(errors),
        'attempt_count': attempt_count,
        'valid': {},
        'all_attempts': {},
    }
    for limit_km in (1, 5, 10):
        limit_meters = limit_km * 1000.0
        valid_count = sum(error <= limit_meters for error in errors)
        result['valid'][f'le_{limit_km}km'] = {
            'count': valid_count,
            'rate': valid_count / len(errors) if errors else None,
        }
        result['all_attempts'][f'le_{limit_km}km'] = {
            'count': valid_count,
            'rate': valid_count / attempt_count if attempt_count else None,
        }
    return result


def _capture_rows(captures_directory, scene_id):
    path = captures_directory / scene_id / 'capture.json'
    if not path.exists():
        return {}
    capture = json.loads(path.read_text(encoding='utf-8'))
    return {
        row['object_id']: row
        for row in capture.get('rows', [])
    }


def _changed_correspondences(result, capture_rows):
    changed = []
    for row in result['selected_identity_comparison']['rows']:
        if row['same_source_as_saved_baseline']:
            continue
        saved = capture_rows.get(row['object_id'], {})
        changed.append({
            'object_id': row['object_id'],
            'independent_face': row['face_name'],
            'independent_pixel_x': row['pixel_x'],
            'independent_pixel_y': row['pixel_y'],
            'match_count': row['match_count'],
            'match_probability': row['match_probability'],
            'saved_baseline_face': saved.get('source_face'),
            'saved_baseline_face_error_deg': saved.get('face_error_deg'),
            'saved_baseline_panorama_error_deg': saved.get('panorama_error_deg'),
            'independent_truth_error_deg': (
                row['truth_error']['angular_error_deg']
                if row['truth_error'] else None
            ),
        })
    return changed


def _rerun_changed_correspondences(result):
    changed = []
    for row in result['selected_identity_comparison']['rows']:
        if row['same_source_as_rerun_panorama_match']:
            continue
        changed.append({
            'object_id': row['object_id'],
            'independent_face': row['face_name'],
            'independent_pixel_x': row['pixel_x'],
            'independent_pixel_y': row['pixel_y'],
            'match_count': row['match_count'],
            'match_probability': row['match_probability'],
            'truth_error_deg': (
                row['truth_error']['angular_error_deg']
                if row['truth_error'] else None
            ),
        })
    return changed


def _scene_row(result, capture_rows):
    baseline = result['solver_variants']['panorama_transferred']
    hybrid = result['solver_variants']['independent_hybrid']
    baseline_error = baseline['geographic_error_meters']
    hybrid_error = hybrid['geographic_error_meters']
    paired = (
        baseline['valid']
        and hybrid['valid']
        and baseline_error is not None
        and hybrid_error is not None
    )
    if paired:
        delta = hybrid_error - baseline_error
        outcome = 'improved' if delta < 0.0 else 'worsened' if delta > 0.0 else 'unchanged'
    else:
        delta = None
        if baseline['valid'] and not hybrid['valid']:
            outcome = 'hybrid_invalidated'
        elif not baseline['valid'] and hybrid['valid']:
            outcome = 'hybrid_recovered'
        elif not baseline['valid'] and not hybrid['valid']:
            outcome = 'both_invalid'
        else:
            outcome = 'unpaired'
    changed = _changed_correspondences(result, capture_rows)
    rerun_changed = _rerun_changed_correspondences(result)
    return {
        'scene_id': result['scene_id'],
        'baseline_valid': baseline['valid'],
        'hybrid_valid': hybrid['valid'],
        'baseline_error_meters': baseline_error,
        'hybrid_error_meters': hybrid_error,
        'delta_hybrid_minus_baseline_meters': delta,
        'outcome': outcome,
        'saved_observation_count': result['observation_counts']['saved_face_observations'],
        'independent_selected_count': result['observation_counts']['independent_selected_matched_ids'],
        'changed_correspondence_count': len(changed),
        'changed_object_ids': [row['object_id'] for row in changed],
        'changed_correspondences': changed,
        'rerun_changed_correspondence_count': len(rerun_changed),
        'rerun_changed_object_ids': [
            row['object_id'] for row in rerun_changed
        ],
        'rerun_changed_correspondences': rerun_changed,
        'causal_attribution_available': False,
    }


def _threshold_report(rows, variant):
    errors = [row[f'{variant}_error_meters'] for row in rows]
    return {
        'valid_count': sum(row[f'{variant}_valid'] for row in rows),
        'invalid_count': sum(not row[f'{variant}_valid'] for row in rows),
        'worst_valid_error_meters': max(
            (error for error in errors if error is not None),
            default=None,
        ),
        'threshold_rates': _error_rates(errors, len(rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv-output', type=Path, required=True)
    args = parser.parse_args()

    campaign = json.loads(args.campaign.read_text(encoding='utf-8'))
    rows = [
        _scene_row(
            result,
            _capture_rows(args.captures, result['scene_id']),
        )
        for result in campaign['results']
    ]
    output = {
        'diagnostic_only': True,
        'ground_truth_used_for_selection': False,
        'production_changed': False,
        'source_campaign': args.campaign.name,
        'capture_directory': str(args.captures),
        'fixed_settings': campaign['fixed_settings'],
        'causal_attribution_available': False,
        'definition': {
            'improved': 'both variants valid and hybrid geographic error is lower',
            'worsened': 'both variants valid and hybrid geographic error is higher',
            'hybrid_recovered': 'baseline invalid and hybrid valid',
            'hybrid_invalidated': 'baseline valid and hybrid invalid',
            'threshold_rates': 'reported over valid fixes and all attempts',
        },
        'summary': {
            'scene_count': len(rows),
            'outcomes': {
                outcome: sum(row['outcome'] == outcome for row in rows)
                for outcome in sorted({row['outcome'] for row in rows})
            },
            'changed_correspondence_scene_count': sum(
                row['changed_correspondence_count'] > 0 for row in rows
            ),
            'changed_correspondence_count': sum(
                row['changed_correspondence_count'] for row in rows
            ),
            'rerun_changed_correspondence_scene_count': sum(
                row['rerun_changed_correspondence_count'] > 0
                for row in rows
            ),
            'rerun_changed_correspondence_count': sum(
                row['rerun_changed_correspondence_count'] for row in rows
            ),
            'baseline': _threshold_report(rows, 'baseline'),
            'hybrid': _threshold_report(rows, 'hybrid'),
            'paired_valid': {
                'count': sum(
                    row['baseline_valid'] and row['hybrid_valid']
                    and row['delta_hybrid_minus_baseline_meters'] is not None
                    for row in rows
                ),
                'improved_count': sum(row['outcome'] == 'improved' for row in rows),
                'worsened_count': sum(row['outcome'] == 'worsened' for row in rows),
                'unchanged_count': sum(row['outcome'] == 'unchanged' for row in rows),
                'delta_meters': _summary(
                    row['delta_hybrid_minus_baseline_meters'] for row in rows
                ),
            },
        },
        'per_scene': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding='utf-8')

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open('w', newline='', encoding='utf-8') as stream:
        fieldnames = [
            'scene_id',
            'outcome',
            'baseline_valid',
            'hybrid_valid',
            'baseline_error_meters',
            'hybrid_error_meters',
            'delta_hybrid_minus_baseline_meters',
            'saved_observation_count',
            'independent_selected_count',
            'changed_correspondence_count',
            'changed_object_ids',
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    ';'.join(row[key]) if isinstance(row[key], list)
                    else row[key]
                )
                for key in fieldnames
            })

    print(json.dumps({
        'output': str(args.output),
        'csv_output': str(args.csv_output),
        'summary': output['summary'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
