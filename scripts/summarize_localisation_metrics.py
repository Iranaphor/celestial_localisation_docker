#!/usr/bin/env python3
"""Summarize localisation metrics without selecting or rewriting records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


THRESHOLDS_METERS = (100.0, 500.0, 1_000.0, 5_000.0, 10_000.0)
DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / 'celestial_localisation'
    / 'debug_output'
    / 'random_localisation_metrics.csv'
)


def _finite_float(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _error_summary(rows):
    errors = [error for row in rows if (error := _finite_float(row, 'error_distance_meters')) is not None]
    result = {
        'count': len(errors),
        'mean_meters': mean(errors) if errors else None,
        'median_meters': median(errors) if errors else None,
        'p90_meters': _percentile(errors, 0.90),
        'p95_meters': _percentile(errors, 0.95),
        'within': {},
    }
    for threshold in THRESHOLDS_METERS:
        result['within'][str(int(threshold))] = {
            'count': sum(error <= threshold for error in errors),
            'fraction': (
                sum(error <= threshold for error in errors) / len(errors)
                if errors
                else None
            ),
        }
    return result


def _signed_summary(rows):
    latitude = [value for row in rows if (value := _finite_float(row, 'latitude_error_meters')) is not None]
    longitude = [value for row in rows if (value := _finite_float(row, 'longitude_error_meters')) is not None]
    return {
        'latitude_mean_meters': mean(latitude) if latitude else None,
        'latitude_median_meters': median(latitude) if latitude else None,
        'longitude_mean_meters': mean(longitude) if longitude else None,
        'longitude_median_meters': median(longitude) if longitude else None,
    }


def summarize(csv_path):
    with Path(csv_path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))

    summaries = [row for row in rows if (row.get('record_type') or '').lower() == 'summary']
    steps = [row for row in rows if (row.get('record_type') or '').lower() == 'step']
    failures = [row for row in rows if (row.get('record_type') or '').lower() == 'failure']
    valid_summaries = [
        row for row in summaries
        if _finite_float(row, 'error_distance_meters') is not None
    ]
    valid_steps = [
        row for row in steps
        if _finite_float(row, 'error_distance_meters') is not None
    ]

    steps_by_sample = defaultdict(list)
    for row in valid_steps:
        steps_by_sample[row.get('sample_id', '')].append(row)
    best_of_five = [
        min((_finite_float(row, 'error_distance_meters') for row in sample_rows), default=None)
        for sample_rows in steps_by_sample.values()
    ]
    best_of_five = [error for error in best_of_five if error is not None]

    attempted_records = len(steps) + len(failures)
    metadata = sorted({row.get('evaluation_config_json', '') for row in rows if row.get('evaluation_config_json')})
    code_revisions = sorted({row.get('code_revision', '') for row in rows if row.get('code_revision')})

    return {
        'file': str(csv_path),
        'rows': len(rows),
        'sample_ids': len({row.get('sample_id', '') for row in rows if row.get('sample_id')}),
        'summary_rows': len(summaries),
        'valid_summary_rows': len(valid_summaries),
        'step_rows': len(steps),
        'valid_step_rows': len(valid_steps),
        'failure_rows': len(failures),
        'attempt_failure_rate': (
            len(failures) / attempted_records if attempted_records else None
        ),
        'summary': {
            **_error_summary(valid_summaries),
            **_signed_summary(valid_summaries),
        },
        'steps': _error_summary(valid_steps),
        'best_of_five_diagnostic': {
            'warning': 'uses ground truth only as an offline diagnostic; it is not an operational selector',
            'sample_count_with_valid_step': len(best_of_five),
            'mean_meters': mean(best_of_five) if best_of_five else None,
            'median_meters': median(best_of_five) if best_of_five else None,
            'p90_meters': _percentile(best_of_five, 0.90),
            'p95_meters': _percentile(best_of_five, 0.95),
            'within': {
                str(int(threshold)): {
                    'count': sum(error <= threshold for error in best_of_five),
                    'fraction': (
                        sum(error <= threshold for error in best_of_five) / len(best_of_five)
                        if best_of_five
                        else None
                    ),
                }
                for threshold in THRESHOLDS_METERS
            },
        },
        'resolved_evaluation_configs': metadata,
        'code_revisions': code_revisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', dest='csv_path', type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv_path), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()