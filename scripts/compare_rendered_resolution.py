#!/usr/bin/env python3
"""Compare rendered-star residuals across resolutions using common star IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_rendered_star_errors import analyze_case


def _case(namespace, face_size, panorama_width, panorama_height):
    return SimpleNamespace(
        **namespace,
        mask_solar_objects=True,
        star_threshold=5.0,
        min_elevation=25.0,
        max_candidates=64,
        star_fov=30.0,
        star_fov_max_error=3.0,
        star_tile_size=512,
        match_radius=0.02,
        match_threshold=0.001,
        min_matches=4,
        star_database_path='',
        face_size=face_size,
        panorama_width=panorama_width,
        panorama_height=panorama_height,
    )


def compare(cases):
    results = []
    for label, args in cases:
        summary, rows, _ = analyze_case(args)
        by_id = {row['object_id']: row for row in rows}
        results.append((label, summary, by_id))

    baseline_label, baseline_summary, baseline_by_id = results[0]
    report = {
        'baseline': baseline_label,
        'cases': [],
    }
    baseline_ids = set(baseline_by_id)
    for label, summary, by_id in results:
        ids = set(by_id)
        common_ids = baseline_ids & ids
        baseline_subset_errors = [
            baseline_by_id[star_id]['angular_error_deg']
            for star_id in common_ids
        ]
        case_subset_errors = [by_id[star_id]['angular_error_deg'] for star_id in common_ids]
        report['cases'].append({
            'label': label,
            'counts': summary['counts'],
            'all_star_ids': sorted(ids),
            'common_with_baseline_count': len(common_ids),
            'lost_from_baseline': sorted(baseline_ids - ids),
            'new_relative_to_baseline': sorted(ids - baseline_ids),
            'common_id_error_median_deg': (
                float(np.median(case_subset_errors)) if case_subset_errors else None
            ),
            'common_id_error_p90_deg': (
                float(np.percentile(case_subset_errors, 90)) if case_subset_errors else None
            ),
            'baseline_same_subset_error_median_deg': (
                float(np.median(baseline_subset_errors)) if baseline_subset_errors else None
            ),
            'baseline_same_subset_error_p90_deg': (
                float(np.percentile(baseline_subset_errors, 90))
                if baseline_subset_errors
                else None
            ),
        })
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--latitude', type=float, default=23.0)
    parser.add_argument('--longitude', type=float, default=-70.0)
    parser.add_argument('--altitude', type=float, default=0.0)
    parser.add_argument('--heading', type=float, default=17.0)
    parser.add_argument('--timestamp', type=float, default=1700000000.0)
    parser.add_argument('--face-fov', type=float, default=95.0)
    parser.add_argument('--engine-js', default='/opt/stellarium/stellarium-web-engine.js')
    parser.add_argument('--engine-wasm', default='/opt/stellarium/stellarium-web-engine.wasm')
    parser.add_argument('--data-root', default='/opt/stellarium/data')
    parser.add_argument('--browser-executable', default='')
    parser.add_argument('--timeout', type=float, default=60.0)
    return parser.parse_args()


def main():
    args = parse_args()
    namespace = vars(args).copy()
    namespace.pop('output', None)
    cases = [
        ('face512_pano2048', _case(namespace, 512, 2048, 1024)),
        ('face768_pano2048', _case(namespace, 768, 2048, 1024)),
        ('face1024_pano2048', _case(namespace, 1024, 2048, 1024)),
        ('face512_pano1024', _case(namespace, 512, 1024, 512)),
    ]
    report = compare(cases)
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
