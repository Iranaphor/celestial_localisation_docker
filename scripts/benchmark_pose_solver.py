#!/usr/bin/env python3
"""Replay saved identified observations through the solver at several iteration limits."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'celestial_localisation' / 'src' / 'celestial_localiser'))

from celestial_localiser.ephemeris import EphemerisProvider
from celestial_localiser.pose_solver import solve


class CountingEphemeris:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0
        self.batch_calls = 0
        self.seconds = 0.0

    def predict(self, *args, **kwargs):
        self.calls += 1
        start = time.perf_counter()
        try:
            return self.provider.predict(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start

    def predict_many(self, object_ids, *args, **kwargs):
        self.batch_calls += 1
        self.calls += len(object_ids)
        start = time.perf_counter()
        try:
            return self.provider.predict_many(object_ids, *args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start


def run(args):
    observations_payload = json.loads(Path(args.observations).read_text(encoding='utf-8'))
    pose_payload = json.loads(Path(args.pose).read_text(encoding='utf-8'))
    observations = [
        observation
        for observation in observations_payload
        if observation.get('object_id', '').upper() != 'UNKNOWN'
    ]
    timestamp_payload = pose_payload['observation_timestamp']
    timestamp = float(timestamp_payload['sec']) + float(timestamp_payload['nanosec']) / 1e9
    initial_guess = pose_payload.get('initial_guess', {})
    initial_state = (
        float(initial_guess.get('latitude', args.initial_latitude)),
        float(initial_guess.get('longitude', args.initial_longitude)),
        float(initial_guess.get('heading', args.initial_heading)),
    )
    provider = EphemerisProvider(args.star_database_path)
    results = []
    max_starts_values = (
        list(args.max_starts)
        if isinstance(args.max_starts, (list, tuple))
        else [args.max_starts]
    )
    for max_starts in max_starts_values:
        for max_nfev in args.max_nfev:
            ephemeris = CountingEphemeris(provider)
            start = time.perf_counter()
            result = solve(
                observations,
                timestamp,
                ephemeris,
                initial_state,
                max_nfev=max_nfev,
                global_search=args.global_search,
                max_starts=max_starts,
            )
            elapsed = time.perf_counter() - start
            row = {
                'max_nfev': max_nfev,
                'max_starts': max_starts,
                'starts': result.starts_tried,
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
                'start_diagnostics': result.start_diagnostics,
                'ephemeris_calls': ephemeris.calls,
                'ephemeris_batch_calls': ephemeris.batch_calls,
                'ephemeris_seconds': ephemeris.seconds,
                'ephemeris_fraction_of_runtime': (
                    ephemeris.seconds / elapsed if elapsed > 0.0 else None
                ),
                'runtime_seconds': elapsed,
                'message': result.message,
            }
            if args.true_latitude is not None:
                row['true_latitude_error_degrees'] = row['latitude'] - args.true_latitude
                row['true_longitude_error_degrees'] = (
                    (row['longitude'] - args.true_longitude + 180.0) % 360.0 - 180.0
                )
            results.append(row)
    return {
        'observations': len(observations),
        'timestamp_unix': timestamp,
        'initial_state': initial_state,
        'global_search': args.global_search,
        'max_starts': max_starts_values,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--pose', type=Path, required=True)
    parser.add_argument('--star-database-path', default='')
    parser.add_argument('--initial-latitude', type=float, default=51.5)
    parser.add_argument('--initial-longitude', type=float, default=-0.1)
    parser.add_argument('--initial-heading', type=float, default=0.0)
    parser.add_argument('--true-latitude', type=float)
    parser.add_argument('--true-longitude', type=float)
    parser.add_argument('--max-nfev', type=int, nargs='+', default=[30, 60, 120, 240])
    parser.add_argument('--global-search', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-starts', type=int, nargs='+', default=[5, 13, 25])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
