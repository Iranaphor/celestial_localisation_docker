#!/usr/bin/env python3
"""Capture the historical face-failure scene definitions once for replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_face_and_panorama import (  # noqa: E402
    EphemerisProvider,
    StarIdentifier,
    build_parser as build_scene_parser,
    create_bridge,
    run as run_scene,
)

DEFAULT_SCENE_IDS = (
    'london_06_h090',
    'quito_06_h150',
    'tokyo_00_h015',
    'perth_00_h075',
    'perth_06_h195',
    'reykjavik_00_h120',
    'anchorage_00_h135',
)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _historical_reference(path, scene_id):
    if not path or not path.is_file():
        return {
            'available': False,
            'artifact': path.name if path else None,
        }
    payload = json.loads(path.read_text(encoding='utf-8'))
    row = next(
        (row for row in payload.get('scenes', []) if row.get('scene_id') == scene_id),
        None,
    )
    return {
        'available': row is not None,
        'artifact': path.name,
        'face_valid': row.get('face_valid') if row else None,
        'face_failure_reason': row.get('face_failure_reason') if row else None,
        'common_ids': row.get('common_ids') if row else None,
    }


def _write_capture(capture_dir, scene, result, historical_reference, captured_utc):
    capture = {
        'schema_version': 1,
        'capture_type': 'new_capture_of_same_scene_definition',
        'exact_historical_replay': False,
        'captured_utc': captured_utc,
        'scene_definition': scene,
        'historical_reference': historical_reference,
        'resolved_settings': {
            'compare_configuration': _json_safe(result['configuration']),
            'solver_configuration': result['solver_configuration'],
            'pairing': result['pairing'],
        },
        'counts': result['counts'],
        'timing': result['timing'],
        'image_files': result['capture_image_files'],
        'face_observations': result['paired_solver_results']['face']['observations'],
        'panorama_observations': result['paired_solver_results']['panorama']['observations'],
        'solver_results_at_capture_settings': result['paired_solver_results'],
        'rows': result['rows'],
    }
    path = capture_dir / 'capture.json'
    path.write_text(
        json.dumps(_json_safe(capture), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    return path


def main():
    parser = build_scene_parser()
    parser.description = __doc__
    parser.add_argument(
        '--scenes',
        type=Path,
        default=Path(__file__).with_name('fixed_face_panorama_scenes.json'),
    )
    parser.add_argument(
        '--scene-id',
        nargs='+',
        default=list(DEFAULT_SCENE_IDS),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output' / 'face_failure_captures_20260929',
    )
    parser.add_argument(
        '--historical-campaign',
        type=Path,
        default=ROOT / 'celestial_localisation' / 'debug_output' / 'face_panorama_campaign_20260929.json',
    )
    args = parser.parse_args()

    scenes = json.loads(args.scenes.read_text(encoding='utf-8'))
    scenes_by_id = {scene['scene_id']: scene for scene in scenes}
    missing = [scene_id for scene_id in args.scene_id if scene_id not in scenes_by_id]
    if missing:
        raise ValueError(f'scene IDs not present in manifest: {missing}')
    if len(set(args.scene_id)) != len(args.scene_id):
        raise ValueError('scene IDs must be unique')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    captured_utc = datetime.now(timezone.utc).isoformat()
    bridge = create_bridge(args)
    identifier = StarIdentifier(tile_size=args.star_tile_size)
    ephemeris = EphemerisProvider()
    manifest_rows = []
    try:
        for index, scene_id in enumerate(args.scene_id, start=1):
            scene = scenes_by_id[scene_id]
            scene_args = argparse.Namespace(**vars(args))
            for key, value in scene.items():
                setattr(scene_args, key, value)
            capture_dir = args.output_dir / scene_id
            started = time.perf_counter()
            result = run_scene(
                scene_args,
                bridge=bridge,
                identifier=identifier,
                ephemeris=ephemeris,
                capture_dir=capture_dir,
            )
            result['scene_id'] = scene_id
            historical_reference = _historical_reference(
                args.historical_campaign,
                scene_id,
            )
            capture_path = _write_capture(
                capture_dir,
                scene,
                result,
                historical_reference,
                captured_utc,
            )
            manifest_rows.append({
                'scene_id': scene_id,
                'capture_type': 'new_capture_of_same_scene_definition',
                'capture_file': str(capture_path.relative_to(args.output_dir)),
                'scene_runtime_seconds': time.perf_counter() - started,
                'historical_reference': historical_reference,
                'face_observation_count': len(
                    result['paired_solver_results']['face']['observations']
                ),
                'historical_face_failure_reason': historical_reference.get(
                    'face_failure_reason'
                ),
            })
            print(
                f"[{index}/{len(args.scene_id)}] {scene_id}: "
                f"common={result['counts']['common_ids']} "
                f"face_capture={capture_path}",
                flush=True,
            )
    finally:
        bridge.close()

    manifest = {
        'schema_version': 1,
        'capture_type': 'new_captures_of_same_scene_definitions',
        'exact_historical_replay': False,
        'captured_utc': captured_utc,
        'source_scene_manifest': args.scenes.name,
        'historical_campaign': args.historical_campaign.name,
        'scene_ids': list(args.scene_id),
        'scenes': manifest_rows,
        'resolved_capture_settings': {
            key: value for key, value in vars(args).items()
            if key not in {'scene_id', 'output_dir', 'historical_campaign'}
        },
    }
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    print(json.dumps({
        'manifest': str(manifest_path),
        'capture_type': manifest['capture_type'],
        'scene_count': len(manifest_rows),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
