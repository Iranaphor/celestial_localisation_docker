#!/usr/bin/env python3
"""Measure rendered-star angular errors and fit a diagnostic common rotation.

The common rotation is diagnostic only. It may absorb a level/heading error and
must not be treated as a correction to the localisation model.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
for source_directory in (
    ROOT / "celestial_localisation" / "src" / "celestial_simulation",
    ROOT / "celestial_localisation" / "src" / "celestial_detector",
    ROOT / "celestial_localisation" / "src" / "celestial_localiser",
):
    sys.path.insert(0, str(source_directory))

from celestial_simulation.projection import cube_faces  # noqa: E402


FACE_COLORS = {
    "azimuth_0": "#e07a5f",
    "azimuth_90": "#3d85c6",
    "azimuth_180": "#81b29a",
    "azimuth_270": "#f2cc8f",
    "zenith": "#8e7dbe",
    "nadir": "#6c9a8b",
    "unknown": "#999999",
}
CSV_FIELDS = (
    "object_id",
    "pixel_x",
    "pixel_y",
    "normalized_x",
    "normalized_y",
    "source_face",
    "expected_azimuth_deg",
    "expected_elevation_deg",
    "observed_azimuth_deg",
    "observed_elevation_deg",
    "azimuth_error_deg",
    "cross_track_azimuth_error_deg",
    "elevation_error_deg",
    "angular_error_deg",
    "common_rotation_azimuth_error_deg",
    "common_rotation_cross_track_error_deg",
    "common_rotation_elevation_error_deg",
    "common_rotation_angular_error_deg",
)


def wrapped_delta(first_degrees, second_degrees):
    return (first_degrees - second_degrees + 180.0) % 360.0 - 180.0


def direction_from_azimuth_elevation(azimuth_degrees, elevation_degrees):
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    cosine = math.cos(elevation)
    return np.asarray(
        (
            math.cos(azimuth) * cosine,
            math.sin(azimuth) * cosine,
            math.sin(elevation),
        ),
        dtype=np.float64,
    )


def azimuth_elevation_from_direction(direction):
    normalized = np.asarray(direction, dtype=np.float64)
    normalized = normalized / np.linalg.norm(normalized)
    return (
        math.degrees(math.atan2(normalized[1], normalized[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(normalized[2], -1.0, 1.0)))),
    )


def source_face_for_direction(direction, field_of_view_degrees):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    tangent = math.tan(math.radians(field_of_view_degrees) / 2.0)
    candidates = []
    for face in cube_faces():
        depth = float(np.dot(direction, face.center))
        horizontal = float(np.dot(direction, face.right))
        vertical = float(np.dot(direction, face.up))
        if (
            depth > 0.0
            and abs(horizontal) <= depth * tangent
            and abs(vertical) <= depth * tangent
        ):
            candidates.append((depth, face.name))
    if not candidates:
        return "unknown"
    return max(candidates)[1]


def angular_error_degrees(first, second):
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def component_errors(
    observed_azimuth,
    observed_elevation,
    expected_azimuth,
    expected_elevation,
):
    azimuth_error = wrapped_delta(observed_azimuth, expected_azimuth)
    elevation_error = observed_elevation - expected_elevation
    cross_track_error = azimuth_error * math.cos(math.radians(expected_elevation))
    return azimuth_error, cross_track_error, elevation_error


def fit_common_rotation(observed_vectors, expected_vectors):
    if len(observed_vectors) < 2:
        return None
    observed = np.asarray(observed_vectors, dtype=np.float64)
    expected = np.asarray(expected_vectors, dtype=np.float64)
    rotation, _ = Rotation.align_vectors(observed, expected)
    rotated_expected = rotation.apply(expected)
    angular_errors = np.asarray(
        [angular_error_degrees(observed_row, rotated_row)
         for observed_row, rotated_row in zip(observed, rotated_expected)],
        dtype=np.float64,
    )
    return {
        "rotation": rotation,
        "rotated_expected": rotated_expected,
        "rotation_vector_degrees": np.degrees(rotation.as_rotvec()).tolist(),
        "rotation_angle_degrees": float(np.degrees(rotation.magnitude())),
        "residuals_degrees": angular_errors.tolist(),
    }


def summarize(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _svg_text(value):
    return html.escape(str(value), quote=True)


def _extent(values, default=(-1.0, 1.0)):
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return default
    lower = min(finite_values)
    upper = max(finite_values)
    if math.isclose(lower, upper):
        padding = max(1.0, abs(lower) * 0.1)
    else:
        padding = (upper - lower) * 0.1
    return lower - padding, upper + padding


def _write_panel(lines, rows, x_key, y_key, left, top, width, height, title, x_label, y_label):
    plot_left = left + 58
    plot_top = top + 28
    plot_width = width - 78
    plot_height = height - 58
    x_min, x_max = _extent([row[x_key] for row in rows])
    y_min, y_max = _extent([row[y_key] for row in rows])

    def scale_x(value):
        return plot_left + (float(value) - x_min) / (x_max - x_min) * plot_width

    def scale_y(value):
        return plot_top + plot_height - (float(value) - y_min) / (y_max - y_min) * plot_height

    lines.append(f'<text class="title" x="{left}" y="{top + 14}">{_svg_text(title)}</text>')
    lines.append(
        f'<rect class="plot" x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}"/>'
    )
    lines.append(
        f'<text class="axis" x="{plot_left + plot_width / 2:.1f}" y="{top + height - 8}" text-anchor="middle">{_svg_text(x_label)}</text>'
    )
    lines.append(
        f'<text class="axis" x="{left + 12}" y="{plot_top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 {left + 12} {plot_top + plot_height / 2:.1f})">{_svg_text(y_label)}</text>'
    )
    lines.append(
        f'<text class="tick" x="{plot_left}" y="{plot_top + plot_height + 16}">{x_min:.2f}</text>'
    )
    lines.append(
        f'<text class="tick" x="{plot_left + plot_width}" y="{plot_top + plot_height + 16}" text-anchor="end">{x_max:.2f}</text>'
    )
    lines.append(
        f'<text class="tick" x="{plot_left - 6}" y="{plot_top + plot_height}" text-anchor="end">{y_min:.2f}</text>'
    )
    lines.append(
        f'<text class="tick" x="{plot_left - 6}" y="{plot_top + 4}" text-anchor="end">{y_max:.2f}</text>'
    )
    for row in rows:
        x = scale_x(row[x_key])
        y = scale_y(row[y_key])
        color = FACE_COLORS.get(row["source_face"], FACE_COLORS["unknown"])
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" opacity="0.76">'
            f'<title>{_svg_text(row["object_id"])} / {_svg_text(row["source_face"])}: '
            f'{float(row[y_key]):.5f}</title></circle>'
        )


def write_svg(rows, output_path, title, common_rotation):
    width = 1120
    height = 790
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        '.title { fill: #e8f1ee; font: 600 14px sans-serif; }',
        '.axis { fill: #b6cbc5; font: 11px sans-serif; }',
        '.tick { fill: #718887; font: 10px monospace; }',
        '.plot { fill: #0b1c20; stroke: #315052; stroke-width: 1; }',
        '.background { fill: #071216; }',
        '</style>',
        '<rect class="background" width="100%" height="100%"/>',
        f'<text class="title" x="32" y="28">{_svg_text(title)}</text>',
    ]
    if common_rotation is not None:
        lines.append(
            f'<text class="axis" x="32" y="50">common rotation angle: {common_rotation["rotation_angle_degrees"]:.5f} deg; '
            f'post-fit residual median: {summarize(common_rotation["residuals_degrees"])["median"]:.5f} deg</text>'
        )
    panels = (
        (32, 70, "cross_track_azimuth_error_deg", "elevation_error_deg", "Signed components", "cross-track azimuth error (deg)", "elevation error (deg)"),
        (570, 70, "expected_elevation_deg", "cross_track_azimuth_error_deg", "Azimuth error vs elevation", "expected elevation (deg)", "cross-track error (deg)"),
        (32, 410, "pixel_x", "cross_track_azimuth_error_deg", "Azimuth error vs image X", "pixel X", "cross-track error (deg)"),
        (570, 410, "pixel_y", "elevation_error_deg", "Elevation error vs image Y", "pixel Y", "elevation error (deg)"),
    )
    for left, top, x_key, y_key, panel_title, x_label, y_label in panels:
        _write_panel(lines, rows, x_key, y_key, left, top, 500, 320, panel_title, x_label, y_label)

    legend_y = 760
    legend_x = 32
    for face_name, color in FACE_COLORS.items():
        lines.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" fill="{color}"/>')
        lines.append(f'<text class="axis" x="{legend_x + 9}" y="{legend_y + 4}">{_svg_text(face_name)}</text>')
        legend_x += 145

    lines.append('</svg>')
    Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def analyze_case(args):
    import cv2
    from celestial_detector.angular_projection import pixel_to_az_el
    from celestial_detector.moon_detector import detect_moon
    from celestial_detector.star_detector import detect_stars
    from celestial_detector.star_identifier import StarIdentifier, identify_stars
    from celestial_detector.sun_detector import detect_sun
    from celestial_localiser.ephemeris import EphemerisProvider
    from celestial_simulation.renderer import StellariumRenderer

    renderer = StellariumRenderer(
        engine_js=args.engine_js,
        engine_wasm=args.engine_wasm,
        data_root=args.data_root,
        browser_executable=args.browser_executable,
        face_size=args.face_size,
        panorama_width=args.panorama_width,
        panorama_height=args.panorama_height,
        face_field_of_view=args.face_fov,
        enable_landscape=False,
        timeout_seconds=args.timeout,
    )
    try:
        image = renderer.render(
            latitude=args.latitude,
            longitude=args.longitude,
            altitude=args.altitude,
            timestamp_ms=args.timestamp * 1000.0,
            yaw_degrees=args.heading,
        )
    finally:
        renderer.close()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    exclusions = []
    if args.mask_solar_objects:
        for detection in (detect_sun(gray), detect_moon(gray)):
            if detection is not None:
                exclusions.append(detection)
    detections = detect_stars(
        gray,
        threshold_sigma=args.star_threshold,
        exclusion_detections=exclusions,
        minimum_elevation_degrees=args.min_elevation,
        max_candidates=args.max_candidates,
    )
    identifier = StarIdentifier(
        database_path=args.star_database_path,
        fov_degrees=args.star_fov,
        fov_max_error_degrees=args.star_fov_max_error,
        tile_size=args.star_tile_size,
        match_radius=args.match_radius,
        match_threshold=args.match_threshold,
        min_matches=args.min_matches,
    )
    detections = identify_stars(
        detections,
        (args.panorama_width, args.panorama_height),
        identifier,
    )
    ephemeris = EphemerisProvider(args.star_database_path)
    rows = []
    observed_vectors = []
    expected_vectors = []
    for detection in detections:
        object_id = str(detection.get("object_id", "UNKNOWN"))
        if object_id == "UNKNOWN":
            continue
        observed_azimuth_local, observed_elevation = pixel_to_az_el(
            detection["pixel_x"],
            detection["pixel_y"],
            args.panorama_width,
            args.panorama_height,
        )
        expected = ephemeris.predict(
            object_id,
            args.timestamp,
            args.latitude,
            args.longitude,
            args.altitude,
        )
        if expected is None:
            continue
        expected_azimuth, expected_elevation = expected
        observed_azimuth = (observed_azimuth_local + args.heading) % 360.0
        observed_vector = direction_from_azimuth_elevation(observed_azimuth, observed_elevation)
        expected_vector = direction_from_azimuth_elevation(expected_azimuth, expected_elevation)
        source_face = source_face_for_direction(
            direction_from_azimuth_elevation(
                expected_azimuth - args.heading,
                expected_elevation,
            ),
            args.face_fov,
        )
        azimuth_error, cross_track_error, elevation_error = component_errors(
            observed_azimuth,
            observed_elevation,
            expected_azimuth,
            expected_elevation,
        )
        row = {
            "object_id": object_id,
            "pixel_x": float(detection["pixel_x"]),
            "pixel_y": float(detection["pixel_y"]),
            "normalized_x": float(detection["pixel_x"]) / args.panorama_width,
            "normalized_y": float(detection["pixel_y"]) / args.panorama_height,
            "source_face": source_face,
            "expected_azimuth_deg": expected_azimuth,
            "expected_elevation_deg": expected_elevation,
            "observed_azimuth_deg": observed_azimuth,
            "observed_elevation_deg": observed_elevation,
            "azimuth_error_deg": azimuth_error,
            "cross_track_azimuth_error_deg": cross_track_error,
            "elevation_error_deg": elevation_error,
            "angular_error_deg": angular_error_degrees(observed_vector, expected_vector),
        }
        rows.append(row)
        observed_vectors.append(observed_vector)
        expected_vectors.append(expected_vector)

    common_rotation = fit_common_rotation(observed_vectors, expected_vectors)
    if common_rotation is not None:
        for row, rotated_expected in zip(rows, common_rotation["rotated_expected"]):
            rotated_azimuth, rotated_elevation = azimuth_elevation_from_direction(rotated_expected)
            azimuth_error, cross_track_error, elevation_error = component_errors(
                row["observed_azimuth_deg"],
                row["observed_elevation_deg"],
                rotated_azimuth,
                rotated_elevation,
            )
            row.update({
                "common_rotation_azimuth_error_deg": azimuth_error,
                "common_rotation_cross_track_error_deg": cross_track_error,
                "common_rotation_elevation_error_deg": elevation_error,
                "common_rotation_angular_error_deg": angular_error_degrees(
                    direction_from_azimuth_elevation(
                        row["observed_azimuth_deg"],
                        row["observed_elevation_deg"],
                    ),
                    rotated_expected,
                ),
            })
    else:
        for row in rows:
            row.update({
                "common_rotation_azimuth_error_deg": float("nan"),
                "common_rotation_cross_track_error_deg": float("nan"),
                "common_rotation_elevation_error_deg": float("nan"),
                "common_rotation_angular_error_deg": float("nan"),
            })

    by_face = defaultdict(list)
    for row in rows:
        by_face[row["source_face"]].append(row)
    summary = {
        "configuration": {
            "latitude": args.latitude,
            "longitude": args.longitude,
            "altitude": args.altitude,
            "heading": args.heading,
            "timestamp_unix": args.timestamp,
            "face_size": args.face_size,
            "panorama_width": args.panorama_width,
            "panorama_height": args.panorama_height,
            "face_fov": args.face_fov,
            "mask_solar_objects": args.mask_solar_objects,
        },
        "counts": {
            "detections": len(detections),
            "identified_detections": sum(
                detection.get("object_id") != "UNKNOWN" for detection in detections
            ),
            "matched_rows": len(rows),
        },
        "before_common_rotation": {
            "angular_error_deg": summarize(row["angular_error_deg"] for row in rows),
            "cross_track_azimuth_error_deg": summarize(row["cross_track_azimuth_error_deg"] for row in rows),
            "elevation_error_deg": summarize(row["elevation_error_deg"] for row in rows),
        },
        "common_rotation": (
            {
                "rotation_vector_degrees": common_rotation["rotation_vector_degrees"],
                "rotation_angle_degrees": common_rotation["rotation_angle_degrees"],
                "post_fit_angular_error_deg": summarize(common_rotation["residuals_degrees"]),
            }
            if common_rotation is not None
            else None
        ),
        "by_source_face": {
            face: {
                "count": len(face_rows),
                "angular_error_deg": summarize(row["angular_error_deg"] for row in face_rows),
                "cross_track_azimuth_error_deg": summarize(row["cross_track_azimuth_error_deg"] for row in face_rows),
                "elevation_error_deg": summarize(row["elevation_error_deg"] for row in face_rows),
            }
            for face, face_rows in sorted(by_face.items())
        },
    }
    return summary, rows, common_rotation


def write_csv(rows, output_path):
    with Path(output_path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, default=23.0)
    parser.add_argument("--longitude", type=float, default=-70.0)
    parser.add_argument("--altitude", type=float, default=0.0)
    parser.add_argument("--heading", type=float, default=17.0)
    parser.add_argument("--timestamp", type=float, default=1700000000.0)
    parser.add_argument("--face-size", type=int, default=512)
    parser.add_argument("--panorama-width", type=int, default=2048)
    parser.add_argument("--panorama-height", type=int, default=1024)
    parser.add_argument("--face-fov", type=float, default=95.0)
    parser.add_argument("--star-threshold", type=float, default=5.0)
    parser.add_argument("--min-elevation", type=float, default=25.0)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--star-fov", type=float, default=30.0)
    parser.add_argument("--star-fov-max-error", type=float, default=3.0)
    parser.add_argument("--star-tile-size", type=int, default=512)
    parser.add_argument("--match-radius", type=float, default=0.02)
    parser.add_argument("--match-threshold", type=float, default=0.001)
    parser.add_argument("--min-matches", type=int, default=4)
    parser.add_argument("--star-database-path", default="")
    parser.add_argument("--engine-js", default="/opt/stellarium/stellarium-web-engine.js")
    parser.add_argument("--engine-wasm", default="/opt/stellarium/stellarium-web-engine.wasm")
    parser.add_argument("--data-root", default="/opt/stellarium/data")
    parser.add_argument("--browser-executable", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--mask-solar-objects", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    summary, rows, common_rotation = analyze_case(args)
    if args.output_prefix:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = args.output_prefix.with_suffix(".json")
        csv_path = args.output_prefix.with_suffix(".csv")
        svg_path = args.output_prefix.with_suffix(".svg")
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_csv(rows, csv_path)
        write_svg(rows, svg_path, "Rendered star angular errors", common_rotation)
        summary["outputs"] = {
            "json": str(json_path),
            "csv": str(csv_path),
            "svg": str(svg_path),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
