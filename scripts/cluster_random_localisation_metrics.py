#!/usr/bin/env python3
"""Cluster random localisation runs in observation/error space."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "celestial_localisation"
    / "debug_output"
    / "random_localisation_metrics.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "random_localisation_mean_shift.svg"

# One decade of error is treated as roughly as important as ten observations.
OBSERVATION_SCALE = 10.0
DEFAULT_BANDWIDTH_OBSERVATIONS = 5.0
DEFAULT_BANDWIDTH_DECADES = 0.35
DEFAULT_MIN_SUPPORT = 10
DEFAULT_MAX_ITERATIONS = 60
DEFAULT_CONVERGENCE_TOLERANCE = 0.01
DEFAULT_MERGE_DISTANCE = 0.5
KERNEL_RADIUS = 3.0
SEEDS = (
    (25.0, 10.0, "low error"),
    (30.0, 100.0, "medium error"),
    (15.0, 10_000.0, "high error"),
    (0.0, 10_000.0, "zero observations"),
)
REQUIRED_COLUMNS = {"run_index", "identified_objects_used", "error_distance_meters"}
CHART_WIDTH = 480
CHART_HEIGHT = 960
PLOT_LEFT = 72
PLOT_RIGHT = 458
PLOT_TOP = 70
PLOT_BOTTOM = 890
CLUSTER_COLORS = ("#94e3c4", "#e4d47b", "#ffb65c", "#76c9d9", "#c0a0ed")
NOISE_COLOR = "#ff7164"
SEED_REFERENCE_COLORS = ("#e4d47b", "#ffb65c", "#ff7164", "#76c9d9")


def load_points(csv_path: Path) -> list[dict[str, float | int]]:
    """Load valid sample summaries and convert error distance to kilometres."""
    points: list[dict[str, float | int]] = []
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            record_type = (row.get("record_type") or "").strip().lower()
            if record_type and record_type != "summary":
                continue
            try:
                observations = float(row["identified_objects_used"])
                error_km = float(row["error_distance_meters"]) / 1000.0
                run_index = int(row["run_index"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value on CSV row {row_number}") from error

            if not (
                math.isfinite(observations)
                and math.isfinite(error_km)
                and observations >= 0
                and error_km > 0
            ):
                continue

            points.append(
                {
                    "run_index": run_index,
                    "observations": observations,
                    "error_km": error_km,
                    "log_error": math.log10(error_km),
                }
            )

    if not points:
        raise ValueError("CSV contains no valid positive-error runs")
    return points


def feature(point: dict[str, float | int]) -> tuple[float, float]:
    return (
        float(point["observations"]) / OBSERVATION_SCALE,
        float(point["log_error"]),
    )


def normalized_distance(
    first: tuple[float, float],
    second: tuple[float, float],
    bandwidth_x: float,
    bandwidth_y: float,
) -> float:
    return math.sqrt(
        ((first[0] - second[0]) / bandwidth_x) ** 2
        + ((first[1] - second[1]) / bandwidth_y) ** 2
    )


def mean_shift(
    points: list[dict[str, float | int]],
    bandwidth_observations: float = DEFAULT_BANDWIDTH_OBSERVATIONS,
    bandwidth_decades: float = DEFAULT_BANDWIDTH_DECADES,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
    merge_distance: float = DEFAULT_MERGE_DISTANCE,
) -> tuple[list[int], int, list[tuple[float, float, int]], list[int | None]]:
    """Find density modes by moving starts uphill through a Gaussian KDE."""
    if bandwidth_observations <= 0 or bandwidth_decades <= 0:
        raise ValueError("Mean-shift bandwidths must be greater than zero")
    if min_support < 1:
        raise ValueError("Mean-shift min_support must be at least one")
    if max_iterations < 1 or convergence_tolerance <= 0 or merge_distance <= 0:
        raise ValueError("Mean-shift iteration, tolerance, and merge settings are invalid")

    features = [feature(point) for point in points]
    bandwidth_x = bandwidth_observations / OBSERVATION_SCALE
    bandwidth_y = bandwidth_decades
    cell_x = bandwidth_x
    cell_y = bandwidth_y
    spatial_index: dict[tuple[int, int], list[int]] = {}
    for index, (x_value, y_value) in enumerate(features):
        cell = (math.floor(x_value / cell_x), math.floor(y_value / cell_y))
        spatial_index.setdefault(cell, []).append(index)

    def nearby_indices(location: tuple[float, float]) -> list[int]:
        cell = (
            math.floor(location[0] / cell_x),
            math.floor(location[1] / cell_y),
        )
        radius_cells = math.ceil(KERNEL_RADIUS)
        candidates: list[int] = []
        for cell_x_index in range(cell[0] - radius_cells, cell[0] + radius_cells + 1):
            for cell_y_index in range(cell[1] - radius_cells, cell[1] + radius_cells + 1):
                candidates.extend(spatial_index.get((cell_x_index, cell_y_index), ()))
        return [
            index
            for index in candidates
            if normalized_distance(location, features[index], bandwidth_x, bandwidth_y)
            <= KERNEL_RADIUS
        ]

    def shift(start: tuple[float, float]) -> tuple[tuple[float, float], int, float]:
        location = start
        for _ in range(max_iterations):
            neighbours = nearby_indices(location)
            if not neighbours:
                return location, 0, 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            total_weight = 0.0
            for index in neighbours:
                distance = normalized_distance(location, features[index], bandwidth_x, bandwidth_y)
                weight = math.exp(-0.5 * distance * distance)
                weighted_x += features[index][0] * weight
                weighted_y += features[index][1] * weight
                total_weight += weight
            shifted = (weighted_x / total_weight, weighted_y / total_weight)
            if normalized_distance(location, shifted, bandwidth_x, bandwidth_y) < convergence_tolerance:
                location = shifted
                break
            location = shifted

        final_neighbours = nearby_indices(location)
        density = sum(
            math.exp(
                -0.5
                * normalized_distance(location, features[index], bandwidth_x, bandwidth_y) ** 2
            )
            for index in final_neighbours
        )
        return location, len(final_neighbours), density

    mode_locations: list[tuple[float, float]] = []
    mode_supports: list[int] = []
    mode_densities: list[float] = []

    def register_mode(location: tuple[float, float], support: int, density: float) -> int | None:
        if support < min_support:
            return None
        for mode_index, mode in enumerate(mode_locations):
            if normalized_distance(location, mode, bandwidth_x, bandwidth_y) <= merge_distance:
                if density > mode_densities[mode_index]:
                    mode_locations[mode_index] = location
                    mode_densities[mode_index] = density
                mode_supports[mode_index] = max(mode_supports[mode_index], support)
                return mode_index
        mode_locations.append(location)
        mode_supports.append(support)
        mode_densities.append(density)
        return len(mode_locations) - 1

    seed_starts = [
        (observations / OBSERVATION_SCALE, math.log10(error_km))
        for observations, error_km, _ in SEEDS
    ]
    seed_locations: list[tuple[float, float]] = []
    for seed in seed_starts:
        location, support, density = shift(seed)
        seed_locations.append(location)
        register_mode(location, support, density)

    point_locations: list[tuple[float, float]] = []
    for start in features:
        location, support, density = shift(start)
        point_locations.append(location)
        register_mode(location, support, density)

    for mode_index, location in enumerate(mode_locations):
        support = nearby_indices(location)
        mode_supports[mode_index] = len(support)
        mode_densities[mode_index] = sum(
            math.exp(
                -0.5
                * normalized_distance(location, features[index], bandwidth_x, bandwidth_y) ** 2
            )
            for index in support
        )

    def nearest_mode(location: tuple[float, float]) -> int | None:
        if not mode_locations:
            return None
        mode_index = min(
            range(len(mode_locations)),
            key=lambda index: normalized_distance(
                location, mode_locations[index], bandwidth_x, bandwidth_y
            ),
        )
        distance = normalized_distance(
            location, mode_locations[mode_index], bandwidth_x, bandwidth_y
        )
        if distance > merge_distance or mode_supports[mode_index] < min_support:
            return None
        return mode_index

    point_assignments: list[int] = []
    for location in point_locations:
        mode_index = nearest_mode(location)
        point_assignments.append(mode_index if mode_index is not None else -1)
    seed_cluster_ids = [nearest_mode(location) for location in seed_locations]

    peaks = [
        (location[0], location[1], mode_supports[index])
        for index, location in enumerate(mode_locations)
    ]
    return point_assignments, len(peaks), peaks, seed_cluster_ids


def format_error(error_km: float) -> str:
    if error_km >= 1000:
        return f"{error_km:,.0f} km"
    if error_km >= 100:
        return f"{error_km:,.1f} km"
    return f"{error_km:,.2f} km"


def format_axis_number(value: float) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"
    return f"{numeric_value:,.1f}"


def observation_ticks(maximum_observations: int) -> list[float]:
    raw_step = maximum_observations / 5
    magnitude = 10 ** math.floor(math.log10(raw_step or 1))
    ratio = raw_step / magnitude
    step_multiplier = 1 if ratio <= 1 else 2 if ratio <= 2 else 5 if ratio <= 5 else 10
    step = step_multiplier * magnitude
    ticks = [0.0]
    value = step
    while value < maximum_observations:
        ticks.append(value)
        value += step
    if ticks[-1] != maximum_observations:
        ticks.append(float(maximum_observations))
    return ticks


def write_svg(
    output_path: Path,
    points: list[dict[str, float | int]],
    assignments: list[int],
    peaks: list[tuple[float, float, int]],
) -> None:
    """Write a self-contained graph matching the HTML observation/error chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    maximum_observations = max(
        math.ceil(max(float(point["observations"]) for point in points)),
        math.ceil(max(seed[0] for seed in SEEDS)),
        1,
    )
    error_values = [float(point["error_km"]) for point in points]
    error_values.extend(seed[1] for seed in SEEDS)
    minimum_exponent = math.floor(math.log10(min(error_values)))
    maximum_exponent = math.ceil(math.log10(max(max(error_values), 1)))
    if maximum_exponent <= minimum_exponent:
        maximum_exponent = minimum_exponent + 1

    cluster_ids = sorted({assignment for assignment in assignments if assignment >= 0})
    cluster_color_by_id = {
        cluster_id: CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
        for index, cluster_id in enumerate(cluster_ids)
    }
    members_by_cluster = {
        cluster_id: sum(assignment == cluster_id for assignment in assignments)
        for cluster_id in cluster_ids
    }
    legend_items = [
        (f"cluster {cluster_id + 1}", members_by_cluster[cluster_id], cluster_color_by_id[cluster_id])
        for cluster_id in cluster_ids
    ]
    noise_count = sum(assignment == -1 for assignment in assignments)
    if noise_count:
        legend_items.append(("noise / outlier", noise_count, NOISE_COLOR))
    plot_top = PLOT_TOP + max(0, ((len(legend_items) + 1) // 2) - 2) * 17

    def scale_x(observations: float) -> float:
        return PLOT_LEFT + (observations / maximum_observations) * (PLOT_RIGHT - PLOT_LEFT)

    def scale_y(error_km: float) -> float:
        return PLOT_BOTTOM - (
            (math.log10(max(error_km, 10**minimum_exponent)) - minimum_exponent)
            / (maximum_exponent - minimum_exponent)
        ) * (PLOT_BOTTOM - plot_top)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_WIDTH}" '
        f'height="{CHART_HEIGHT}" viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" '
        'role="img" aria-labelledby="chart-title chart-description">',
        '  <title id="chart-title">Observations versus error with mean-shift density peaks</title>',
        '  <desc id="chart-description">'
        'Each point is a localisation run. The horizontal axis shows observations used '
        'and the vertical axis shows error in kilometres on a logarithmic scale. '
        'Mean shift discovers local density peaks and marks sparse points as noise. '
        'Crosshairs show the original seed locations for reference.</desc>',
        '  <style>',
        '    .axis { fill: #5f7779; font-family: "IBM Plex Mono", monospace; font-size: 10px; letter-spacing: 0.04em; }',
        '    .axis-title { fill: #91a8a8; font-family: "IBM Plex Mono", monospace; font-size: 10px; letter-spacing: 0.05em; text-transform: uppercase; }',
        '    .grid { stroke: rgba(154, 196, 190, 0.13); stroke-width: 1; }',
        '    .plot { fill: rgba(148, 227, 196, 0.025); stroke: rgba(154, 196, 190, 0.17); stroke-width: 1; }',
        '    .point { stroke: #10272b; stroke-width: 0.75; opacity: 0.86; }',
        '    .seed { fill: none; stroke-width: 1.5; stroke-dasharray: 3 2; }',
        '    .peak { stroke: #071216; stroke-width: 1.5; }',
        '    .title { fill: #e8f1ee; font-family: "Space Grotesk", sans-serif; font-size: 13px; font-weight: 600; }',
        '    .legend { fill: #b6cbc5; font-family: "IBM Plex Mono", monospace; font-size: 9px; }',
        '  </style>',
        '  <rect width="100%" height="100%" fill="#071216"/>',
        '  <text class="title" x="72" y="24">Observations vs error / mean shift</text>',
    ]

    for item_index, (name, count, color) in enumerate(legend_items):
        x = 72 if item_index % 2 == 0 else 270
        y = 41 + (item_index // 2) * 17
        lines.append(
            f'  <circle cx="{x}" cy="{y}" r="4" fill="{color}"/>'
        )
        lines.append(
            f'  <text class="legend" x="{x + 9}" y="{y + 3}">{name} (n={count})</text>'
        )

    lines.append(
        f'  <rect class="plot" x="{PLOT_LEFT}" y="{plot_top}" '
        f'width="{PLOT_RIGHT - PLOT_LEFT}" height="{PLOT_BOTTOM - plot_top}"/>'
    )
    for value in observation_ticks(maximum_observations):
        x = scale_x(value)
        lines.append(
            f'  <line class="grid" x1="{x:.2f}" y1="{plot_top}" '
            f'x2="{x:.2f}" y2="{PLOT_BOTTOM}"/>'
        )
        lines.append(
            f'  <text class="axis" x="{x:.2f}" y="914" text-anchor="middle">'
            f'{format_axis_number(value)}</text>'
        )
    for exponent in range(minimum_exponent, maximum_exponent + 1):
        error_km = 10**exponent
        y = scale_y(error_km)
        lines.append(
            f'  <line class="grid" x1="{PLOT_LEFT}" y1="{y:.2f}" '
            f'x2="{PLOT_RIGHT}" y2="{y:.2f}"/>'
        )
        lines.append(
            f'  <text class="axis" x="{PLOT_LEFT - 12}" y="{y + 4:.2f}" '
            f'text-anchor="end">{format_axis_number(error_km)}</text>'
        )

    for point, assignment in zip(points, assignments):
        x = scale_x(float(point["observations"]))
        y = scale_y(float(point["error_km"]))
        point_color = NOISE_COLOR if assignment == -1 else cluster_color_by_id[assignment]
        cluster_name = "noise / outlier" if assignment == -1 else f"cluster {assignment + 1}"
        lines.append(
            f'  <circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="2.7" '
            f'fill="{point_color}">'
            f'<title>Run {point["run_index"]}: {format_axis_number(float(point["observations"]))} '
            f'observations, {format_error(float(point["error_km"]))}, '
            f'{cluster_name}</title></circle>'
        )

    for cluster_index, (peak_x, peak_y, support) in enumerate(peaks):
        color = cluster_color_by_id.get(
            cluster_index,
            CLUSTER_COLORS[cluster_index % len(CLUSTER_COLORS)],
        )
        peak_observations = peak_x * OBSERVATION_SCALE
        peak_error = 10**peak_y
        lines.append(
            f'  <circle class="peak" cx="{scale_x(peak_observations):.2f}" '
            f'cy="{scale_y(peak_error):.2f}" r="5" fill="{color}">'
            f'<title>Density peak {cluster_index + 1}: '
            f'{format_axis_number(peak_observations)} observations, '
            f'{format_error(peak_error)}, support {support}</title></circle>'
        )

    for (seed_observations, seed_error, name), color in zip(SEEDS, SEED_REFERENCE_COLORS):
        seed_x = scale_x(seed_observations)
        seed_y = scale_y(seed_error)
        lines.append(
            f'  <g class="seed" stroke="{color}">'
            f'<line x1="{seed_x - 5:.2f}" y1="{seed_y:.2f}" '
            f'x2="{seed_x + 5:.2f}" y2="{seed_y:.2f}"/>'
            f'<line x1="{seed_x:.2f}" y1="{seed_y - 5:.2f}" '
            f'x2="{seed_x:.2f}" y2="{seed_y + 5:.2f}"/>'
            f'<title>{name} initial seed reference: {seed_observations:g} observations, '
            f'{format_error(seed_error)}</title></g>'
        )

    lines.extend(
        [
            '  <text class="axis-title" x="265" y="950" text-anchor="middle">observations used</text>',
            '  <text class="axis-title" x="20" y="472" text-anchor="middle" transform="rotate(-90 20 472)">error km / log scale</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_results(
    csv_path: Path,
    points: list[dict[str, float | int]],
    assignments: list[int],
    peaks: list[tuple[float, float, int]],
    seed_cluster_ids: list[int | None],
    bandwidth_observations: float,
    bandwidth_decades: float,
    min_support: int,
    output_path: Path,
) -> None:
    print(f"Input: {csv_path}")
    print(f"Valid runs: {len(points)}")
    print(
        f"Mean shift: bandwidth={bandwidth_observations:g} observations x "
        f"{bandwidth_decades:g} decades | min_support={min_support}"
    )
    print(f"Graph: {output_path}")
    noise_count = sum(assignment == -1 for assignment in assignments)
    print(f"Density peaks: {len(peaks)} | noise/outliers: {noise_count}")
    print("Seed starts:")
    for seed, cluster_id in zip(SEEDS, seed_cluster_ids):
        seed_observations, seed_error, name = seed
        cluster_name = f"cluster {cluster_id + 1}" if cluster_id is not None else "noise"
        print(
            f"  {name}: {seed_observations:g} observations / {format_error(seed_error)} "
            f"-> {cluster_name}"
        )

    for cluster_index, (peak_x, peak_y, support) in enumerate(peaks):
        members = [
            point
            for point, assignment in zip(points, assignments)
            if assignment == cluster_index
        ]
        if not members:
            continue
        mean_observations = sum(float(point["observations"]) for point in members) / len(members)
        geometric_mean_error = 10 ** (
            sum(float(point["log_error"]) for point in members) / len(members)
        )
        minimum_error = min(float(point["error_km"]) for point in members)
        maximum_error = max(float(point["error_km"]) for point in members)
        print()
        print(
            f"Cluster {cluster_index + 1}: {len(members)} runs "
            f"| peak {peak_x * OBSERVATION_SCALE:.1f} observations / "
            f"{format_error(10**peak_y)} / support {support} "
            f"| mean {mean_observations:.1f} observations / "
            f"{format_error(geometric_mean_error)} geometric-mean error"
        )
        print(
            f"  error range: {format_error(minimum_error)} to {format_error(maximum_error)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover localisation density peaks by observations used and "
            "log-scaled error distance."
        )
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=DEFAULT_CSV,
        help=f"metrics CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"SVG graph path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--bandwidth-observations",
        type=float,
        default=DEFAULT_BANDWIDTH_OBSERVATIONS,
        help=(
            "KDE bandwidth along observations (default: "
            f"{DEFAULT_BANDWIDTH_OBSERVATIONS})"
        ),
    )
    parser.add_argument(
        "--bandwidth-decades",
        type=float,
        default=DEFAULT_BANDWIDTH_DECADES,
        help=f"KDE bandwidth in log10 decades (default: {DEFAULT_BANDWIDTH_DECADES})",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help=f"Minimum support for a density peak (default: {DEFAULT_MIN_SUPPORT})",
    )
    parser.add_argument(
        "--merge-distance",
        type=float,
        default=DEFAULT_MERGE_DISTANCE,
        help=f"Peak merge distance in bandwidth units (default: {DEFAULT_MERGE_DISTANCE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.csv_path)
    (
        assignments,
        _cluster_count,
        peaks,
        seed_cluster_ids,
    ) = mean_shift(
        points,
        args.bandwidth_observations,
        args.bandwidth_decades,
        args.min_support,
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_CONVERGENCE_TOLERANCE,
        args.merge_distance,
    )
    write_svg(args.output_path, points, assignments, peaks)
    print_results(
        args.csv_path,
        points,
        assignments,
        peaks,
        seed_cluster_ids,
        args.bandwidth_observations,
        args.bandwidth_decades,
        args.min_support,
        args.output_path,
    )


if __name__ == "__main__":
    main()
