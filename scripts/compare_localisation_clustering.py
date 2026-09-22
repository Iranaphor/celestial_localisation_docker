#!/usr/bin/env python3
"""Compare seeded clustering methods for random localisation metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    from cluster_random_localisation_metrics import (
        CLUSTER_COLORS,
        DEFAULT_BANDWIDTH_DECADES,
        DEFAULT_BANDWIDTH_OBSERVATIONS,
        DEFAULT_CONVERGENCE_TOLERANCE,
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_MERGE_DISTANCE,
        DEFAULT_MIN_SUPPORT,
        NOISE_COLOR,
        OBSERVATION_SCALE,
        SEED_REFERENCE_COLORS,
        SEEDS,
        feature,
        load_points,
        mean_shift,
        observation_ticks,
    )
except ModuleNotFoundError:
    from scripts.cluster_random_localisation_metrics import (
        CLUSTER_COLORS,
        DEFAULT_BANDWIDTH_DECADES,
        DEFAULT_BANDWIDTH_OBSERVATIONS,
        DEFAULT_CONVERGENCE_TOLERANCE,
        DEFAULT_MAX_ITERATIONS,
        DEFAULT_MERGE_DISTANCE,
        DEFAULT_MIN_SUPPORT,
        NOISE_COLOR,
        OBSERVATION_SCALE,
        SEED_REFERENCE_COLORS,
        SEEDS,
        feature,
        load_points,
        mean_shift,
        observation_ticks,
    )

try:
    from celestial_detector.gmm_classifier import (
        GMM_MAX_ITERATIONS,
        GMM_REGULARIZATION,
        GMM_TOLERANCE,
        GmmModel,
        fit_gmm,
        save_gmm_boundaries,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "celestial_localisation" / "src" / "celestial_detector"))
    from celestial_detector.gmm_classifier import (
        GMM_MAX_ITERATIONS,
        GMM_REGULARIZATION,
        GMM_TOLERANCE,
        GmmModel,
        fit_gmm,
        save_gmm_boundaries,
    )


DEFAULT_CSV = ROOT / "celestial_localisation" / "debug_output" / "random_localisation_metrics.csv"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "docs"
DEFAULT_BOUNDARIES_OUTPUT = ROOT / "celestial_localisation" / "debug_output" / "gmm_boundaries.json"
COMPARISON_OUTPUT_FILENAMES = (
    "random_localisation_kmeans.png",
    "random_localisation_dbscan.png",
    "random_localisation_kde.png",
)
PNG_WIDTH = 720
PNG_HEIGHT = 1080
PLOT_LEFT = 98
PLOT_RIGHT = 680
PLOT_BOTTOM = 965
BACKGROUND = "#071216"
PLOT_BACKGROUND = "#0b1c20"
GRID_COLOR = "#1c3c3e"
AXIS_COLOR = "#5f7779"
MUTED_COLOR = "#91a8a8"
TEXT_COLOR = "#e8f1ee"
LEGEND_COLOR = "#b6cbc5"
DBSCAN_EPS = 0.55
DBSCAN_MIN_SAMPLES = 10

Color = tuple[int, int, int, int]
Point = dict[str, float | int]
Feature = tuple[float, float]


@dataclass
class AlgorithmResult:
    name: str
    filename: str
    parameters: str
    assignments: list[int]
    centers: list[Feature]


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
    ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    ",": ("00000", "00000", "00000", "00000", "00110", "00100", "01000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"),
    ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
}


def parse_color(value: str, alpha: int = 255) -> Color:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha)


def blend_pixel(pixels: bytearray, width: int, x: int, y: int, color: Color) -> None:
    if x < 0 or y < 0 or x >= width:
        return
    offset = (y * width + x) * 4
    if offset < 0 or offset + 3 >= len(pixels):
        return
    alpha = color[3] / 255.0
    if alpha >= 1.0:
        pixels[offset : offset + 4] = bytes(color)
        return
    inverse_alpha = 1.0 - alpha
    pixels[offset] = int(color[0] * alpha + pixels[offset] * inverse_alpha)
    pixels[offset + 1] = int(color[1] * alpha + pixels[offset + 1] * inverse_alpha)
    pixels[offset + 2] = int(color[2] * alpha + pixels[offset + 2] * inverse_alpha)
    pixels[offset + 3] = 255


def fill_rect(
    pixels: bytearray,
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: Color,
) -> None:
    for y in range(max(0, top), min(height, bottom + 1)):
        for x in range(max(0, left), min(width, right + 1)):
            blend_pixel(pixels, width, x, y, color)


def draw_line(
    pixels: bytearray,
    width: int,
    height: int,
    first: tuple[float, float],
    second: tuple[float, float],
    color: Color,
    thickness: int = 1,
    dash: int | None = None,
) -> None:
    x_start, y_start = first
    x_end, y_end = second
    distance = max(abs(x_end - x_start), abs(y_end - y_start), 1.0)
    steps = int(math.ceil(distance))
    for step in range(steps + 1):
        if dash is not None and (step // dash) % 2:
            continue
        progress = step / steps
        x = round(x_start + (x_end - x_start) * progress)
        y = round(y_start + (y_end - y_start) * progress)
        radius = max(0, thickness // 2)
        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                if 0 <= x + offset_x < width and 0 <= y + offset_y < height:
                    blend_pixel(pixels, width, x + offset_x, y + offset_y, color)


def draw_circle(
    pixels: bytearray,
    width: int,
    height: int,
    center: tuple[float, float],
    radius: float,
    color: Color,
) -> None:
    center_x, center_y = center
    integer_radius = math.ceil(radius)
    radius_squared = radius * radius
    for offset_y in range(-integer_radius, integer_radius + 1):
        for offset_x in range(-integer_radius, integer_radius + 1):
            if offset_x * offset_x + offset_y * offset_y <= radius_squared:
                x = round(center_x + offset_x)
                y = round(center_y + offset_y)
                if 0 <= x < width and 0 <= y < height:
                    blend_pixel(pixels, width, x, y, color)


def draw_text(
    pixels: bytearray,
    width: int,
    height: int,
    position: tuple[int, int],
    text: str,
    color: Color,
    scale: int = 2,
    anchor: str = "left",
) -> None:
    text = text.upper()
    character_width = 6 * scale
    total_width = max(0, len(text) * character_width - scale)
    x_start, y_start = position
    if anchor == "center":
        x_start -= total_width // 2
    elif anchor == "right":
        x_start -= total_width
    for character_index, character in enumerate(text):
        glyph = FONT.get(character)
        if glyph is None:
            continue
        x_offset = x_start + character_index * character_width
        for row, glyph_row in enumerate(glyph):
            for column, pixel in enumerate(glyph_row):
                if pixel == "1":
                    fill_rect(
                        pixels,
                        width,
                        height,
                        x_offset + column * scale,
                        y_start + row * scale,
                        x_offset + (column + 1) * scale - 1,
                        y_start + (row + 1) * scale - 1,
                        color,
                    )


def draw_vertical_text(
    pixels: bytearray,
    width: int,
    height: int,
    position: tuple[int, int],
    text: str,
    color: Color,
    scale: int = 2,
) -> None:
    character_height = 8 * scale
    for character_index, character in enumerate(text.upper()):
        draw_text(
            pixels,
            width,
            height,
            (position[0], position[1] + character_index * character_height),
            character,
            color,
            scale,
        )


def write_png(output_path: Path, pixels: bytearray, width: int, height: int) -> None:
    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    scanlines = bytearray()
    row_size = width * 4
    for row in range(height):
        scanlines.append(0)
        start = row * row_size
        scanlines.extend(pixels[start : start + row_size])
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6)))
    png.extend(chunk(b"IEND", b""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)


def squared_distance(first: Feature, second: Feature) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def seed_features() -> list[Feature]:
    return [
        (observations / OBSERVATION_SCALE, math.log10(error_km))
        for observations, error_km, _ in SEEDS
    ]


def kmeans(
    points: list[Point],
    max_iterations: int = 100,
    convergence_tolerance: float = 1e-5,
) -> tuple[list[int], list[Feature]]:
    """Run Lloyd's algorithm from the four supplied seed locations."""
    features = [feature(point) for point in points]
    centers = seed_features()
    assignments = [-1] * len(features)
    for _ in range(max_iterations):
        new_assignments = [
            min(range(len(centers)), key=lambda index: squared_distance(value, centers[index]))
            for value in features
        ]
        new_centers: list[Feature] = []
        for cluster_index, center in enumerate(centers):
            members = [
                value for value, assignment in zip(features, new_assignments)
                if assignment == cluster_index
            ]
            if not members:
                new_centers.append(center)
                continue
            new_centers.append(
                (
                    sum(value[0] for value in members) / len(members),
                    sum(value[1] for value in members) / len(members),
                )
            )
        center_shift = max(
            squared_distance(old, new) for old, new in zip(centers, new_centers)
        )
        centers = new_centers
        if new_assignments == assignments or center_shift <= convergence_tolerance**2:
            assignments = new_assignments
            break
        assignments = new_assignments
    return assignments, centers


def spatial_neighbors(features: list[Feature], radius: float) -> list[list[int]]:
    cell_size = radius
    spatial_index: dict[tuple[int, int], list[int]] = {}
    for index, value in enumerate(features):
        cell = (math.floor(value[0] / cell_size), math.floor(value[1] / cell_size))
        spatial_index.setdefault(cell, []).append(index)

    neighbours: list[list[int]] = []
    for value in features:
        cell = (math.floor(value[0] / cell_size), math.floor(value[1] / cell_size))
        candidates: list[int] = []
        for cell_x in range(cell[0] - 1, cell[0] + 2):
            for cell_y in range(cell[1] - 1, cell[1] + 2):
                candidates.extend(spatial_index.get((cell_x, cell_y), ()))
        neighbours.append(
            [index for index in candidates if squared_distance(value, features[index]) <= radius**2]
        )
    return neighbours


def dbscan(
    points: list[Point],
    radius: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> list[int]:
    """Cluster connected dense regions with a grid-backed DBSCAN search."""
    if radius <= 0 or min_samples < 1:
        raise ValueError("DBSCAN radius must be positive and min_samples must be at least one")
    features = [feature(point) for point in points]
    neighbours = spatial_neighbors(features, radius)
    assignments = [-1] * len(features)
    visited = [False] * len(features)
    cluster_index = 0
    for point_index in range(len(features)):
        if visited[point_index]:
            continue
        visited[point_index] = True
        point_neighbours = neighbours[point_index]
        if len(point_neighbours) < min_samples:
            continue
        assignments[point_index] = cluster_index
        queue = list(point_neighbours)
        queue_position = 0
        queued = set(queue)
        while queue_position < len(queue):
            neighbour_index = queue[queue_position]
            queue_position += 1
            if not visited[neighbour_index]:
                visited[neighbour_index] = True
                neighbour_neighbours = neighbours[neighbour_index]
                if len(neighbour_neighbours) >= min_samples:
                    for candidate in neighbour_neighbours:
                        if candidate not in queued:
                            queue.append(candidate)
                            queued.add(candidate)
            if assignments[neighbour_index] == -1:
                assignments[neighbour_index] = cluster_index
        cluster_index += 1
    return assignments


def gaussian_mixture(
    points: list[Point],
    max_iterations: int = GMM_MAX_ITERATIONS,
    convergence_tolerance: float = GMM_TOLERANCE,
    regularization: float = GMM_REGULARIZATION,
) -> tuple[list[int], list[Feature], GmmModel]:
    """Fit the shared four-component model and return zero-based plot ids."""
    model = fit_gmm(
        [
            (float(point["observations"]), float(point["error_km"]))
            for point in points
        ],
        max_iterations=max_iterations,
        convergence_tolerance=convergence_tolerance,
        regularization=regularization,
    )
    assignments = [
        model.classify(float(point["observations"]), float(point["error_km"])) - 1
        for point in points
    ]
    return assignments, [component.mean for component in model.components], model


def format_axis_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def format_count(value: int) -> str:
    return f"{value:,}"


def render_result(
    output_path: Path,
    points: list[Point],
    result: AlgorithmResult,
) -> None:
    width = PNG_WIDTH
    height = PNG_HEIGHT
    pixels = bytearray(parse_color(BACKGROUND) * (width * height))
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

    cluster_ids = sorted({assignment for assignment in result.assignments if assignment >= 0})
    color_by_cluster = {
        cluster_id: parse_color(CLUSTER_COLORS[index % len(CLUSTER_COLORS)])
        for index, cluster_id in enumerate(cluster_ids)
    }
    members_by_cluster = {
        cluster_id: sum(assignment == cluster_id for assignment in result.assignments)
        for cluster_id in cluster_ids
    }
    noise_count = sum(assignment == -1 for assignment in result.assignments)
    legend_items = [
        (f"CLUSTER {cluster_id + 1}", members_by_cluster[cluster_id], color_by_cluster[cluster_id])
        for cluster_id in cluster_ids
    ]
    if noise_count:
        legend_items.append(("NOISE / OUTLIER", noise_count, parse_color(NOISE_COLOR)))
    legend_rows = max(1, (len(legend_items) + 2) // 3)
    plot_top = 135 + (legend_rows - 1) * 24

    def scale_x(observations: float) -> float:
        return PLOT_LEFT + (observations / maximum_observations) * (PLOT_RIGHT - PLOT_LEFT)

    def scale_y(error_km: float) -> float:
        return PLOT_BOTTOM - (
            (math.log10(max(error_km, 10**minimum_exponent)) - minimum_exponent)
            / (maximum_exponent - minimum_exponent)
        ) * (PLOT_BOTTOM - plot_top)

    draw_text(
        pixels,
        width,
        height,
        (PLOT_LEFT, 28),
        f"OBSERVATIONS VS ERROR / {result.name}",
        parse_color(TEXT_COLOR),
        scale=3 if len(result.name) <= 8 else 2,
    )
    draw_text(
        pixels,
        width,
        height,
        (PLOT_LEFT, 64),
        result.parameters,
        parse_color(MUTED_COLOR),
        scale=2,
    )
    for item_index, (name, count, color) in enumerate(legend_items):
        column = item_index % 3
        row = item_index // 3
        x = PLOT_LEFT + column * 195
        y = 96 + row * 24
        draw_circle(pixels, width, height, (x + 6, y + 7), 5, color)
        draw_text(
            pixels,
            width,
            height,
            (x + 17, y),
            f"{name} N={format_count(count)}",
            parse_color(LEGEND_COLOR),
            scale=1,
        )

    fill_rect(
        pixels,
        width,
        height,
        PLOT_LEFT,
        plot_top,
        PLOT_RIGHT,
        PLOT_BOTTOM,
        parse_color(PLOT_BACKGROUND),
    )
    draw_line(
        pixels,
        width,
        height,
        (PLOT_LEFT, plot_top),
        (PLOT_RIGHT, plot_top),
        parse_color(GRID_COLOR),
    )
    draw_line(
        pixels,
        width,
        height,
        (PLOT_RIGHT, plot_top),
        (PLOT_RIGHT, PLOT_BOTTOM),
        parse_color(GRID_COLOR),
    )
    for value in observation_ticks(maximum_observations):
        x = scale_x(value)
        draw_line(
            pixels,
            width,
            height,
            (x, plot_top),
            (x, PLOT_BOTTOM),
            parse_color(GRID_COLOR, 170),
        )
        draw_text(
            pixels,
            width,
            height,
            (round(x), PLOT_BOTTOM + 18),
            format_axis_number(value),
            parse_color(AXIS_COLOR),
            scale=2,
            anchor="center",
        )
    for exponent in range(minimum_exponent, maximum_exponent + 1):
        error_km = 10**exponent
        y = scale_y(error_km)
        draw_line(
            pixels,
            width,
            height,
            (PLOT_LEFT, y),
            (PLOT_RIGHT, y),
            parse_color(GRID_COLOR, 170),
        )
        draw_text(
            pixels,
            width,
            height,
            (PLOT_LEFT - 12, round(y) - 7),
            format_axis_number(error_km),
            parse_color(AXIS_COLOR),
            scale=2,
            anchor="right",
        )

    for point, assignment in zip(points, result.assignments):
        point_color = parse_color(NOISE_COLOR) if assignment == -1 else color_by_cluster[assignment]
        draw_circle(
            pixels,
            width,
            height,
            (scale_x(float(point["observations"])), scale_y(float(point["error_km"]))),
            2.2,
            (point_color[0], point_color[1], point_color[2], 190),
        )

    for center_index, center in enumerate(result.centers):
        color = parse_color(CLUSTER_COLORS[center_index % len(CLUSTER_COLORS)])
        center_x = scale_x(center[0] * OBSERVATION_SCALE)
        center_y = scale_y(10**center[1])
        draw_line(
            pixels,
            width,
            height,
            (center_x - 9, center_y),
            (center_x + 9, center_y),
            parse_color(BACKGROUND),
            thickness=3,
        )
        draw_line(
            pixels,
            width,
            height,
            (center_x, center_y - 9),
            (center_x, center_y + 9),
            parse_color(BACKGROUND),
            thickness=3,
        )
        draw_circle(pixels, width, height, (center_x, center_y), 5, color)
        draw_line(
            pixels,
            width,
            height,
            (center_x - 8, center_y),
            (center_x + 8, center_y),
            color,
        )
        draw_line(
            pixels,
            width,
            height,
            (center_x, center_y - 8),
            (center_x, center_y + 8),
            color,
        )

    for (seed_observations, seed_error, _), color_name in zip(SEEDS, SEED_REFERENCE_COLORS):
        seed_x = scale_x(seed_observations)
        seed_y = scale_y(seed_error)
        color = parse_color(color_name)
        draw_line(
            pixels,
            width,
            height,
            (seed_x - 7, seed_y),
            (seed_x + 7, seed_y),
            color,
            dash=3,
        )
        draw_line(
            pixels,
            width,
            height,
            (seed_x, seed_y - 7),
            (seed_x, seed_y + 7),
            color,
            dash=3,
        )

    draw_text(
        pixels,
        width,
        height,
        ((PLOT_LEFT + PLOT_RIGHT) // 2, PLOT_BOTTOM + 56),
        "OBSERVATIONS USED",
        parse_color(MUTED_COLOR),
        scale=2,
        anchor="center",
    )
    draw_vertical_text(
        pixels,
        width,
        height,
        (18, plot_top + 90),
        "ERROR KM LOG",
        parse_color(MUTED_COLOR),
        scale=2,
    )
    draw_text(
        pixels,
        width,
        height,
        (PLOT_LEFT, PLOT_BOTTOM + 91),
        "DOTS ARE RUNS / CROSSES ARE ESTIMATED CENTERS / DASHED MARKS ARE INITIAL SEEDS",
        parse_color(AXIS_COLOR),
        scale=1,
    )
    write_png(output_path, pixels, width, height)


def build_results(
    points: list[Point],
    dbscan_radius: float,
    dbscan_min_samples: int,
    kde_bandwidth_observations: float,
    kde_bandwidth_decades: float,
    kde_min_support: int,
    kde_merge_distance: float,
    include_comparison: bool = True,
) -> tuple[list[AlgorithmResult], GmmModel]:
    results: list[AlgorithmResult] = []
    if include_comparison:
        kmeans_assignments, kmeans_centers = kmeans(points)
        results.append(
            AlgorithmResult(
                "K MEANS",
                "random_localisation_kmeans.png",
                "4 CENTERS / SEEDED INITIALIZATION",
                kmeans_assignments,
                kmeans_centers,
            )
        )

        dbscan_assignments = dbscan(points, dbscan_radius, dbscan_min_samples)
        dbscan_centers = []
        for cluster_index in sorted({assignment for assignment in dbscan_assignments if assignment >= 0}):
            members = [
                feature(point)
                for point, assignment in zip(points, dbscan_assignments)
                if assignment == cluster_index
            ]
            dbscan_centers.append(
                (
                    sum(value[0] for value in members) / len(members),
                    sum(value[1] for value in members) / len(members),
                )
            )
        results.append(
            AlgorithmResult(
                "DBSCAN",
                "random_localisation_dbscan.png",
                f"EPS={dbscan_radius:g} / MIN SAMPLES={dbscan_min_samples}",
                dbscan_assignments,
                dbscan_centers,
            )
        )

        kde_assignments, _, kde_peaks, _ = mean_shift(
            points,
            kde_bandwidth_observations,
            kde_bandwidth_decades,
            kde_min_support,
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_CONVERGENCE_TOLERANCE,
            kde_merge_distance,
        )
        results.append(
            AlgorithmResult(
                "KDE / MEAN SHIFT",
                "random_localisation_kde.png",
                f"BANDWIDTH={kde_bandwidth_observations:g} OBS / {kde_bandwidth_decades:g} DECADES",
                kde_assignments,
                [(peak[0], peak[1]) for peak in kde_peaks],
            )
        )

    gmm_assignments, gmm_centers, gmm_model = gaussian_mixture(points)
    results.append(
        AlgorithmResult(
            "GAUSSIAN MIXTURE",
            "random_localisation_gaussian_mixture.png",
            "4 COMPONENTS / SEEDED MEANS / EM",
            gmm_assignments,
            gmm_centers,
        ),
    )
    return results, gmm_model


def annotate_metrics_csv(csv_path: Path, model: GmmModel) -> int:
    """Backfill cluster ids for existing rows using the persisted model."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        if "cluster_id" not in fieldnames:
            insert_at = fieldnames.index("identified_objects_used") + 1
            fieldnames.insert(insert_at, "cluster_id")
        rows = list(reader)

    annotated_count = 0
    for row in rows:
        try:
            observations = float(row["identified_objects_used"])
            error_km = float(row["error_distance_meters"]) / 1000.0
            row["cluster_id"] = str(model.classify(observations, error_km))
            annotated_count += 1
        except (KeyError, TypeError, ValueError):
            row["cluster_id"] = ""

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
        dir=csv_path.parent,
    )
    try:
        with os.fdopen(temporary_fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, csv_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return annotated_count


def remove_comparison_outputs(output_dir: Path) -> None:
    for filename in COMPARISON_OUTPUT_FILENAMES:
        try:
            (output_dir / filename).unlink()
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and render the seeded Gaussian mixture; use --all-methods for "
            "the older algorithm comparison."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"directory for PNG outputs (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--boundaries-output",
        type=Path,
        default=DEFAULT_BOUNDARIES_OUTPUT,
        help=f"GMM boundary JSON path (default: {DEFAULT_BOUNDARIES_OUTPUT})",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=DBSCAN_EPS,
        help=f"DBSCAN feature-space radius (default: {DBSCAN_EPS})",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=DBSCAN_MIN_SAMPLES,
        help=f"DBSCAN core-point threshold (default: {DBSCAN_MIN_SAMPLES})",
    )
    parser.add_argument(
        "--kde-bandwidth-observations",
        type=float,
        default=DEFAULT_BANDWIDTH_OBSERVATIONS,
        help=f"KDE observation bandwidth (default: {DEFAULT_BANDWIDTH_OBSERVATIONS})",
    )
    parser.add_argument(
        "--kde-bandwidth-decades",
        type=float,
        default=DEFAULT_BANDWIDTH_DECADES,
        help=f"KDE log-error bandwidth (default: {DEFAULT_BANDWIDTH_DECADES})",
    )
    parser.add_argument(
        "--kde-min-support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help=f"minimum KDE peak support (default: {DEFAULT_MIN_SUPPORT})",
    )
    parser.add_argument(
        "--kde-merge-distance",
        type=float,
        default=DEFAULT_MERGE_DISTANCE,
        help=f"KDE peak merge distance (default: {DEFAULT_MERGE_DISTANCE})",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="also regenerate the K-means, DBSCAN, and KDE comparison PNGs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_methods:
        remove_comparison_outputs(args.output_dir)
    points = load_points(args.csv_path)
    results, gmm_model = build_results(
        points,
        args.dbscan_eps,
        args.dbscan_min_samples,
        args.kde_bandwidth_observations,
        args.kde_bandwidth_decades,
        args.kde_min_support,
        args.kde_merge_distance,
        args.all_methods,
    )
    save_gmm_boundaries(gmm_model, args.boundaries_output)
    annotated_count = annotate_metrics_csv(args.csv_path, gmm_model)
    print(f"Input: {args.csv_path}")
    print(f"Valid runs: {len(points)}")
    print(f"GMM boundaries: {args.boundaries_output}")
    print(f"CSV cluster ids: {annotated_count}")
    for result in results:
        output_path = args.output_dir / result.filename
        render_result(output_path, points, result)
        cluster_count = len({assignment for assignment in result.assignments if assignment >= 0})
        noise_count = sum(assignment == -1 for assignment in result.assignments)
        print(
            f"{result.name}: clusters={cluster_count} | noise/outliers={noise_count} "
            f"| output={output_path}"
        )


if __name__ == "__main__":
    main()