#!/usr/bin/env python3
"""Render one random-localisation run from the metrics CSV as KML."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = (
    ROOT
    / "celestial_localisation"
    / "debug_output"
    / "random_localisation_metrics.csv"
)
KML_NAMESPACE = "http://www.opengis.net/kml/2.2"
ORANGE_ICON_URL = "http://maps.google.com/mapfiles/kml/pushpin/orange-pushpin.png"
BLUE_ICON_URL = "http://maps.google.com/mapfiles/kml/pushpin/blue-pushpin.png"
ElementTree.register_namespace("", KML_NAMESPACE)


def _tag(name: str) -> str:
    return f"{{{KML_NAMESPACE}}}{name}"


def _add_text(parent, name: str, text: str):
    element = ElementTree.SubElement(parent, _tag(name))
    element.text = text
    return element


def _parse_float(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} on CSV row {row_number}") from error
    if not math.isfinite(value):
        raise ValueError(f"Invalid {field} on CSV row {row_number}: value is not finite")
    return value


def _parse_coordinate(row: dict[str, str], field: str, row_number: int, minimum: float, maximum: float) -> float:
    value = _parse_float(row, field, row_number)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"Invalid {field} on CSV row {row_number}: expected {minimum} to {maximum}"
        )
    return value


def _parse_record(row: dict[str, str], row_number: int, require_repetition_index: bool) -> dict[str, float | int | str]:
    record = {
        "latitude": _parse_coordinate(row, "estimated_latitude", row_number, -90.0, 90.0),
        "longitude": _parse_coordinate(row, "estimated_longitude", row_number, -180.0, 180.0),
        "ground_truth_latitude": _parse_coordinate(
            row, "ground_truth_latitude", row_number, -90.0, 90.0
        ),
        "ground_truth_longitude": _parse_coordinate(
            row, "ground_truth_longitude", row_number, -180.0, 180.0
        ),
        "record_type": (row.get("record_type") or "").strip().lower(),
        "is_outlier": (row.get("is_outlier") or "").strip().lower(),
    }
    if require_repetition_index:
        try:
            repetition_index = int(row["repetition_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid repetition_index on CSV row {row_number}") from error
        if repetition_index < 1:
            raise ValueError(f"Invalid repetition_index on CSV row {row_number}")
        record["repetition_index"] = repetition_index
    return record


def load_run_records(
    csv_path: Path, run_index: int
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    """Load the ordered step records and summary record for one run."""
    if run_index < 1:
        raise ValueError("run_index must be greater than zero")

    required_columns = {
        "run_index",
        "record_type",
        "repetition_index",
        "ground_truth_latitude",
        "ground_truth_longitude",
        "estimated_latitude",
        "estimated_longitude",
    }
    steps: list[dict[str, float | int | str]] = []
    summary = None
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                row_run_index = int(row["run_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid run_index on CSV row {row_number}") from error
            if row_run_index != run_index:
                continue

            record_type = (row.get("record_type") or "").strip().lower()
            if record_type == "step":
                steps.append(_parse_record(row, row_number, True))
            elif record_type == "summary":
                if summary is not None:
                    raise ValueError(f"Run {run_index} contains more than one summary row")
                summary = _parse_record(row, row_number, False)
            else:
                raise ValueError(
                    f"Run {run_index} has unsupported record_type on CSV row {row_number}: "
                    f"{record_type or '<empty>'}"
                )

    if not steps:
        raise ValueError(f"Run {run_index} contains no step rows")
    if summary is None:
        raise ValueError(f"Run {run_index} contains no summary row")
    steps.sort(key=lambda record: int(record["repetition_index"]))
    return steps, summary


def _coordinates(record: dict[str, float | int | str]) -> str:
    return f"{float(record['longitude']):.12f},{float(record['latitude']):.12f},0"


def _add_icon_style(document, style_id: str, color: str, scale: str, icon_url: str) -> None:
    style = ElementTree.SubElement(document, _tag("Style"), {"id": style_id})
    icon_style = ElementTree.SubElement(style, _tag("IconStyle"))
    _add_text(icon_style, "color", color)
    _add_text(icon_style, "scale", scale)
    icon = ElementTree.SubElement(icon_style, _tag("Icon"))
    _add_text(icon, "href", icon_url)


def _add_placemark(
    parent,
    name: str,
    style_id: str,
    record: dict[str, float | int | str],
    description: str,
) -> None:
    placemark = ElementTree.SubElement(parent, _tag("Placemark"))
    _add_text(placemark, "name", name)
    _add_text(placemark, "description", description)
    _add_text(placemark, "styleUrl", f"#{style_id}")
    point = ElementTree.SubElement(placemark, _tag("Point"))
    _add_text(point, "coordinates", _coordinates(record))


def _add_path(parent, run_index: int, steps: list[dict[str, float | int | str]]) -> None:
    if len(steps) < 2:
        return
    placemark = ElementTree.SubElement(parent, _tag("Placemark"))
    _add_text(placemark, "name", f"Run {run_index} estimated step path")
    _add_text(placemark, "styleUrl", "#estimated-path")
    line_string = ElementTree.SubElement(placemark, _tag("LineString"))
    _add_text(line_string, "tessellate", "1")
    _add_text(line_string, "coordinates", "\n".join(_coordinates(step) for step in steps))


def write_kml(
    output_path: Path,
    run_index: int,
    steps: list[dict[str, float | int | str]],
    summary: dict[str, float | int | str],
) -> None:
    """Write step estimates, summary fixes, and the estimated path to KML."""
    root = ElementTree.Element(_tag("kml"))
    document = ElementTree.SubElement(root, _tag("Document"))
    _add_text(document, "name", f"Random localisation run {run_index}")
    _add_text(
        document,
        "description",
        "Orange markers show estimated fixes. The larger translucent summary "
        "markers show the summary estimate and ground truth.",
    )
    _add_icon_style(document, "estimated-step", "ff0066ff", "0.8", ORANGE_ICON_URL)
    _add_icon_style(document, "estimated-summary", "990066ff", "1.8", ORANGE_ICON_URL)
    _add_icon_style(document, "ground-truth-summary", "99ff6600", "1.8", BLUE_ICON_URL)
    path_style = ElementTree.SubElement(document, _tag("Style"), {"id": "estimated-path"})
    line_style = ElementTree.SubElement(path_style, _tag("LineStyle"))
    _add_text(line_style, "color", "cc0066ff")
    _add_text(line_style, "width", "4")

    estimated_folder = ElementTree.SubElement(document, _tag("Folder"))
    _add_text(estimated_folder, "name", "Estimated fixes")
    for step in steps:
        outlier_suffix = " (outlier)" if step["is_outlier"] == "true" else ""
        repetition_index = int(step["repetition_index"])
        _add_placemark(
            estimated_folder,
            f"Run {run_index} step {repetition_index}{outlier_suffix}",
            "estimated-step",
            step,
            f"Estimated fix for repetition {repetition_index}: "
            f"{float(step['latitude']):.8f}, {float(step['longitude']):.8f}",
        )
    _add_placemark(
        estimated_folder,
        f"Run {run_index} summary estimated fix",
        "estimated-summary",
        summary,
        f"Summary estimated fix: {float(summary['latitude']):.8f}, "
        f"{float(summary['longitude']):.8f}",
    )

    ground_truth_folder = ElementTree.SubElement(document, _tag("Folder"))
    _add_text(ground_truth_folder, "name", "Ground truth")
    ground_truth_record = {
        "latitude": summary["ground_truth_latitude"],
        "longitude": summary["ground_truth_longitude"],
    }
    _add_placemark(
        ground_truth_folder,
        f"Run {run_index} summary ground truth",
        "ground-truth-summary",
        ground_truth_record,
        f"Summary ground truth: {float(summary['ground_truth_latitude']):.8f}, "
        f"{float(summary['ground_truth_longitude']):.8f}",
    )

    path_folder = ElementTree.SubElement(document, _tag("Folder"))
    _add_text(path_folder, "name", "Estimated path")
    _add_path(path_folder, run_index, steps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ElementTree.ElementTree(root)
    ElementTree.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one random-localisation run from the metrics CSV as KML."
    )
    parser.add_argument("run_index", type=int, help="run index to render")
    parser.add_argument(
        "--csv-path",
        "--csv",
        dest="csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"metrics CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        help="KML output path (default: alongside the CSV as random_localisation_run_NNNNNN.kml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps, summary = load_run_records(args.csv_path, args.run_index)
    output_path = args.output_path or (
        args.csv_path.parent / f"random_localisation_run_{args.run_index:06d}.kml"
    )
    write_kml(output_path, args.run_index, steps, summary)
    print(f"Input: {args.csv_path}")
    print(f"Run: {args.run_index}")
    print(f"Step fixes: {len(steps)}")
    print(f"KML: {output_path}")


if __name__ == "__main__":
    main()