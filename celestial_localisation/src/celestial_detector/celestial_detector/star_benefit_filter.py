"""Identify catalogue stars associated with poor historical localisation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


HIGH_ERROR_GROUP_ID = 3
MEDIUM_ERROR_GROUP_ID = 2


@dataclass(frozen=True)
class StarBenefitMetric:
    star_id: str
    group_id: Optional[int]
    present_count: int
    absent_count: int
    present_mean_km: float
    absent_mean_km: float
    benefit_score: float


@dataclass(frozen=True)
class StarBenefitFilter:
    excluded_star_ids: frozenset[str]
    metrics: tuple[StarBenefitMetric, ...]
    source_paths: tuple[Path, ...] = ()

    @classmethod
    def empty(cls):
        return cls(frozenset(), ())

    @classmethod
    def from_csv(
        cls,
        path: Path,
        min_benefit_score: float = -0.1,
        min_present_count: int = 3,
        min_absent_count: int = 2,
        use_high_error_group: bool = True,
        use_medium_error_group: bool = False,
        record_type: str = 'step',
    ):
        source_paths = _expand_csv_paths(path)
        rows = [
            row
            for source_path in source_paths
            for row in _load_rows(source_path, record_type)
        ]
        metrics = tuple(calculate_star_benefit(
            rows,
            min_present_count=min_present_count,
            min_absent_count=min_absent_count,
            group_ids=(
                HIGH_ERROR_GROUP_ID if use_high_error_group else None,
                MEDIUM_ERROR_GROUP_ID if use_medium_error_group else None,
            ),
        ))
        excluded_star_ids = frozenset(
            metric.star_id
            for metric in metrics
            if metric.benefit_score <= float(min_benefit_score)
        )
        return cls(excluded_star_ids, metrics, tuple(source_paths))

    def excludes(self, star_id: str) -> bool:
        return _normalise_star_id(star_id) in self.excluded_star_ids


@dataclass(frozen=True)
class _MetricRow:
    error_km: float
    star_ids: frozenset[str]
    group_id: Optional[int]


def calculate_star_benefit(
    rows: Iterable[_MetricRow],
    min_present_count: int = 2,
    min_absent_count: int = 2,
    group_ids: Optional[Iterable[Optional[int]]] = None,
) -> list[StarBenefitMetric]:
    """Compare each star's error when present and absent within each group."""
    rows = list(rows)
    if min_present_count < 1 or min_absent_count < 1:
        raise ValueError('minimum star benefit counts must be positive')

    cluster_ids = {row.group_id for row in rows if row.group_id is not None}
    if group_ids is None:
        comparison_group_ids = sorted(cluster_ids) or [None]
    else:
        comparison_group_ids = [
            group_id
            for group_id in dict.fromkeys(group_ids)
            if group_id in cluster_ids
        ]

    star_ids = sorted({star_id for row in rows for star_id in row.star_ids})
    metrics = []
    for star_id in star_ids:
        for group_id in comparison_group_ids:
            group_rows = [row for row in rows if row.group_id == group_id]
            present_errors = [row.error_km for row in group_rows if star_id in row.star_ids]
            absent_errors = [row.error_km for row in group_rows if star_id not in row.star_ids]
            if len(present_errors) < min_present_count or len(absent_errors) < min_absent_count:
                continue

            present_mean = sum(present_errors) / len(present_errors)
            absent_mean = sum(absent_errors) / len(absent_errors)
            benefit_score = math.log10((absent_mean + 1.0) / (present_mean + 1.0))
            if not math.isfinite(benefit_score):
                continue
            metrics.append(StarBenefitMetric(
                star_id=star_id,
                group_id=group_id,
                present_count=len(present_errors),
                absent_count=len(absent_errors),
                present_mean_km=present_mean,
                absent_mean_km=absent_mean,
                benefit_score=benefit_score,
            ))
    return sorted(metrics, key=lambda metric: (
        metric.benefit_score,
        -metric.present_count,
        metric.star_id,
    ))


def _load_rows(path: Path, record_type: str) -> list[_MetricRow]:
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        has_record_type = 'record_type' in fieldnames
        rows = []
        for row in reader:
            if (
                has_record_type
                and record_type
                and row.get('record_type', '').strip().lower() != record_type.lower()
            ):
                continue
            error_meters = _parse_float(row.get('error_distance_meters'))
            if error_meters is None or error_meters < 0.0:
                continue
            rows.append(_MetricRow(
                error_km=error_meters / 1000.0,
                star_ids=_parse_star_ids(row.get('identified_star_ids')),
                group_id=_parse_group_id(row.get('cluster_id')),
            ))
    return rows


def _expand_csv_paths(path: Path) -> list[Path]:
    path = Path(path)
    if not any(character in path.name for character in '*?['):
        return [path]

    matching_paths = sorted(path.parent.glob(path.name))
    if not matching_paths:
        raise FileNotFoundError(f'no CSV files match star benefit pattern: {path}')
    return matching_paths


def _parse_float(value) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_group_id(value) -> Optional[int]:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


def _parse_star_ids(value) -> frozenset[str]:
    if not value:
        return frozenset()
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return frozenset()
    if not isinstance(parsed, list):
        return frozenset()
    star_ids = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        normalised = _normalise_star_id(item)
        if normalised:
            star_ids.add(normalised)
    return frozenset(star_ids)


def _normalise_star_id(star_id: str) -> str:
    return str(star_id).strip().upper()