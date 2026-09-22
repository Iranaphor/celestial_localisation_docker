"""Shared Gaussian-mixture fitting, persistence, and sample classification."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
OBSERVATION_SCALE = 10.0
GMM_MAX_ITERATIONS = 100
GMM_TOLERANCE = 1e-4
GMM_REGULARIZATION = 0.02
CLUSTER_COLORS = ("#94e3c4", "#e4d47b", "#ffb65c", "#76c9d9")
CLUSTER_NAMES = ("lowerror", "mediumerror", "higherror", "lowsamples")
SEEDS = (
    (25.0, 10.0, "low error"),
    (30.0, 100.0, "medium error"),
    (15.0, 10_000.0, "high error"),
    (0.0, 10_000.0, "zero observations"),
)

Feature = tuple[float, float]
Sample = tuple[float, float]
Covariance = tuple[float, float, float]


@dataclass(frozen=True)
class GmmComponent:
    cluster_id: int
    weight: float
    mean: Feature
    covariance: Covariance
    color: str


@dataclass(frozen=True)
class GmmModel:
    components: tuple[GmmComponent, ...]

    def classify(self, observations: float, error_km: float) -> int:
        value = feature(observations, error_km)
        return max(
            self.components,
            key=lambda component: _component_log_score(component, value),
        ).cluster_id

    def classify_observations(self, observations: float) -> int:
        """Classify a live detector sample using the x-axis marginal only."""
        value = float(observations) / OBSERVATION_SCALE
        return max(
            self.components,
            key=lambda component: _marginal_observation_log_score(component, value),
        ).cluster_id

    def scores(self, observations: float, error_km: float) -> dict[int, float]:
        value = feature(observations, error_km)
        return {
            component.cluster_id: _component_log_score(component, value)
            for component in self.components
        }


def feature(observations: float, error_km: float) -> Feature:
    observations = float(observations)
    error_km = float(error_km)
    if not math.isfinite(observations) or observations < 0:
        raise ValueError("observations must be finite and non-negative")
    if not math.isfinite(error_km) or error_km <= 0:
        raise ValueError("error_km must be finite and greater than zero")
    return observations / OBSERVATION_SCALE, math.log10(error_km)


def _seed_features() -> list[Feature]:
    return [feature(observations, error_km) for observations, error_km, _ in SEEDS]


def _squared_distance(first: Feature, second: Feature) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _log_gaussian(value: Feature, mean: Feature, covariance: Covariance) -> float:
    variance_x, covariance_xy, variance_y = covariance
    determinant = variance_x * variance_y - covariance_xy * covariance_xy
    if determinant <= 0:
        raise ValueError("Gaussian covariance matrix is not positive definite")
    difference_x = value[0] - mean[0]
    difference_y = value[1] - mean[1]
    mahalanobis = (
        variance_y * difference_x * difference_x
        - 2.0 * covariance_xy * difference_x * difference_y
        + variance_x * difference_y * difference_y
    ) / determinant
    return -0.5 * (2.0 * math.log(2.0 * math.pi) + math.log(determinant) + mahalanobis)


def _component_log_score(component: GmmComponent, value: Feature) -> float:
    return math.log(max(component.weight, 1e-12)) + _log_gaussian(
        value,
        component.mean,
        component.covariance,
    )


def _marginal_observation_log_score(component: GmmComponent, value: float) -> float:
    variance_x = component.covariance[0]
    difference = value - component.mean[0]
    return math.log(max(component.weight, 1e-12)) - 0.5 * (
        math.log(2.0 * math.pi * variance_x) + difference * difference / variance_x
    )


def fit_gmm(
    samples: Iterable[Sample],
    max_iterations: int = GMM_MAX_ITERATIONS,
    convergence_tolerance: float = GMM_TOLERANCE,
    regularization: float = GMM_REGULARIZATION,
) -> GmmModel:
    """Fit four full-covariance components from the fixed project seeds."""
    if max_iterations < 1 or convergence_tolerance <= 0 or regularization <= 0:
        raise ValueError("invalid Gaussian-mixture fitting settings")

    features = [feature(observations, error_km) for observations, error_km in samples]
    if not features:
        raise ValueError("cannot fit a Gaussian mixture without samples")

    means = _seed_features()
    component_count = len(means)
    mean_x = sum(value[0] for value in features) / len(features)
    mean_y = sum(value[1] for value in features) / len(features)
    variance_x = max(
        sum((value[0] - mean_x) ** 2 for value in features) / len(features),
        regularization,
    )
    variance_y = max(
        sum((value[1] - mean_y) ** 2 for value in features) / len(features),
        regularization,
    )
    covariance: list[Covariance] = [
        (variance_x + regularization, 0.0, variance_y + regularization)
    ] * component_count
    weights = [1.0 / component_count] * component_count

    for _ in range(max_iterations):
        responsibilities: list[list[float]] = []
        for value in features:
            log_probabilities = [
                math.log(max(weight, 1e-12)) + _log_gaussian(value, mean, covariance_value)
                for weight, mean, covariance_value in zip(weights, means, covariance)
            ]
            maximum_log_probability = max(log_probabilities)
            unnormalized = [
                math.exp(log_probability - maximum_log_probability)
                for log_probability in log_probabilities
            ]
            normalization = sum(unnormalized)
            responsibilities.append([probability / normalization for probability in unnormalized])

        effective_counts = [
            sum(row[component_index] for row in responsibilities)
            for component_index in range(component_count)
        ]
        new_weights = [count / len(features) for count in effective_counts]
        new_means: list[Feature] = []
        new_covariance: list[Covariance] = []
        for component_index, effective_count in enumerate(effective_counts):
            if effective_count <= 1e-9:
                new_means.append(means[component_index])
                new_covariance.append(covariance[component_index])
                continue
            new_mean = (
                sum(row[component_index] * value[0] for row, value in zip(responsibilities, features))
                / effective_count,
                sum(row[component_index] * value[1] for row, value in zip(responsibilities, features))
                / effective_count,
            )
            variance_component_x = (
                sum(
                    row[component_index] * (value[0] - new_mean[0]) ** 2
                    for row, value in zip(responsibilities, features)
                )
                / effective_count
                + regularization
            )
            covariance_component_xy = (
                sum(
                    row[component_index]
                    * (value[0] - new_mean[0])
                    * (value[1] - new_mean[1])
                    for row, value in zip(responsibilities, features)
                )
                / effective_count
            )
            variance_component_y = (
                sum(
                    row[component_index] * (value[1] - new_mean[1]) ** 2
                    for row, value in zip(responsibilities, features)
                )
                / effective_count
                + regularization
            )
            new_means.append(new_mean)
            new_covariance.append(
                (variance_component_x, covariance_component_xy, variance_component_y)
            )

        movement = max(
            _squared_distance(old, new)
            for old, new in zip(means, new_means)
        )
        means = new_means
        covariance = new_covariance
        weights = new_weights
        if movement <= convergence_tolerance**2:
            break

    components = tuple(
        GmmComponent(
            cluster_id=index + 1,
            weight=weights[index],
            mean=means[index],
            covariance=covariance[index],
            color=CLUSTER_COLORS[index % len(CLUSTER_COLORS)],
        )
        for index in range(component_count)
    )
    return GmmModel(components)


def _component_boundary_coefficients(component: GmmComponent) -> dict[str, float]:
    variance_x, covariance_xy, variance_y = component.covariance
    determinant = variance_x * variance_y - covariance_xy * covariance_xy
    inverse_xx = variance_y / determinant
    inverse_xy = -covariance_xy / determinant
    inverse_yy = variance_x / determinant
    mean_x, mean_y = component.mean
    return {
        "x2": -0.5 * inverse_xx,
        "xy": -inverse_xy,
        "y2": -0.5 * inverse_yy,
        "x": inverse_xx * mean_x + inverse_xy * mean_y,
        "y": inverse_xy * mean_x + inverse_yy * mean_y,
        "constant": (
            -0.5
            * (
                inverse_xx * mean_x * mean_x
                + 2.0 * inverse_xy * mean_x * mean_y
                + inverse_yy * mean_y * mean_y
            )
            + math.log(max(component.weight, 1e-12))
            - 0.5 * math.log(determinant)
            - math.log(2.0 * math.pi)
        ),
    }


def _pairwise_boundary(first: GmmComponent, second: GmmComponent) -> dict[str, object]:
    first_coefficients = _component_boundary_coefficients(first)
    second_coefficients = _component_boundary_coefficients(second)
    return {
        "between": [first.cluster_id, second.cluster_id],
        "equation": f"score_{first.cluster_id} - score_{second.cluster_id} = 0",
        "coefficients": {
            key: first_coefficients[key] - second_coefficients[key]
            for key in first_coefficients
        },
    }


def _model_payload(model: GmmModel) -> dict[str, object]:
    components = []
    for component in model.components:
        mean_x, mean_y = component.mean
        variance_x, covariance_xy, variance_y = component.covariance
        components.append(
            {
                "cluster_id": component.cluster_id,
                "color": component.color,
                "weight": component.weight,
                "mean": {
                    "observations": mean_x * OBSERVATION_SCALE,
                    "log10_error_km": mean_y,
                    "error_km": 10**mean_y,
                },
                "covariance": {
                    "observations_scaled_xx": variance_x,
                    "observations_scaled_log_error_xy": covariance_xy,
                    "log_error_yy": variance_y,
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "model": "gaussian_mixture",
        "decision_rule": "assign the maximum weighted Gaussian log-likelihood",
        "feature_space": {
            "x": "identified_objects_used / 10",
            "y": "log10(error_distance_meters / 1000)",
            "observation_scale": OBSERVATION_SCALE,
        },
        "seed_locations": [
            {
                "name": name,
                "observations": observations,
                "error_km": error_km,
            }
            for observations, error_km, name in SEEDS
        ],
        "components": components,
        "pairwise_boundaries": [
            _pairwise_boundary(first, second)
            for first_index, first in enumerate(model.components)
            for second in model.components[first_index + 1 :]
        ],
    }


def save_gmm_boundaries(model: GmmModel, output_path: Path) -> None:
    """Persist the model and explicit pairwise decision-boundary equations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as stream:
            json.dump(_model_payload(model), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_gmm_boundaries(path: Path) -> GmmModel:
    """Load a persisted GMM boundary model and reject incompatible schemas."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported GMM boundary schema: {payload.get('schema_version')}")
    if payload.get("model") != "gaussian_mixture":
        raise ValueError("boundary file is not a Gaussian-mixture model")

    components = []
    for item in payload.get("components", []):
        mean = item["mean"]
        covariance = item["covariance"]
        components.append(
            GmmComponent(
                cluster_id=int(item["cluster_id"]),
                weight=float(item["weight"]),
                mean=(
                    float(mean["observations"]) / OBSERVATION_SCALE,
                    float(mean["log10_error_km"]),
                ),
                covariance=(
                    float(covariance["observations_scaled_xx"]),
                    float(covariance["observations_scaled_log_error_xy"]),
                    float(covariance["log_error_yy"]),
                ),
                color=str(item["color"]),
            )
        )
    if not components:
        raise ValueError("boundary file contains no GMM components")
    return GmmModel(tuple(sorted(components, key=lambda component: component.cluster_id)))


def classify_sample(model: GmmModel, observations: float, error_km: float) -> int:
    return model.classify(observations, error_km)


def classify_live_observations(model: GmmModel, observations: float) -> int:
    return model.classify_observations(observations)


def cluster_filename(cluster_id: int) -> str:
    name = (
        CLUSTER_NAMES[cluster_id - 1]
        if 1 <= cluster_id <= len(CLUSTER_NAMES)
        else f"cluster{cluster_id}"
    )
    return f"cluster{cluster_id}_{name}.png"