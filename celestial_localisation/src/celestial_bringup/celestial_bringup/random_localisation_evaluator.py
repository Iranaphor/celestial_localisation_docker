#!/usr/bin/env python3
import csv
import importlib.metadata
import json
import math
import os
import random
from statistics import median
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
from builtin_interfaces.msg import Time as RosTime
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from celestial_interfaces.srv import LoadGps, RunRandomEvaluation
from celestial_detector.gmm_classifier import (
    classify_sample,
    cluster_filename,
    load_gmm_boundaries,
)


EARTH_RADIUS_METERS = 6_371_000.0
NANOSECONDS_PER_SECOND = 1_000_000_000
CSV_FIELDS = (
    'run_index',
    'record_type',
    'sample_id',
    'sample_index',
    'repetition_index',
    'is_outlier',
    'timestamp_utc',
    'sample_timestamp_utc',
    'request_timestamp_utc',
    'request_yaw_degrees',
    'random_seed',
    'evaluation_config_json',
    'code_revision',
    'dependency_versions_json',
    'sample_ground_truth_latitude',
    'sample_ground_truth_longitude',
    'ground_truth_latitude',
    'ground_truth_longitude',
    'estimated_latitude',
    'estimated_longitude',
    'estimated_heading_degrees',
    'identified_objects_used',
    'used_object_types_json',
    'identified_star_ids',
    'cluster_id',
    'inlier_count',
    'outlier_count',
    'attempted_count',
    'valid_count',
    'failure_count',
    'valid_estimate',
    'failure_reason',
    'solver_status',
    'solver_success',
    'solver_message',
    'solver_cost',
    'solver_optimality',
    'solver_nfev',
    'solver_total_nfev',
    'solver_rms_residual',
    'solver_max_residual',
    'solver_boundary_solution',
    'solver_initial_latitude',
    'solver_initial_longitude',
    'solver_initial_heading_degrees',
    'solver_selected_start_latitude',
    'solver_selected_start_longitude',
    'solver_selected_start_heading_degrees',
    'solver_starts_tried',
    'latitude_error_degrees',
    'longitude_error_degrees',
    'latitude_error_meters',
    'longitude_error_meters',
    'error_distance_meters',
    'cumulative_mean_error_meters',
)


def random_surface_location(random_source):
    """Return a position uniformly distributed over the surface of the Earth."""
    longitude = random_source.uniform(-180.0, 180.0)
    latitude = math.degrees(math.asin(random_source.uniform(-1.0, 1.0)))
    return latitude, longitude


def perturb_surface_location(latitude, longitude, maximum_distance_meters, random_source):
    if maximum_distance_meters == 0.0:
        return latitude, longitude

    distance = math.sqrt(random_source.uniform(0.0, 1.0)) * maximum_distance_meters
    bearing = random_source.uniform(0.0, 2.0 * math.pi)
    angular_distance = distance / EARTH_RADIUS_METERS
    latitude_radians = math.radians(latitude)
    longitude_radians = math.radians(longitude)
    destination_latitude = math.asin(
        math.sin(latitude_radians) * math.cos(angular_distance)
        + math.cos(latitude_radians) * math.sin(angular_distance) * math.cos(bearing)
    )
    destination_longitude = longitude_radians + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(latitude_radians),
        math.cos(angular_distance) - math.sin(latitude_radians) * math.sin(destination_latitude),
    )
    return (
        math.degrees(destination_latitude),
        (math.degrees(destination_longitude) + 180.0) % 360.0 - 180.0,
    )


def _wrapped_longitude_delta(estimated_longitude, ground_truth_longitude):
    return (estimated_longitude - ground_truth_longitude + 180.0) % 360.0 - 180.0


def calculate_error_metrics(ground_truth_latitude, ground_truth_longitude, estimated_latitude, estimated_longitude):
    latitude_error_degrees = estimated_latitude - ground_truth_latitude
    longitude_error_degrees = _wrapped_longitude_delta(estimated_longitude, ground_truth_longitude)
    latitude_error_meters = math.radians(latitude_error_degrees) * EARTH_RADIUS_METERS
    mean_latitude = math.radians((ground_truth_latitude + estimated_latitude) / 2.0)
    longitude_error_meters = (
        math.radians(longitude_error_degrees) * EARTH_RADIUS_METERS * math.cos(mean_latitude)
    )

    ground_truth_latitude_radians = math.radians(ground_truth_latitude)
    estimated_latitude_radians = math.radians(estimated_latitude)
    latitude_delta_radians = estimated_latitude_radians - ground_truth_latitude_radians
    longitude_delta_radians = math.radians(longitude_error_degrees)
    haversine_a = (
        math.sin(latitude_delta_radians / 2.0) ** 2
        + math.cos(ground_truth_latitude_radians)
        * math.cos(estimated_latitude_radians)
        * math.sin(longitude_delta_radians / 2.0) ** 2
    )
    error_distance_meters = 2.0 * EARTH_RADIUS_METERS * math.atan2(
        math.sqrt(haversine_a), math.sqrt(max(0.0, 1.0 - haversine_a))
    )

    return {
        'latitude_error_degrees': latitude_error_degrees,
        'longitude_error_degrees': longitude_error_degrees,
        'latitude_error_meters': latitude_error_meters,
        'longitude_error_meters': longitude_error_meters,
        'error_distance_meters': error_distance_meters,
    }


def _circular_median_longitude(longitudes):
    candidates = [((longitude + 180.0) % 360.0) - 180.0 for longitude in longitudes]
    return min(
        candidates,
        key=lambda candidate: sum(
            abs(_wrapped_longitude_delta(longitude, candidate))
            for longitude in candidates
        ),
    )


def average_pose_estimates(estimates, ground_truth_latitude=None, ground_truth_longitude=None):
    """Aggregate estimates in estimate space; ground truth is scoring-only."""
    if not estimates:
        raise ValueError('at least one pose estimate is required')

    reference_latitude = median(estimate['latitude'] for estimate in estimates)
    reference_longitude = _circular_median_longitude(
        estimate['longitude'] for estimate in estimates
    )
    reference_latitude_radians = math.radians(reference_latitude)
    reference_cosine = math.cos(reference_latitude_radians)
    offsets = []
    for estimate in estimates:
        east_offset = (
            math.radians(_wrapped_longitude_delta(estimate['longitude'], reference_longitude))
            * EARTH_RADIUS_METERS
            * reference_cosine
        )
        north_offset = math.radians(estimate['latitude'] - reference_latitude) * EARTH_RADIUS_METERS
        offsets.append((east_offset, north_offset))

    center_east = median(offset[0] for offset in offsets)
    center_north = median(offset[1] for offset in offsets)
    distances = [
        math.hypot(east - center_east, north - center_north)
        for east, north in offsets
    ]
    median_distance = sorted(distances)[len(distances) // 2]
    deviations = [abs(distance - median_distance) for distance in distances]
    median_absolute_deviation = sorted(deviations)[len(deviations) // 2]
    outlier_threshold = max(
        1e-3,
        median_distance + 3.0 * 1.4826 * median_absolute_deviation,
    )
    inlier_indices = [
        index for index, distance in enumerate(distances) if distance <= outlier_threshold
    ]
    if not inlier_indices:
        inlier_indices = [min(range(len(distances)), key=distances.__getitem__)]

    sample_east = median(offsets[index][0] for index in inlier_indices)
    sample_north = median(offsets[index][1] for index in inlier_indices)
    sample_latitude = max(
        -90.0,
        min(90.0, reference_latitude + math.degrees(sample_north / EARTH_RADIUS_METERS)),
    )
    if abs(reference_cosine) < 1e-12:
        sample_longitude = reference_longitude
    else:
        sample_longitude = (
            reference_longitude
            + math.degrees(sample_east / (EARTH_RADIUS_METERS * reference_cosine))
        )
        sample_longitude = (sample_longitude + 180.0) % 360.0 - 180.0

    heading_sine = sum(
        math.sin(math.radians(estimates[index]['heading'])) for index in inlier_indices
    )
    heading_cosine = sum(
        math.cos(math.radians(estimates[index]['heading'])) for index in inlier_indices
    )
    mean_heading = math.degrees(math.atan2(heading_sine, heading_cosine))
    mean_identified_objects = int(round(
        sum(estimates[index]['identified_objects_used'] for index in inlier_indices)
        / len(inlier_indices)
    ))
    identified_star_ids = sorted({
        star_id
        for index in inlier_indices
        for star_id in estimates[index].get('identified_star_ids', [])
        if star_id
    })
    used_object_types = {}
    for index in inlier_indices:
        for object_type, count in estimates[index].get('used_object_types', {}).items():
            used_object_types[object_type] = used_object_types.get(object_type, 0) + int(count)
    representative_index = min(
        inlier_indices,
        key=lambda index: math.hypot(
            offsets[index][0] - sample_east,
            offsets[index][1] - sample_north,
        ),
    )
    return {
        'latitude': sample_latitude,
        'longitude': sample_longitude,
        'heading': mean_heading,
        'identified_objects_used': mean_identified_objects,
        'identified_star_ids': identified_star_ids,
        'used_object_types': used_object_types,
        'inlier_count': len(inlier_indices),
        'outlier_count': len(estimates) - len(inlier_indices),
        'inlier_indices': inlier_indices,
        'sky_map_ready': estimates[representative_index]['sky_map_ready'],
    }


def _timestamp_from_nanoseconds(timestamp_ns):
    timestamp = RosTime()
    timestamp.sec = timestamp_ns // NANOSECONDS_PER_SECOND
    timestamp.nanosec = timestamp_ns % NANOSECONDS_PER_SECOND
    return timestamp


def _extract_identified_star_ids(used_observations):
    excluded_ids = {'', 'sun', 'moon', 'unknown'}
    star_ids = set()
    for observation in used_observations:
        if not isinstance(observation, dict):
            continue
        object_id = observation.get('object_id')
        if not isinstance(object_id, str):
            continue
        object_id = object_id.strip()
        if object_id and object_id.lower() not in excluded_ids:
            star_ids.add(object_id)
    return sorted(star_ids)


class RandomLocalisationEvaluator(Node):
    def __init__(self):
        super().__init__('random_localisation_evaluator')

        self.declare_parameter('load_gps_service', '/test/load_gps')
        self.declare_parameter('evaluation_service', '/test/run_random_evaluation')
        self.declare_parameter(
            'debug_output_dir',
            os.environ.get('CELESTIAL_DEBUG_OUTPUT_DIR', ''),
        )
        self.declare_parameter('pose_filename', 'pose.json')
        self.declare_parameter('metrics_filename', 'random_localisation_metrics.csv')
        self.declare_parameter('gmm_boundaries_filename', 'gmm_boundaries.json')
        self.declare_parameter('fixed_altitude', 0.0)
        self.declare_parameter('result_timeout_seconds', 180.0)
        self.declare_parameter('poll_interval_seconds', 0.25)
        self.declare_parameter('max_samples', 20)
        self.declare_parameter('max_reps', 10)

        self.debug_dir = self._resolve_debug_dir()
        self.pose_path = self.debug_dir / self.get_parameter('pose_filename').value if self.debug_dir else None
        self.metrics_path = self.debug_dir / self.get_parameter('metrics_filename').value if self.debug_dir else None
        self.sky_map_path = self.debug_dir / 'sky_map.png' if self.debug_dir else None
        self.sky_map_classification_path = (
            self.debug_dir / 'sky_map_classification.json' if self.debug_dir else None
        )
        self.gmm_boundaries_path = (
            self.debug_dir / self.get_parameter('gmm_boundaries_filename').value
            if self.debug_dir
            else None
        )
        self.gmm_model = None
        if self.gmm_boundaries_path and self.gmm_boundaries_path.exists():
            try:
                self.gmm_model = load_gmm_boundaries(self.gmm_boundaries_path)
                self.get_logger().info(
                    f'loaded GMM boundaries from {self.gmm_boundaries_path}'
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.get_logger().warning(
                    f'could not load GMM boundaries from {self.gmm_boundaries_path}: {error}'
                )
        elif self.gmm_boundaries_path:
            self.get_logger().warning(
                f'GMM boundaries not found at {self.gmm_boundaries_path}; '
                'CSV rows will remain unclassified'
            )
        self.result_timeout_seconds = float(self.get_parameter('result_timeout_seconds').value)
        self.poll_interval_seconds = float(self.get_parameter('poll_interval_seconds').value)
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.max_reps = int(self.get_parameter('max_reps').value)
        if self.max_samples <= 0 or self.max_reps <= 0:
            raise ValueError('random evaluation limits must be greater than zero')
        self.fixed_altitude = float(self.get_parameter('fixed_altitude').value)
        self.code_revision = os.environ.get('CELESTIAL_CODE_REVISION', '')
        self.dependency_versions = self._load_dependency_versions()
        (
            self._metrics_run_count,
            self._total_error_meters,
            self._summary_count,
        ) = self._load_metrics_state()

        self._random_source = random.SystemRandom()
        self._evaluation_lock = threading.Lock()
        self._last_timestamp_ns = 0
        self._used_timestamp_ns = set()
        self._callback_group = ReentrantCallbackGroup()
        load_gps_service = self.get_parameter('load_gps_service').value
        evaluation_service = self.get_parameter('evaluation_service').value

        self.load_gps_client = self.create_client(
            LoadGps,
            load_gps_service,
            callback_group=self._callback_group,
        )
        self.evaluation_service = self.create_service(
            RunRandomEvaluation,
            evaluation_service,
            self._run_evaluation,
            callback_group=self._callback_group,
        )

        self.get_logger().info(
            f'random localisation evaluator ready on {evaluation_service}; '
            f'LoadGps service is {load_gps_service}'
            + (f', writing metrics to {self.metrics_path}' if self.metrics_path else '')
        )

    def _resolve_debug_dir(self):
        configured = self.get_parameter('debug_output_dir').value
        if not configured:
            return None

        debug_dir = Path(configured)
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self.get_logger().error(f'cannot create debug output directory {debug_dir}: {error}')
            return None
        return debug_dir

    @staticmethod
    def _load_dependency_versions():
        package_names = (
            'numpy',
            'scipy',
            'astropy',
            'tetra3',
            'photutils',
            'opencv-python-headless',
            'playwright',
        )
        versions = {}
        for package_name in package_names:
            try:
                versions[package_name] = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return versions

    def _load_metrics_state(self):
        if self.metrics_path is None or not self.metrics_path.exists():
            return 0, 0.0, 0

        highest_run_index = 0
        total_error_meters = 0.0
        summary_count = 0
        try:
            with self.metrics_path.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    record_type = (row.get('record_type') or '').strip().lower()
                    try:
                        highest_run_index = max(highest_run_index, int(row['run_index']))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if record_type and record_type != 'summary':
                        continue
                    total_error_meters += float(row['error_distance_meters'])
                    summary_count += 1
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.get_logger().warning(
                f'could not restore cumulative metrics from {self.metrics_path}: {error}'
            )
            return 0, 0.0, 0
        return highest_run_index, total_error_meters, summary_count

    def _run_evaluation(self, request, response):
        response.completed_runs = 0
        samples = int(request.samples)
        reps = int(request.reps)
        variance_values = {
            'var_time': float(request.var_time),
            'var_xy': float(request.var_xy),
            'var_yaw': float(request.var_yaw),
        }
        if samples <= 0:
            response.success = False
            response.message = 'samples must be greater than zero'
            return response
        if reps <= 0:
            response.success = False
            response.message = 'reps must be greater than zero'
            return response
        if samples > self.max_samples:
            response.success = False
            response.message = f'samples must not exceed {self.max_samples}'
            return response
        if reps > self.max_reps:
            response.success = False
            response.message = f'reps must not exceed {self.max_reps}'
            return response
        for name, value in variance_values.items():
            if not math.isfinite(value) or value < 0.0:
                response.success = False
                response.message = f'{name} must be finite and non-negative'
                return response
        if variance_values['var_yaw'] > 360.0:
            response.success = False
            response.message = 'var_yaw must not be greater than 360 degrees'
            return response
        if self.debug_dir is None or self.pose_path is None or self.metrics_path is None:
            response.success = False
            response.message = 'debug_output_dir must be configured for random evaluation'
            return response
        if not self._evaluation_lock.acquire(blocking=False):
            response.success = False
            response.message = 'a random evaluation is already running'
            return response

        try:
            if not self.load_gps_client.wait_for_service(timeout_sec=self.result_timeout_seconds):
                response.success = False
                response.message = 'LoadGps service did not become available before the timeout'
                return response

            requested_seed = int(getattr(request, 'random_seed', 0))
            random_seed = requested_seed or random.SystemRandom().randrange(1, 2**63)
            random_source = random.Random(random_seed)
            evaluation_config = {
                'samples': samples,
                'reps': reps,
                'var_time': variance_values['var_time'],
                'var_xy': variance_values['var_xy'],
                'var_yaw': variance_values['var_yaw'],
                'fixed_altitude': self.fixed_altitude,
                'random_seed': random_seed,
            }
            evaluation_config_json = json.dumps(evaluation_config, sort_keys=True, separators=(',', ':'))
            dependency_versions_json = json.dumps(
                self.dependency_versions,
                sort_keys=True,
                separators=(',', ':'),
            )
            batch_total_error_meters = 0.0
            completed_summaries = 0
            total_failures = 0
            for sample_index in range(1, samples + 1):
                ground_truth_latitude, ground_truth_longitude = random_surface_location(random_source)
                run_index = self._metrics_run_count + 1
                sample_id = f'sample-{run_index:06d}'
                sample_timestamp = self._next_timestamp()
                estimates = []
                for repetition_index in range(1, reps + 1):
                    varied_latitude, varied_longitude = perturb_surface_location(
                        ground_truth_latitude,
                        ground_truth_longitude,
                        variance_values['var_xy'],
                        random_source,
                    )
                    timestamp = self._timestamp_with_variation(
                        sample_timestamp,
                        variance_values['var_time'],
                        random_source,
                    )
                    yaw = random_source.uniform(
                        -variance_values['var_yaw'],
                        variance_values['var_yaw'],
                    )
                    load_request = LoadGps.Request()
                    load_request.latitude = varied_latitude
                    load_request.longitude = varied_longitude
                    load_request.altitude = self.fixed_altitude
                    load_request.yaw = yaw
                    load_request.timestamp = timestamp

                    try:
                        load_response = self._wait_for_future(
                            self.load_gps_client.call_async(load_request),
                            self.result_timeout_seconds,
                        )
                    except Exception as error:
                        total_failures += 1
                        self._append_failure(
                            run_index,
                            ground_truth_latitude,
                            ground_truth_longitude,
                            varied_latitude,
                            varied_longitude,
                            sample_timestamp,
                            timestamp,
                            yaw,
                            f'load_gps_exception: {error}',
                            sample_id=sample_id,
                            sample_index=sample_index,
                            repetition_index=repetition_index,
                            attempted_count=reps,
                            valid_count=len(estimates),
                            random_seed=random_seed,
                            evaluation_config_json=evaluation_config_json,
                            code_revision=self.code_revision,
                            dependency_versions_json=dependency_versions_json,
                        )
                        continue

                    if load_response is None:
                        total_failures += 1
                        self._append_failure(
                            run_index,
                            ground_truth_latitude,
                            ground_truth_longitude,
                            varied_latitude,
                            varied_longitude,
                            sample_timestamp,
                            timestamp,
                            yaw,
                            'load_gps_timeout',
                            sample_id=sample_id,
                            sample_index=sample_index,
                            repetition_index=repetition_index,
                            attempted_count=reps,
                            valid_count=len(estimates),
                            identified_objects_used=estimated.get('identified_objects_used', ''),
                            used_object_types=estimated.get('used_object_types', {}),
                            random_seed=random_seed,
                            evaluation_config_json=evaluation_config_json,
                            code_revision=self.code_revision,
                            dependency_versions_json=dependency_versions_json,
                        )
                        continue
                    if not load_response.success:
                        total_failures += 1
                        self._append_failure(
                            run_index,
                            ground_truth_latitude,
                            ground_truth_longitude,
                            varied_latitude,
                            varied_longitude,
                            sample_timestamp,
                            timestamp,
                            yaw,
                            f'load_gps_rejected: {load_response.message}',
                            sample_id=sample_id,
                            sample_index=sample_index,
                            repetition_index=repetition_index,
                            attempted_count=reps,
                            valid_count=len(estimates),
                            random_seed=random_seed,
                            evaluation_config_json=evaluation_config_json,
                            code_revision=self.code_revision,
                            dependency_versions_json=dependency_versions_json,
                        )
                        continue

                    estimated = self._wait_for_pose(timestamp, self.result_timeout_seconds)
                    if estimated is None:
                        total_failures += 1
                        self._append_failure(
                            run_index,
                            ground_truth_latitude,
                            ground_truth_longitude,
                            varied_latitude,
                            varied_longitude,
                            sample_timestamp,
                            timestamp,
                            yaw,
                            'pose_timeout',
                            sample_id=sample_id,
                            sample_index=sample_index,
                            repetition_index=repetition_index,
                            attempted_count=reps,
                            valid_count=len(estimates),
                            random_seed=random_seed,
                            evaluation_config_json=evaluation_config_json,
                            code_revision=self.code_revision,
                            dependency_versions_json=dependency_versions_json,
                        )
                        continue

                    if not estimated['valid']:
                        total_failures += 1
                        self._append_failure(
                            run_index,
                            ground_truth_latitude,
                            ground_truth_longitude,
                            varied_latitude,
                            varied_longitude,
                            sample_timestamp,
                            timestamp,
                            yaw,
                            estimated['failure_reason'],
                            diagnostics=estimated['diagnostics'],
                            sample_id=sample_id,
                            sample_index=sample_index,
                            repetition_index=repetition_index,
                            attempted_count=reps,
                            valid_count=len(estimates),
                            random_seed=random_seed,
                            evaluation_config_json=evaluation_config_json,
                            code_revision=self.code_revision,
                            dependency_versions_json=dependency_versions_json,
                        )
                        continue

                    sky_map_ready = self._wait_for_sky_map(
                        timestamp,
                        min(self.result_timeout_seconds, max(5.0, self.poll_interval_seconds * 20.0)),
                    )

                    estimates.append({
                        'latitude': estimated['latitude'],
                        'longitude': estimated['longitude'],
                        'heading': estimated['heading'],
                        'identified_objects_used': estimated['identified_objects_used'],
                        'identified_star_ids': estimated['identified_star_ids'],
                        'used_object_types': estimated['used_object_types'],
                        'diagnostics': estimated['diagnostics'],
                        'sky_map_ready': sky_map_ready,
                        'ground_truth_latitude': varied_latitude,
                        'ground_truth_longitude': varied_longitude,
                        'timestamp': timestamp,
                        'yaw': yaw,
                        'repetition_index': repetition_index,
                    })

                self._metrics_run_count = run_index
                response.completed_runs = sample_index
                if not estimates:
                    self.get_logger().warning(
                        f'random evaluation sample {sample_index}/{samples} had no valid estimates'
                    )
                    continue

                averaged_estimate = average_pose_estimates(
                    estimates,
                    ground_truth_latitude,
                    ground_truth_longitude,
                )
                inlier_indices = set(averaged_estimate['inlier_indices'])
                failure_count = reps - len(estimates)
                for estimate_index, estimate in enumerate(estimates):
                    repetition_metrics = calculate_error_metrics(
                        estimate['ground_truth_latitude'],
                        estimate['ground_truth_longitude'],
                        estimate['latitude'],
                        estimate['longitude'],
                    )
                    self._append_metrics(
                        run_index,
                        estimate['ground_truth_latitude'],
                        estimate['ground_truth_longitude'],
                        estimate['latitude'],
                        estimate['longitude'],
                        estimate['identified_objects_used'],
                        estimate['identified_star_ids'],
                        repetition_metrics,
                        '',
                        record_type='step',
                        sample_id=sample_id,
                        sample_index=sample_index,
                        repetition_index=estimate['repetition_index'],
                        is_outlier=estimate_index not in inlier_indices,
                        sample_timestamp_utc=self._timestamp_to_utc(sample_timestamp),
                        sample_ground_truth_latitude=ground_truth_latitude,
                        sample_ground_truth_longitude=ground_truth_longitude,
                        request_timestamp_utc=self._timestamp_to_utc(estimate['timestamp']),
                        request_yaw_degrees=estimate['yaw'],
                        estimated_heading_degrees=estimate['heading'],
                        used_object_types=estimate['used_object_types'],
                        diagnostics=estimate['diagnostics'],
                        inlier_count=averaged_estimate['inlier_count'],
                        outlier_count=averaged_estimate['outlier_count'],
                        attempted_count=reps,
                        valid_count=len(estimates),
                        failure_count=failure_count,
                        random_seed=random_seed,
                        evaluation_config_json=evaluation_config_json,
                        code_revision=self.code_revision,
                        dependency_versions_json=dependency_versions_json,
                    )
                metrics = calculate_error_metrics(
                    ground_truth_latitude,
                    ground_truth_longitude,
                    averaged_estimate['latitude'],
                    averaged_estimate['longitude'],
                )
                batch_total_error_meters += metrics['error_distance_meters']
                self._total_error_meters += metrics['error_distance_meters']
                self._summary_count += 1
                completed_summaries += 1
                cluster_id = self._append_metrics(
                    run_index,
                    ground_truth_latitude,
                    ground_truth_longitude,
                    averaged_estimate['latitude'],
                    averaged_estimate['longitude'],
                    averaged_estimate['identified_objects_used'],
                    averaged_estimate['identified_star_ids'],
                    metrics,
                    self._total_error_meters / self._summary_count,
                    record_type='summary',
                    sample_id=sample_id,
                    sample_index=sample_index,
                    is_outlier='',
                    sample_timestamp_utc=self._timestamp_to_utc(sample_timestamp),
                    sample_ground_truth_latitude=ground_truth_latitude,
                    sample_ground_truth_longitude=ground_truth_longitude,
                    estimated_heading_degrees=averaged_estimate['heading'],
                    used_object_types=averaged_estimate['used_object_types'],
                    inlier_count=averaged_estimate['inlier_count'],
                    outlier_count=averaged_estimate['outlier_count'],
                    attempted_count=reps,
                    valid_count=len(estimates),
                    failure_count=failure_count,
                    valid_estimate='aggregated',
                    random_seed=random_seed,
                    evaluation_config_json=evaluation_config_json,
                    code_revision=self.code_revision,
                    dependency_versions_json=dependency_versions_json,
                )
                self._save_cluster_image(
                    cluster_id,
                    averaged_estimate['sky_map_ready'],
                    ground_truth_latitude,
                    ground_truth_longitude,
                    averaged_estimate['latitude'],
                    averaged_estimate['longitude'],
                    metrics['error_distance_meters'],
                )
                self.get_logger().info(
                    f'random evaluation sample {sample_index}/{samples} '
                    f'({averaged_estimate["inlier_count"]}/{reps} repetitions kept): '
                    f'ground_truth=({ground_truth_latitude:.6f}, {ground_truth_longitude:.6f}), '
                    f'estimated=({averaged_estimate["latitude"]:.6f}, '
                    f'{averaged_estimate["longitude"]:.6f}), '
                    f'identified_objects={averaged_estimate["identified_objects_used"]}, '
                    f'cluster={cluster_id if cluster_id is not None else "unclassified"}, '
                    f'error={metrics["error_distance_meters"]:.2f} m, '
                    f'batch_mean={batch_total_error_meters / completed_summaries:.2f} m, '
                    f'cumulative_mean={self._total_error_meters / self._summary_count:.2f} m'
                )

            response.success = True
            batch_mean = (
                batch_total_error_meters / completed_summaries
                if completed_summaries
                else float('nan')
            )
            cumulative_mean = (
                self._total_error_meters / self._summary_count
                if self._summary_count
                else float('nan')
            )
            response.message = (
                f'completed {samples} random localisation samples with {reps} repetitions each; '
                f'valid summaries={completed_summaries}; failures={total_failures}; '
                f'batch mean error={batch_mean:.2f} m; '
                f'cumulative mean error={cumulative_mean:.2f} m; '
                f'random_seed={random_seed}; '
                f'metrics saved to {self.metrics_path}'
            )
            return response
        except Exception as error:
            self.get_logger().error(f'random localisation evaluation failed: {error}')
            response.success = False
            response.message = f'random evaluation failed: {error}'
            return response
        finally:
            self._evaluation_lock.release()

    def _next_timestamp(self):
        timestamp = self.get_clock().now().to_msg()
        timestamp_ns = timestamp.sec * NANOSECONDS_PER_SECOND + timestamp.nanosec
        if timestamp_ns <= self._last_timestamp_ns:
            timestamp_ns = self._last_timestamp_ns + 1
        while timestamp_ns in self._used_timestamp_ns:
            timestamp_ns += 1
        self._last_timestamp_ns = timestamp_ns
        self._used_timestamp_ns.add(timestamp_ns)
        return _timestamp_from_nanoseconds(timestamp_ns)

    def _timestamp_with_variation(self, base_timestamp, maximum_seconds, random_source):
        base_timestamp_ns = (
            base_timestamp.sec * NANOSECONDS_PER_SECOND + base_timestamp.nanosec
        )
        variation_ns = round(
            random_source.uniform(-maximum_seconds, maximum_seconds)
            * NANOSECONDS_PER_SECOND
        )
        timestamp_ns = max(1, base_timestamp_ns + variation_ns)
        while timestamp_ns in self._used_timestamp_ns:
            timestamp_ns += 1
        self._used_timestamp_ns.add(timestamp_ns)
        self._last_timestamp_ns = max(self._last_timestamp_ns, timestamp_ns)
        return _timestamp_from_nanoseconds(timestamp_ns)

    @staticmethod
    def _timestamp_to_utc(timestamp):
        timestamp_seconds = timestamp.sec + timestamp.nanosec / NANOSECONDS_PER_SECOND
        return datetime.fromtimestamp(timestamp_seconds, timezone.utc).isoformat().replace('+00:00', 'Z')

    def _wait_for_pose(self, expected_timestamp, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            pose = self._read_pose(expected_timestamp)
            if pose is not None:
                return pose
            time.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))
        return None

    def _read_pose(self, expected_timestamp):
        try:
            with self.pose_path.open(encoding='utf-8') as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

        observed_timestamp = payload.get('observation_timestamp')
        if not isinstance(observed_timestamp, dict):
            return None
        try:
            observed_sec = int(observed_timestamp.get('sec', -1))
            observed_nanosec = int(observed_timestamp.get('nanosec', -1))
        except (TypeError, ValueError, OverflowError):
            return None
        if observed_sec != int(expected_timestamp.sec) or observed_nanosec != int(expected_timestamp.nanosec):
            return None

        solver_payload = payload.get('solver')
        if not isinstance(solver_payload, dict):
            solver_payload = {}

        def numeric(name, default=''):
            value = solver_payload.get(name, default)
            if value == '':
                return ''
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        def integer(name, default=''):
            value = solver_payload.get(name, default)
            if value == '':
                return ''
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        initial_guess = payload.get('initial_guess')
        if not isinstance(initial_guess, dict):
            initial_guess = {}
        selected_start = solver_payload.get('selected_start')
        if not isinstance(selected_start, dict):
            selected_start = {}
        diagnostics = {
            'status': integer('status'),
            'success': solver_payload.get('success', ''),
            'message': str(solver_payload.get('message', '')),
            'cost': numeric('cost'),
            'optimality': numeric('optimality'),
            'nfev': integer('nfev'),
            'total_nfev': integer('total_nfev'),
            'rms_residual': numeric('rms_residual'),
            'max_residual': numeric('max_residual'),
            'boundary_solution': solver_payload.get('boundary_solution', ''),
            'initial_latitude': initial_guess.get('latitude', ''),
            'initial_longitude': initial_guess.get('longitude', ''),
            'initial_heading': initial_guess.get('heading', ''),
            'selected_start_latitude': selected_start.get('latitude', ''),
            'selected_start_longitude': selected_start.get('longitude', ''),
            'selected_start_heading': selected_start.get('heading', ''),
            'starts_tried': integer('starts_tried'),
        }
        valid = payload.get('valid', True) is not False
        if not valid:
            used_observations = payload.get('used_observations')
            if not isinstance(used_observations, list):
                used_observations = []
            used_object_types = payload.get('used_object_types')
            if not isinstance(used_object_types, dict):
                used_object_types = {}
            return {
                'valid': False,
                'failure_reason': str(
                    solver_payload.get('failure_reason') or 'invalid_pose'
                ),
                'identified_objects_used': len(used_observations),
                'identified_star_ids': _extract_identified_star_ids(used_observations),
                'used_object_types': used_object_types,
                'diagnostics': diagnostics,
            }

        refined_estimate = payload.get('refined_estimate')
        if not isinstance(refined_estimate, dict):
            return None
        try:
            latitude = float(refined_estimate['latitude'])
            longitude = float(refined_estimate['longitude'])
            heading = float(refined_estimate.get('heading', 0.0))
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not math.isfinite(heading)
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
        ):
            return None
        used_observations = payload.get('used_observations')
        if not isinstance(used_observations, list):
            return None
        identified_star_ids = _extract_identified_star_ids(used_observations)
        used_object_types = payload.get('used_object_types')
        if not isinstance(used_object_types, dict):
            used_object_types = {}
            for observation in used_observations:
                if not isinstance(observation, dict):
                    continue
                object_type = observation.get('object_type')
                if not isinstance(object_type, str):
                    object_id = str(observation.get('object_id', '')).lower()
                    object_type = 'sun' if object_id == 'sun' else 'moon' if object_id == 'moon' else 'star'
                used_object_types[object_type] = used_object_types.get(object_type, 0) + 1
        return {
            'valid': True,
            'failure_reason': '',
            'latitude': latitude,
            'longitude': longitude,
            'heading': heading,
            'identified_objects_used': len(used_observations),
            'identified_star_ids': identified_star_ids,
            'used_object_types': used_object_types,
            'diagnostics': diagnostics,
        }

    def _wait_for_sky_map(self, expected_timestamp, timeout_seconds):
        if self.sky_map_classification_path is None:
            return False

        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                with self.sky_map_classification_path.open(encoding='utf-8') as stream:
                    payload = json.load(stream)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                payload = None

            observed_timestamp = payload.get('observation_timestamp') if isinstance(payload, dict) else None
            if isinstance(observed_timestamp, dict):
                try:
                    observed_sec = int(observed_timestamp.get('sec', -1))
                    observed_nanosec = int(observed_timestamp.get('nanosec', -1))
                except (TypeError, ValueError, OverflowError):
                    observed_sec = -1
                    observed_nanosec = -1
                if (
                    observed_sec == int(expected_timestamp.sec)
                    and observed_nanosec == int(expected_timestamp.nanosec)
                    and self.sky_map_path is not None
                    and self.sky_map_path.exists()
                ):
                    return True
            time.sleep(min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic())))

        self.get_logger().warning(
            f'sky_map.png was not confirmed for timestamp '
            f'{expected_timestamp.sec}.{expected_timestamp.nanosec:09d}'
        )
        return False

    def _load_gmm_model_if_available(self):
        if self.gmm_model is not None or self.gmm_boundaries_path is None:
            return
        if not self.gmm_boundaries_path.exists():
            return
        try:
            self.gmm_model = load_gmm_boundaries(self.gmm_boundaries_path)
            self.get_logger().info(
                f'loaded GMM boundaries from {self.gmm_boundaries_path}'
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(
                f'could not load GMM boundaries from {self.gmm_boundaries_path}: {error}'
            )

    @staticmethod
    def _wait_for_future(future, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if not future.done():
            return None
        return future.result()

    @staticmethod
    def _solver_diagnostic_fields(diagnostics):
        diagnostics = diagnostics or {}
        return {
            'solver_status': diagnostics.get('status', ''),
            'solver_success': diagnostics.get('success', ''),
            'solver_message': diagnostics.get('message', ''),
            'solver_cost': diagnostics.get('cost', ''),
            'solver_optimality': diagnostics.get('optimality', ''),
            'solver_nfev': diagnostics.get('nfev', ''),
            'solver_total_nfev': diagnostics.get('total_nfev', ''),
            'solver_rms_residual': diagnostics.get('rms_residual', ''),
            'solver_max_residual': diagnostics.get('max_residual', ''),
            'solver_boundary_solution': diagnostics.get('boundary_solution', ''),
            'solver_initial_latitude': diagnostics.get('initial_latitude', ''),
            'solver_initial_longitude': diagnostics.get('initial_longitude', ''),
            'solver_initial_heading_degrees': diagnostics.get('initial_heading', ''),
            'solver_selected_start_latitude': diagnostics.get('selected_start_latitude', ''),
            'solver_selected_start_longitude': diagnostics.get('selected_start_longitude', ''),
            'solver_selected_start_heading_degrees': diagnostics.get('selected_start_heading', ''),
            'solver_starts_tried': diagnostics.get('starts_tried', ''),
        }

    def _write_metrics_row(self, row):
        self._ensure_metrics_schema()
        file_exists = self.metrics_path.exists() and self.metrics_path.stat().st_size > 0
        with self.metrics_path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())

    def _append_failure(
        self,
        run_index,
        sample_ground_truth_latitude,
        sample_ground_truth_longitude,
        ground_truth_latitude,
        ground_truth_longitude,
        sample_timestamp,
        request_timestamp,
        request_yaw,
        failure_reason,
        *,
        diagnostics=None,
        sample_id='',
        sample_index='',
        repetition_index='',
        attempted_count='',
        valid_count='',
        identified_objects_used='',
        used_object_types=None,
        random_seed='',
        evaluation_config_json='',
        code_revision='',
        dependency_versions_json='',
    ):
        row = {field: '' for field in CSV_FIELDS}
        row.update({
            'run_index': run_index,
            'record_type': 'failure',
            'sample_id': sample_id,
            'sample_index': sample_index,
            'repetition_index': repetition_index,
            'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'sample_timestamp_utc': self._timestamp_to_utc(sample_timestamp),
            'request_timestamp_utc': self._timestamp_to_utc(request_timestamp),
            'request_yaw_degrees': request_yaw,
            'random_seed': random_seed,
            'evaluation_config_json': evaluation_config_json,
            'code_revision': code_revision,
            'dependency_versions_json': dependency_versions_json,
            'sample_ground_truth_latitude': sample_ground_truth_latitude,
            'sample_ground_truth_longitude': sample_ground_truth_longitude,
            'ground_truth_latitude': ground_truth_latitude,
            'ground_truth_longitude': ground_truth_longitude,
            'identified_objects_used': identified_objects_used,
            'used_object_types_json': json.dumps(
                used_object_types or {},
                separators=(',', ':'),
                sort_keys=True,
            ),
            'identified_star_ids': '[]',
            'attempted_count': attempted_count,
            'valid_count': valid_count,
            'failure_count': '',
            'valid_estimate': False,
            'failure_reason': failure_reason,
        })
        row.update(self._solver_diagnostic_fields(diagnostics))
        self._write_metrics_row(row)

    def _append_metrics(
        self,
        run_index,
        ground_truth_latitude,
        ground_truth_longitude,
        estimated_latitude,
        estimated_longitude,
        identified_objects_used,
        identified_star_ids,
        metrics,
        cumulative_mean_error_meters,
        *,
        record_type='summary',
        sample_id='',
        sample_index='',
        repetition_index='',
        is_outlier='',
        sample_timestamp_utc='',
        sample_ground_truth_latitude='',
        sample_ground_truth_longitude='',
        request_timestamp_utc='',
        request_yaw_degrees='',
        estimated_heading_degrees='',
        used_object_types=None,
        diagnostics=None,
        inlier_count='',
        outlier_count='',
        attempted_count='',
        valid_count='',
        failure_count='',
        valid_estimate=True,
        failure_reason='',
        random_seed='',
        evaluation_config_json='',
        code_revision='',
        dependency_versions_json='',
    ):
        self._load_gmm_model_if_available()
        cluster_id = None
        if self.gmm_model is not None:
            cluster_id = classify_sample(
                self.gmm_model,
                identified_objects_used,
                metrics['error_distance_meters'] / 1000.0,
            )
        row = {field: '' for field in CSV_FIELDS}
        row.update({
            'run_index': run_index,
            'record_type': record_type,
            'sample_id': sample_id,
            'sample_index': sample_index,
            'repetition_index': repetition_index,
            'is_outlier': is_outlier,
            'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'sample_timestamp_utc': sample_timestamp_utc,
            'sample_ground_truth_latitude': sample_ground_truth_latitude,
            'sample_ground_truth_longitude': sample_ground_truth_longitude,
            'request_timestamp_utc': request_timestamp_utc,
            'request_yaw_degrees': request_yaw_degrees,
            'random_seed': random_seed,
            'evaluation_config_json': evaluation_config_json,
            'code_revision': code_revision,
            'dependency_versions_json': dependency_versions_json,
            'ground_truth_latitude': ground_truth_latitude,
            'ground_truth_longitude': ground_truth_longitude,
            'estimated_latitude': estimated_latitude,
            'estimated_longitude': estimated_longitude,
            'estimated_heading_degrees': estimated_heading_degrees,
            'identified_objects_used': identified_objects_used,
            'used_object_types_json': json.dumps(
                used_object_types or {},
                separators=(',', ':'),
                sort_keys=True,
            ),
            'identified_star_ids': json.dumps(identified_star_ids, separators=(',', ':')),
            'cluster_id': cluster_id if cluster_id is not None else '',
            'inlier_count': inlier_count,
            'outlier_count': outlier_count,
            'attempted_count': attempted_count,
            'valid_count': valid_count,
            'failure_count': failure_count,
            'valid_estimate': valid_estimate,
            'failure_reason': failure_reason,
            'latitude_error_degrees': metrics['latitude_error_degrees'],
            'longitude_error_degrees': metrics['longitude_error_degrees'],
            'latitude_error_meters': metrics['latitude_error_meters'],
            'longitude_error_meters': metrics['longitude_error_meters'],
            'error_distance_meters': metrics['error_distance_meters'],
            'cumulative_mean_error_meters': cumulative_mean_error_meters,
        })
        row.update(self._solver_diagnostic_fields(diagnostics))
        self._write_metrics_row(row)
        return cluster_id

    def _save_cluster_image(
        self,
        cluster_id,
        sky_map_ready,
        ground_truth_latitude,
        ground_truth_longitude,
        estimated_latitude,
        estimated_longitude,
        error_distance_meters,
    ):
        if cluster_id is None or not sky_map_ready or self.sky_map_path is None:
            return
        if not self.sky_map_path.exists():
            self.get_logger().warning(
                f'cannot save cluster {cluster_id} image; {self.sky_map_path} is missing'
            )
            return

        image = cv2.imread(str(self.sky_map_path), cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warning(
                f'cannot annotate cluster {cluster_id} image; '
                f'{self.sky_map_path} could not be read'
            )
            return

        self._annotate_cluster_image(
            image,
            ground_truth_latitude,
            ground_truth_longitude,
            estimated_latitude,
            estimated_longitude,
            error_distance_meters,
        )
        cluster_path = self.debug_dir / cluster_filename(cluster_id)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f'.{cluster_path.stem}.',
            suffix='.png',
            dir=self.debug_dir,
        )
        os.close(temporary_fd)
        try:
            if not cv2.imwrite(temporary_name, image):
                raise OSError(f'failed to write {temporary_name}')
            os.replace(temporary_name, cluster_path)
            legacy_cluster_path = self.debug_dir / f'cluster{cluster_id}.png'
            try:
                legacy_cluster_path.unlink()
            except FileNotFoundError:
                pass
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        self.get_logger().info(f'saved classified sample to {cluster_path}')

    @staticmethod
    def _annotate_cluster_image(
        image,
        ground_truth_latitude,
        ground_truth_longitude,
        estimated_latitude,
        estimated_longitude,
        error_distance_meters,
    ):
        lines = (
            f'ERROR: {error_distance_meters:,.2f} m',
            f'GROUND TRUTH LATLON: {ground_truth_latitude:.6f}, {ground_truth_longitude:.6f}',
            f'EST LATLON: {estimated_latitude:.6f}, {estimated_longitude:.6f}',
        )
        height = image.shape[0]
        baseline = height - 24
        line_height = 32
        font = cv2.FONT_HERSHEY_SIMPLEX
        for line in reversed(lines):
            position = (24, baseline)
            cv2.putText(image, line, position, font, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(image, line, position, font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            baseline -= line_height

    def _ensure_metrics_schema(self):
        if not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0:
            return

        with self.metrics_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError(f'metrics file has no CSV header: {self.metrics_path}')
            if tuple(reader.fieldnames) == CSV_FIELDS:
                return
            rows = list(reader)

        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f'.{self.metrics_path.name}.',
            suffix='.tmp',
            dir=self.metrics_path.parent,
        )
        try:
            with os.fdopen(temporary_fd, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, '') for field in CSV_FIELDS})
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.metrics_path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = RandomLocalisationEvaluator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()