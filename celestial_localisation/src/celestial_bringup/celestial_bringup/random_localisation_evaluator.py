#!/usr/bin/env python3
import csv
import json
import math
import os
import random
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from celestial_interfaces.srv import LoadGps, RunRandomEvaluation


EARTH_RADIUS_METERS = 6_371_000.0
CSV_FIELDS = (
    'run_index',
    'timestamp_utc',
    'ground_truth_latitude',
    'ground_truth_longitude',
    'estimated_latitude',
    'estimated_longitude',
    'identified_objects_used',
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
        self.declare_parameter('fixed_altitude', 0.0)
        self.declare_parameter('result_timeout_seconds', 180.0)
        self.declare_parameter('poll_interval_seconds', 0.25)

        self.debug_dir = self._resolve_debug_dir()
        self.pose_path = self.debug_dir / self.get_parameter('pose_filename').value if self.debug_dir else None
        self.metrics_path = self.debug_dir / self.get_parameter('metrics_filename').value if self.debug_dir else None
        self.result_timeout_seconds = float(self.get_parameter('result_timeout_seconds').value)
        self.poll_interval_seconds = float(self.get_parameter('poll_interval_seconds').value)
        self.fixed_altitude = float(self.get_parameter('fixed_altitude').value)
        self._metrics_run_count, self._total_error_meters = self._load_metrics_state()

        self._random_source = random.SystemRandom()
        self._evaluation_lock = threading.Lock()
        self._last_timestamp_ns = 0
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

    def _load_metrics_state(self):
        if self.metrics_path is None or not self.metrics_path.exists():
            return 0, 0.0

        run_count = 0
        total_error_meters = 0.0
        try:
            with self.metrics_path.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    total_error_meters += float(row['error_distance_meters'])
                    run_count += 1
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.get_logger().warning(
                f'could not restore cumulative metrics from {self.metrics_path}: {error}'
            )
            return 0, 0.0
        return run_count, total_error_meters

    def _run_evaluation(self, request, response):
        response.completed_runs = 0
        repetitions = int(request.repetitions)
        if repetitions <= 0:
            response.success = False
            response.message = 'repetitions must be greater than zero'
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

            batch_total_error_meters = 0.0
            for batch_run_index in range(1, repetitions + 1):
                ground_truth_latitude, ground_truth_longitude = random_surface_location(self._random_source)
                timestamp = self._next_timestamp()
                load_request = LoadGps.Request()
                load_request.latitude = ground_truth_latitude
                load_request.longitude = ground_truth_longitude
                load_request.altitude = self.fixed_altitude
                load_request.timestamp = timestamp

                try:
                    load_response = self._wait_for_future(
                        self.load_gps_client.call_async(load_request),
                        self.result_timeout_seconds,
                    )
                except Exception as error:
                    response.success = False
                    response.message = f'LoadGps call failed on run {batch_run_index}: {error}'
                    return response

                if load_response is None:
                    response.success = False
                    response.message = f'LoadGps call timed out on run {batch_run_index}'
                    return response
                if not load_response.success:
                    response.success = False
                    response.message = f'LoadGps rejected run {batch_run_index}: {load_response.message}'
                    return response

                estimated = self._wait_for_pose(timestamp, self.result_timeout_seconds)
                if estimated is None:
                    response.success = False
                    response.message = f'pose.json did not complete for run {batch_run_index}'
                    return response

                estimated_latitude, estimated_longitude, identified_objects_used = estimated
                metrics = calculate_error_metrics(
                    ground_truth_latitude,
                    ground_truth_longitude,
                    estimated_latitude,
                    estimated_longitude,
                )
                batch_total_error_meters += metrics['error_distance_meters']
                run_index = self._metrics_run_count + 1
                self._total_error_meters += metrics['error_distance_meters']
                self._append_metrics(
                    run_index,
                    ground_truth_latitude,
                    ground_truth_longitude,
                    estimated_latitude,
                    estimated_longitude,
                    identified_objects_used,
                    metrics,
                    self._total_error_meters / run_index,
                )
                self._metrics_run_count = run_index
                response.completed_runs = batch_run_index
                self.get_logger().info(
                    f'random evaluation run {batch_run_index}/{repetitions}: '
                    f'ground_truth=({ground_truth_latitude:.6f}, {ground_truth_longitude:.6f}), '
                    f'estimated=({estimated_latitude:.6f}, {estimated_longitude:.6f}), '
                    f'identified_objects={identified_objects_used}, '
                    f'error={metrics["error_distance_meters"]:.2f} m, '
                    f'batch_mean={batch_total_error_meters / batch_run_index:.2f} m, '
                    f'cumulative_mean={self._total_error_meters / run_index:.2f} m'
                )

            response.success = True
            response.message = (
                f'completed {repetitions} random localisation runs; '
                f'batch mean error={batch_total_error_meters / repetitions:.2f} m; '
                f'cumulative mean error={self._total_error_meters / self._metrics_run_count:.2f} m; '
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
        timestamp_ns = timestamp.sec * 1_000_000_000 + timestamp.nanosec
        if timestamp_ns <= self._last_timestamp_ns:
            timestamp_ns = self._last_timestamp_ns + 1
        self._last_timestamp_ns = timestamp_ns
        timestamp.sec = timestamp_ns // 1_000_000_000
        timestamp.nanosec = timestamp_ns % 1_000_000_000
        return timestamp

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

        refined_estimate = payload.get('refined_estimate')
        if not isinstance(refined_estimate, dict):
            return None
        try:
            latitude = float(refined_estimate['latitude'])
            longitude = float(refined_estimate['longitude'])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not math.isfinite(latitude)
            or not math.isfinite(longitude)
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
        ):
            return None
        used_observations = payload.get('used_observations')
        if not isinstance(used_observations, list):
            return None
        return latitude, longitude, len(used_observations)

    @staticmethod
    def _wait_for_future(future, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if not future.done():
            return None
        return future.result()

    def _append_metrics(
        self,
        run_index,
        ground_truth_latitude,
        ground_truth_longitude,
        estimated_latitude,
        estimated_longitude,
        identified_objects_used,
        metrics,
        cumulative_mean_error_meters,
    ):
        self._ensure_metrics_schema()
        file_exists = self.metrics_path.exists() and self.metrics_path.stat().st_size > 0
        row = {
            'run_index': run_index,
            'timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'ground_truth_latitude': ground_truth_latitude,
            'ground_truth_longitude': ground_truth_longitude,
            'estimated_latitude': estimated_latitude,
            'estimated_longitude': estimated_longitude,
            'identified_objects_used': identified_objects_used,
            'latitude_error_degrees': metrics['latitude_error_degrees'],
            'longitude_error_degrees': metrics['longitude_error_degrees'],
            'latitude_error_meters': metrics['latitude_error_meters'],
            'longitude_error_meters': metrics['longitude_error_meters'],
            'error_distance_meters': metrics['error_distance_meters'],
            'cumulative_mean_error_meters': cumulative_mean_error_meters,
        }
        with self.metrics_path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())

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