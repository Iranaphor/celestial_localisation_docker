import json
import random

from celestial_bringup.random_localisation_evaluator import (
    RandomLocalisationEvaluator,
    _timestamp_from_nanoseconds,
    _extract_identified_star_ids,
    average_pose_estimates,
    calculate_error_metrics,
    perturb_surface_location,
)


def test_extract_identified_star_ids_excludes_non_star_observations():
    observations = [
        {'object_id': 'sun'},
        {'object_id': 'HIP_200'},
        {'object_id': 'HIP_100'},
        {'object_id': 'HIP_200'},
        {'object_id': 'UNKNOWN'},
        {'object_id': 'moon'},
        {'object_id': None},
    ]

    assert _extract_identified_star_ids(observations) == ['HIP_100', 'HIP_200']


def test_perturb_surface_location_stays_within_requested_radius():
    latitude = 35.0
    longitude = -120.0
    maximum_distance = 200.0

    varied_latitude, varied_longitude = perturb_surface_location(
        latitude,
        longitude,
        maximum_distance,
        random.Random(7),
    )
    distance = calculate_error_metrics(
        latitude,
        longitude,
        varied_latitude,
        varied_longitude,
    )['error_distance_meters']

    assert distance <= maximum_distance + 1e-6


def test_average_pose_estimates_excludes_spatial_outlier():
    estimates = [
        {
            'latitude': 35.0,
            'longitude': -120.0,
            'heading': 10.0,
            'identified_objects_used': 4,
            'identified_star_ids': ['HIP_100', 'HIP_200'],
            'sky_map_ready': True,
        },
        {
            'latitude': 35.000001,
            'longitude': -120.000001,
            'heading': 11.0,
            'identified_objects_used': 5,
            'identified_star_ids': ['HIP_200', 'HIP_300'],
            'sky_map_ready': True,
        },
        {
            'latitude': 34.999999,
            'longitude': -119.999999,
            'heading': 12.0,
            'identified_objects_used': 4,
            'identified_star_ids': ['HIP_100'],
            'sky_map_ready': True,
        },
        {
            'latitude': 40.0,
            'longitude': -115.0,
            'heading': 180.0,
            'identified_objects_used': 1,
            'identified_star_ids': ['HIP_OUTLIER'],
            'sky_map_ready': False,
        },
    ]

    averaged = average_pose_estimates(estimates, 35.0, -120.0)

    assert averaged['inlier_count'] == 3
    assert averaged['outlier_count'] == 1
    assert averaged['inlier_indices'] == [0, 1, 2]
    assert abs(averaged['latitude'] - 35.0) < 1e-5
    assert abs(averaged['longitude'] + 120.0) < 1e-5
    assert averaged['identified_objects_used'] == 4
    assert averaged['identified_star_ids'] == ['HIP_100', 'HIP_200', 'HIP_300']
    assert averaged['sky_map_ready'] is True


def test_average_pose_estimates_uses_median_for_sample_coordinates():
    estimates = [
        {
            'latitude': 35.0,
            'longitude': -120.0,
            'heading': 10.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
        {
            'latitude': 35.000001,
            'longitude': -120.0,
            'heading': 10.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
        {
            'latitude': 35.000002,
            'longitude': -120.0,
            'heading': 10.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
        {
            'latitude': 35.000002,
            'longitude': -120.0,
            'heading': 10.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
    ]

    averaged = average_pose_estimates(estimates, 35.0, -120.0)

    assert averaged['inlier_indices'] == [1, 2, 3]
    assert abs(averaged['latitude'] - 35.000002) < 1e-9
    assert abs(averaged['longitude'] + 120.0) < 1e-9


def test_average_pose_estimates_does_not_use_ground_truth_for_inlier_selection():
    estimates = [
        {
            'latitude': 35.0,
            'longitude': 10.0,
            'heading': 0.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
        {
            'latitude': 35.000001,
            'longitude': 10.000001,
            'heading': 0.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
        {
            'latitude': 35.000002,
            'longitude': 10.000002,
            'heading': 0.0,
            'identified_objects_used': 4,
            'identified_star_ids': [],
            'sky_map_ready': True,
        },
        {
            'latitude': 80.0,
            'longitude': 10.0,
            'heading': 0.0,
            'identified_objects_used': 1,
            'identified_star_ids': ['HIP_OUTLIER'],
            'sky_map_ready': False,
        },
    ]

    averaged = average_pose_estimates(estimates, 90.0, 0.0)

    assert averaged['inlier_indices'] == [0, 1, 2]


def test_read_pose_preserves_timestamped_solver_failure(tmp_path):
    timestamp = _timestamp_from_nanoseconds(123456789)
    pose_path = tmp_path / 'pose.json'
    pose_path.write_text(json.dumps({
        'valid': False,
        'observation_timestamp': {
            'sec': timestamp.sec,
            'nanosec': timestamp.nanosec,
        },
        'solver': {
            'status': -1,
            'failure_reason': 'insufficient_observations',
        },
    }), encoding='utf-8')

    evaluator = RandomLocalisationEvaluator.__new__(RandomLocalisationEvaluator)
    evaluator.pose_path = pose_path

    result = evaluator._read_pose(timestamp)

    assert result['valid'] is False
    assert result['failure_reason'] == 'insufficient_observations'
    assert result['diagnostics']['status'] == -1