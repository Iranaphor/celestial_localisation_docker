import random

from celestial_bringup.random_localisation_evaluator import (
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