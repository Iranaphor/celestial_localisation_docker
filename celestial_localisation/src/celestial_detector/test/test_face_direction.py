import math

import numpy as np

from celestial_detector.face_direction import (
    face_pixel_to_direction,
    match_face_detections,
)


def _face():
    return {
        'center': [1.0, 0.0, 0.0],
        'right': [0.0, 1.0, 0.0],
        'up': [0.0, 0.0, 1.0],
    }


def _direction(azimuth_degrees, elevation_degrees=0.0):
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    cosine = math.cos(elevation)
    return np.asarray([
        math.cos(azimuth) * cosine,
        math.sin(azimuth) * cosine,
        math.sin(elevation),
    ])


def test_face_centroid_uses_original_perspective_geometry():
    direction = face_pixel_to_direction(_face(), 49.5, 49.5, 100, 100, 95.0)

    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-12)


def test_face_pixels_are_projected_from_pixel_centers():
    face = _face()
    width = height = 512
    field_of_view = 95.0
    tangent = math.tan(math.radians(field_of_view) / 2.0)
    expected = np.asarray([1.0, 0.31, 0.17])
    expected /= np.linalg.norm(expected)

    depth = float(np.dot(expected, face['center']))
    horizontal = float(np.dot(expected, face['right']))
    vertical = float(np.dot(expected, face['up']))
    # Stellarium's perspective project_to_win uses these NDC/window equations;
    # OpenCV centroid coordinates are one half-pixel before window coordinates.
    window_x = ((horizontal / depth / tangent) + 1.0) * 0.5 * width
    window_y = (1.0 - (vertical / depth / tangent)) * 0.5 * height
    pixel_x = window_x - 0.5
    pixel_y = window_y - 0.5

    direction = face_pixel_to_direction(
        face,
        pixel_x,
        pixel_y,
        width,
        height,
        field_of_view,
    )

    np.testing.assert_allclose(direction, expected, atol=1e-12)


def test_matching_accepts_mutual_observation_only_nearest_neighbours():
    panorama = [
        {'object_id': 'HIP_1', 'direction': _direction(0.0), 'brightness': 10.0},
        {'object_id': 'HIP_2', 'direction': _direction(90.0), 'brightness': 9.0},
    ]
    faces = [
        {'face_name': 'azimuth_0', 'direction': _direction(0.1), 'brightness': 10.0},
        {'face_name': 'azimuth_90', 'direction': _direction(89.9), 'brightness': 9.0},
    ]

    result = match_face_detections(panorama, faces, (2048, 1024))

    assert set(result['matches']) == {'HIP_1', 'HIP_2'}
    assert result['lost_object_ids'] == []
    assert result['ambiguous_object_ids'] == []


def test_matching_rejects_close_observed_candidates_as_ambiguous():
    panorama = [
        {'object_id': 'HIP_1', 'direction': _direction(0.0), 'brightness': 10.0},
        {'object_id': 'HIP_2', 'direction': _direction(0.1), 'brightness': 9.0},
    ]
    faces = [
        {'face_name': 'azimuth_0', 'direction': _direction(0.02), 'brightness': 10.0},
    ]

    result = match_face_detections(
        panorama,
        faces,
        (2048, 1024),
        ambiguity_margin_degrees=0.25,
    )

    assert result['matches'] == {}
    assert result['ambiguous_object_ids'] == ['HIP_1', 'HIP_2']
    assert result['lost_object_ids'] == []


def test_matching_merges_near_identical_overlap_detections_before_matching():
    panorama = [
        {'object_id': 'HIP_1', 'direction': _direction(0.0), 'brightness': 10.0},
    ]
    faces = [
        {'face_name': 'azimuth_0', 'direction': _direction(0.02), 'brightness': 10.0},
        {'face_name': 'azimuth_90', 'direction': _direction(0.04), 'brightness': 8.0},
    ]

    result = match_face_detections(
        panorama,
        faces,
        (2048, 1024),
        duplicate_radius_degrees=0.15,
    )

    assert set(result['matches']) == {'HIP_1'}
    assert result['deduplicated_face_detection_count'] == 1


