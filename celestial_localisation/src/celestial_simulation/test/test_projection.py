import math

import numpy as np

from celestial_simulation.projection import (
    compose_equirectangular,
    cube_faces,
    direction_from_angles,
)


def test_direction_matches_equirectangular_convention():
    direction = direction_from_angles(math.pi / 2.0, 0.0)
    np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-12)


def test_cube_faces_cover_a_small_panorama():
    faces = {
        'azimuth_0': np.full((8, 8, 3), [1, 2, 3], dtype=np.uint8),
        'azimuth_90': np.full((8, 8, 3), [4, 5, 6], dtype=np.uint8),
        'azimuth_180': np.full((8, 8, 3), [7, 8, 9], dtype=np.uint8),
        'azimuth_270': np.full((8, 8, 3), [10, 11, 12], dtype=np.uint8),
        'zenith': np.full((8, 8, 3), [13, 14, 15], dtype=np.uint8),
        'nadir': np.full((8, 8, 3), [16, 17, 18], dtype=np.uint8),
    }
    panorama = compose_equirectangular(faces, 32, 16)
    assert panorama.shape == (16, 32, 3)
    assert panorama.dtype == np.uint8
    assert np.all(panorama.sum(axis=2) > 0)


def test_compositor_preserves_continuous_directions_at_seams_and_face_boundaries():
    face_size = 128
    panorama_width = 128
    panorama_height = 64
    field_of_view_degrees = 95.0
    tangent = math.tan(math.radians(field_of_view_degrees) / 2.0)
    coordinates = np.linspace(-1.0, 1.0, face_size, dtype=np.float64)
    horizontal, vertical = np.meshgrid(coordinates * tangent, -coordinates * tangent)

    faces = {}
    for face in cube_faces():
        directions = (
            face.center
            + horizontal[..., None] * face.right
            + vertical[..., None] * face.up
        )
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        faces[face.name] = np.clip(
            (directions + 1.0) * 127.5,
            0.0,
            255.0,
        ).astype(np.uint8)

    panorama = compose_equirectangular(
        faces,
        panorama_width,
        panorama_height,
        field_of_view_degrees,
    )

    for pixel_y in (0, 1, 31, 32, 63):
        for pixel_x in (0, 1, 31, 32, 63, 64, 127):
            azimuth = (pixel_x + 0.5) / panorama_width * 2.0 * math.pi
            elevation = math.pi / 2.0 - (pixel_y + 0.5) / panorama_height * math.pi
            expected = direction_from_angles(azimuth, elevation)
            measured = panorama[pixel_y, pixel_x].astype(np.float64) / 127.5 - 1.0
            measured /= np.linalg.norm(measured)
            angular_error = math.acos(float(np.clip(np.dot(expected, measured), -1.0, 1.0)))
            assert angular_error < math.radians(2.0)
