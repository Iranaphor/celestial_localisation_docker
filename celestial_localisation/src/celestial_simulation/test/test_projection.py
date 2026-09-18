import math

import numpy as np

from celestial_simulation.projection import compose_equirectangular, direction_from_angles


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
