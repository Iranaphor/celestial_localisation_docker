import math

import pytest

from celestial_detector.angular_projection import pixel_to_az_el
from sky_mapper.projection import az_el_to_pixel


PANORAMA_WIDTH = 2048
PANORAMA_HEIGHT = 1024


@pytest.mark.parametrize(
    'azimuth, elevation',
    (
        (0.0, 90.0),
        (0.001, 89.999),
        (123.456789, 12.345678),
        (180.0, 0.0),
        (359.999, -89.999),
    ),
)
def test_pixel_and_angle_mappings_round_trip_subpixel_directions(azimuth, elevation):
    pixel_x, pixel_y = az_el_to_pixel(
        azimuth,
        elevation,
        PANORAMA_WIDTH,
        PANORAMA_HEIGHT,
    )

    measured_azimuth, measured_elevation = pixel_to_az_el(
        pixel_x,
        pixel_y,
        PANORAMA_WIDTH,
        PANORAMA_HEIGHT,
    )

    assert math.isclose(measured_azimuth, azimuth, abs_tol=1e-12)
    assert math.isclose(measured_elevation, elevation, abs_tol=1e-12)


@pytest.mark.parametrize('pixel_x', (0.0, 0.125, 1023.375, 2047.0, 2047.875))
@pytest.mark.parametrize('pixel_y', (0.125, 511.625, 1023.0))
def test_detector_mapping_preserves_arbitrary_centroid_coordinates(pixel_x, pixel_y):
    azimuth, elevation = pixel_to_az_el(
        pixel_x,
        pixel_y,
        PANORAMA_WIDTH,
        PANORAMA_HEIGHT,
    )
    round_trip_x, round_trip_y = az_el_to_pixel(
        azimuth,
        elevation,
        PANORAMA_WIDTH,
        PANORAMA_HEIGHT,
    )

    assert math.isclose(round_trip_x, pixel_x, abs_tol=1e-12)
    assert math.isclose(round_trip_y, pixel_y, abs_tol=1e-12)


def test_seam_coordinates_remain_wrapped_by_angle_to_pixel_mapping():
    pixel_x, pixel_y = az_el_to_pixel(
        360.0,
        0.0,
        PANORAMA_WIDTH,
        PANORAMA_HEIGHT,
    )

    assert pixel_x == 0.0
    assert pixel_y == PANORAMA_HEIGHT / 2.0
    assert pixel_to_az_el(pixel_x, pixel_y, PANORAMA_WIDTH, PANORAMA_HEIGHT) == (
        0.0,
        0.0,
    )