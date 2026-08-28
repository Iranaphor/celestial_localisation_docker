"""Pixel <-> az/el conversion shared with sky_mapper's equirectangular convention."""


def pixel_to_az_el(x, y, width, height):
    azimuth = (x / float(width)) * 360.0
    elevation = 90.0 - (y / float(height)) * 180.0
    return azimuth, elevation
