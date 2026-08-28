"""Pixel <-> celestial-sphere projection helpers for an equirectangular sky map.

These assume the sky map horizontal axis spans azimuth [0, 360) degrees and
the vertical axis spans elevation [+90 (zenith) .. -90] degrees. Replace with
a calibrated model (lens distortion, mounting offset, etc.) as calibration
data becomes available.
"""
import math


def pixel_to_az_el(x, y, width, height):
    azimuth = (x / float(width)) * 360.0
    elevation = 90.0 - (y / float(height)) * 180.0
    return azimuth, elevation


def az_el_to_pixel(azimuth, elevation, width, height):
    x = (azimuth % 360.0) / 360.0 * width
    y = (90.0 - elevation) / 180.0 * height
    return x, y


def az_el_to_unit_vector(azimuth, elevation):
    az = math.radians(azimuth)
    el = math.radians(elevation)
    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)
    return x, y, z


def pixel_to_unit_vector(x, y, width, height):
    azimuth, elevation = pixel_to_az_el(x, y, width, height)
    return az_el_to_unit_vector(azimuth, elevation)
