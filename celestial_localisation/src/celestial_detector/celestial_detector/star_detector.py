"""Astronomical point-source (star) detection using photutils.

Bright, roughly circular, small point sources are found using DAOStarFinder.
Sun/moon disc rejection is handled here by masking regions supplied by the
caller before DAOStarFinder runs.
"""
import math

import cv2
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats


def detect_stars(
    gray_image,
    threshold_sigma=5.0,
    exclusion_detections=None,
    minimum_elevation_degrees=0.0,
    max_candidates=None,
):
    if not -90.0 <= float(minimum_elevation_degrees) <= 90.0:
        raise ValueError('minimum elevation must be between -90 and 90 degrees')

    exclusion_mask = np.zeros(gray_image.shape, dtype=np.uint8)
    for detection in exclusion_detections or []:
        radius = max(4.0, float(detection.get('radius', 4.0)) * 2.5)
        centre = (
            int(round(float(detection['pixel_x']))),
            int(round(float(detection['pixel_y']))),
        )
        cv2.circle(exclusion_mask, centre, int(math.ceil(radius)), True, -1)

    exclusion_mask_bool = exclusion_mask.astype(bool)
    horizon_row = int(np.ceil(
        (90.0 - float(minimum_elevation_degrees)) / 180.0 * gray_image.shape[0]
    ))
    below_horizon_mask = np.zeros(gray_image.shape, dtype=bool)
    below_horizon_mask[min(gray_image.shape[0], horizon_row):, :] = True
    stats_mask = exclusion_mask_bool | below_horizon_mask
    if not np.any(stats_mask):
        stats_mask = None
    _, median, std = sigma_clipped_stats(gray_image, sigma=3.0, mask=stats_mask)
    detection_image = gray_image.copy()
    detection_image[exclusion_mask_bool | below_horizon_mask] = median

    if not np.isfinite(std) or std <= 0.0:
        return []

    finder = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma * std)
    sources = finder(detection_image.astype(np.float64) - median)
    if sources is None:
        return []

    stars = []
    for row in sources:
        pixel_x = float(row['xcentroid'])
        pixel_y = float(row['ycentroid'])
        brightness = float(row['flux'])
        if pixel_y >= horizon_row or brightness <= 0.0:
            continue
        stars.append({
            'pixel_x': pixel_x,
            'pixel_y': pixel_y,
            'brightness': brightness,
            'confidence': min(1.0, brightness / (10.0 * std + 1e-6)),
        })

    stars.sort(key=lambda star: star['brightness'], reverse=True)
    if max_candidates is not None:
        stars = stars[:max(0, int(max_candidates))]
    return stars
