"""Astronomical point-source (star) detection using photutils.

Bright, roughly circular, small point sources are found using DAOStarFinder.
Sun/moon disc rejection is handled by the caller (celestial_detector_node)
by masking out already-claimed regions before this runs.
"""
import cv2
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats


def detect_stars(gray_image, threshold_sigma=5.0):
    mean, median, std = sigma_clipped_stats(gray_image, sigma=3.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=threshold_sigma * std)
    sources = finder(gray_image.astype(np.float64) - median)
    if sources is None:
        return []

    stars = []
    for row in sources:
        stars.append({
            'pixel_x': float(row['xcentroid']),
            'pixel_y': float(row['ycentroid']),
            'brightness': float(row['flux']),
            'confidence': min(1.0, float(row['flux']) / (10.0 * std + 1e-6)),
        })
    return stars
