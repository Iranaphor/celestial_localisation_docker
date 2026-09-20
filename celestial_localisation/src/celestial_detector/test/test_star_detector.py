import numpy as np
import cv2

from celestial_detector.star_detector import detect_stars


def _noisy_image():
    generator = np.random.default_rng(42)
    return np.clip(generator.normal(20.0, 2.0, (100, 200)), 0, 255).astype(np.uint8)


def test_detector_rejects_sources_below_minimum_elevation():
    image = _noisy_image()
    cv2.circle(image, (30, 20), 3, 255, -1)
    cv2.circle(image, (150, 80), 3, 255, -1)

    detections = detect_stars(
        image,
        threshold_sigma=1.0,
        minimum_elevation_degrees=0.0,
    )

    assert detections
    assert all(detection['pixel_y'] < 50 for detection in detections)


def test_detector_returns_brightest_candidates_first():
    image = _noisy_image()
    cv2.circle(image, (30, 20), 3, 180, -1)
    cv2.circle(image, (100, 20), 3, 255, -1)
    cv2.circle(image, (170, 20), 3, 220, -1)

    detections = detect_stars(
        image,
        threshold_sigma=1.0,
        minimum_elevation_degrees=0.0,
        max_candidates=2,
    )

    assert len(detections) <= 2
    assert detections[0]['brightness'] >= detections[-1]['brightness']


def test_detector_rejects_invalid_elevation():
    image = _noisy_image()

    try:
        detect_stars(image, minimum_elevation_degrees=91.0)
    except ValueError as error:
        assert 'minimum elevation' in str(error)
    else:
        raise AssertionError('expected invalid elevation to raise ValueError')