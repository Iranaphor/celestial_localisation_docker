"""Classical computer-vision Sun detector.

Finds the brightest large saturated blob in the image and fits its centroid
and radius, which approximates the solar disc centre.
"""
import cv2
import numpy as np


def detect_sun(gray_image, min_area=20):
    _, thresh = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    (x, y), radius = cv2.minEnclosingCircle(largest)
    return {
        'pixel_x': float(x),
        'pixel_y': float(y),
        'brightness': float(gray_image.max()),
        'confidence': min(1.0, cv2.contourArea(largest) / (np.pi * radius * radius + 1e-6)),
        'radius': float(radius),
    }
