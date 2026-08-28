"""Camera calibration hooks.

For now this is a passthrough model (no lens distortion correction). Populate
`calibration_file` with a saved OpenCV camera matrix / distortion coefficients
to enable undistortion before panorama construction.
"""
import cv2
import numpy as np
import yaml


class Calibration:
    def __init__(self, calibration_file=""):
        self.camera_matrix = None
        self.dist_coeffs = None
        if calibration_file:
            self._load(calibration_file)

    def _load(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)

    def undistort(self, image):
        if self.camera_matrix is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
