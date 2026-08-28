"""Sky panorama construction.

Stage 1 assumes the source camera already provides a wide/equirectangular
view of the sky (e.g. a 360 camera or fisheye lens), so panorama
construction is a passthrough resize onto the configured panorama
dimensions. Multi-frame stitching (cv2.Stitcher) can be added here once
multiple simultaneous camera feeds are available.
"""
import cv2


def build_panorama(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
