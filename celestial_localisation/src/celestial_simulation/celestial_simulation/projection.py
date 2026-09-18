"""Projection helpers shared by the simulator and its tests."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceView:
    name: str
    center: np.ndarray
    right: np.ndarray
    up: np.ndarray


def direction_from_angles(azimuth_radians, elevation_radians):
    cos_elevation = math.cos(elevation_radians)
    return np.array([
        math.cos(azimuth_radians) * cos_elevation,
        math.sin(azimuth_radians) * cos_elevation,
        math.sin(elevation_radians),
    ], dtype=np.float64)


def _horizon_face(name, azimuth_radians):
    center = direction_from_angles(azimuth_radians, 0.0)
    right = np.array([
        -math.sin(azimuth_radians),
        math.cos(azimuth_radians),
        0.0,
    ], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return FaceView(name, center, right, up)


def cube_faces():
    return (
        _horizon_face('azimuth_0', 0.0),
        _horizon_face('azimuth_90', math.pi / 2.0),
        _horizon_face('azimuth_180', math.pi),
        _horizon_face('azimuth_270', 3.0 * math.pi / 2.0),
        FaceView(
            'zenith',
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        ),
        FaceView(
            'nadir',
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        ),
    )


def compose_equirectangular(faces, width, height, field_of_view_degrees=95.0):
    if width <= 0 or height <= 0:
        raise ValueError('panorama dimensions must be positive')
    if not 90.0 < field_of_view_degrees < 179.0:
        raise ValueError('face field of view must be between 90 and 179 degrees')

    face_by_name = dict(faces)
    expected_names = {face.name for face in cube_faces()}
    if set(face_by_name) != expected_names:
        missing = sorted(expected_names - set(face_by_name))
        extra = sorted(set(face_by_name) - expected_names)
        raise ValueError(f'face set mismatch; missing={missing}, extra={extra}')

    first_image = next(iter(face_by_name.values()))
    if first_image.ndim != 3 or first_image.shape[2] != 3:
        raise ValueError('face images must be three-channel BGR images')
    face_height, face_width = first_image.shape[:2]
    if face_width <= 1 or face_height <= 1:
        raise ValueError('face images must be at least 2x2 pixels')
    for image in face_by_name.values():
        if image.shape != first_image.shape or image.dtype != np.uint8:
            raise ValueError('all face images must have the same uint8 shape')

    x = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 * math.pi
    y = (np.arange(height, dtype=np.float64) + 0.5) / height * math.pi
    azimuth, elevation = np.meshgrid(x, math.pi / 2.0 - y)
    cos_elevation = np.cos(elevation)
    directions = np.stack((
        np.cos(azimuth) * cos_elevation,
        np.sin(azimuth) * cos_elevation,
        np.sin(elevation),
    ), axis=-1).reshape(-1, 3)

    tangent = math.tan(math.radians(field_of_view_degrees) / 2.0)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    best_score = np.full(directions.shape[0], -np.inf, dtype=np.float64)
    filled = np.zeros(directions.shape[0], dtype=bool)

    for face in cube_faces():
        image = face_by_name[face.name]
        depth = directions @ face.center
        horizontal = directions @ face.right
        vertical = directions @ face.up
        visible = (
            (depth > 0.0)
            & (np.abs(horizontal) <= depth * tangent)
            & (np.abs(vertical) <= depth * tangent)
        )
        replace = visible & (depth > best_score)
        if not np.any(replace):
            continue

        map_x = ((horizontal / depth / tangent) + 1.0) * 0.5 * (face_width - 1)
        map_y = (1.0 - (vertical / depth / tangent)) * 0.5 * (face_height - 1)
        sampled = cv2.remap(
            image,
            map_x.reshape(height, width).astype(np.float32),
            map_y.reshape(height, width).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1, 3)
        result.reshape(-1, 3)[replace] = sampled[replace]
        best_score[replace] = depth[replace]
        filled[replace] = True

    if not np.all(filled):
        raise ValueError('face field of view leaves gaps in the equirectangular panorama')
    return result
