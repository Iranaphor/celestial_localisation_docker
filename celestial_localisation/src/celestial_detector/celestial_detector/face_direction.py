"""Face projection and observation-only face-to-panorama matching helpers."""
import math

import numpy as np

from celestial_detector.angular_projection import pixel_to_az_el


def _vector(face, name):
    value = getattr(face, name, None)
    if value is None:
        value = face[name]
    return np.asarray(value, dtype=np.float64)


def face_pixel_to_direction(face, pixel_x, pixel_y, width, height, field_of_view_degrees):
    """Convert a pixel-center face centroid into a local direction vector."""
    if width <= 1 or height <= 1:
        raise ValueError('face image dimensions must be greater than one')
    if not 90.0 < float(field_of_view_degrees) < 179.0:
        raise ValueError('face field of view must be between 90 and 179 degrees')

    center = _vector(face, 'center')
    right = _vector(face, 'right')
    up = _vector(face, 'up')
    tangent = math.tan(math.radians(float(field_of_view_degrees)) / 2.0)
    horizontal = (2.0 * (float(pixel_x) + 0.5) / width - 1.0) * tangent
    vertical = (1.0 - 2.0 * (float(pixel_y) + 0.5) / height) * tangent
    direction = center + horizontal * right + vertical * up
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError('face orientation produced an invalid direction')
    return direction / norm


def direction_to_azimuth_elevation(direction):
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError('direction must be finite and non-zero')
    direction = direction / norm
    return (
        math.degrees(math.atan2(direction[1], direction[0])) % 360.0,
        math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0)))),
    )


def angular_separation_degrees(first, second):
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _deduplicate_face_detections(face_detections, duplicate_radius_degrees):
    ordered = sorted(
        enumerate(face_detections),
        key=lambda item: item[1].get('brightness', 0.0),
        reverse=True,
    )
    retained = []
    source_indices = []
    for source_index, detection in ordered:
        direction = np.asarray(detection['direction'], dtype=np.float64)
        if any(
            angular_separation_degrees(direction, retained_direction)
            <= duplicate_radius_degrees
            for retained_direction in retained
        ):
            continue
        retained.append(direction)
        source_indices.append(source_index)
    return [face_detections[index] for index in source_indices]


def _unique_panorama_detections(panorama_detections, panorama_size):
    width, height = panorama_size
    by_id = {}
    for detection in panorama_detections:
        object_id = str(detection.get('object_id', '')).strip()
        if not object_id or object_id.upper() == 'UNKNOWN':
            continue
        direction = detection.get('direction')
        if direction is None:
            azimuth, elevation = pixel_to_az_el(
                detection['pixel_x'], detection['pixel_y'], width, height
            )
            azimuth_radians = math.radians(azimuth)
            elevation_radians = math.radians(elevation)
            cosine = math.cos(elevation_radians)
            direction = np.asarray((
                math.cos(azimuth_radians) * cosine,
                math.sin(azimuth_radians) * cosine,
                math.sin(elevation_radians),
            ), dtype=np.float64)
        candidate = dict(detection)
        candidate['direction'] = np.asarray(direction, dtype=np.float64)
        previous = by_id.get(object_id)
        if previous is None or candidate.get('brightness', 0.0) > previous.get('brightness', 0.0):
            by_id[object_id] = candidate
    return by_id


def match_face_detections(
    panorama_detections,
    face_detections,
    panorama_size,
    max_distance_degrees=2.0,
    ambiguity_margin_degrees=0.25,
    duplicate_radius_degrees=0.15,
):
    """Match face centroids to identified panorama directions only.

    A match is accepted only when it is a mutual nearest neighbour with a
    clear angular margin on both sides.  Near-identical detections from
    overlapping cube faces are merged before ambiguity checks.  No catalogue
    prediction or ground-truth direction is used here.
    """
    panorama_by_id = _unique_panorama_detections(panorama_detections, panorama_size)
    deduplicated_faces = _deduplicate_face_detections(
        face_detections,
        float(duplicate_radius_degrees),
    )
    object_ids = sorted(panorama_by_id)
    if not object_ids or not deduplicated_faces:
        return {
            'matches': {},
            'lost_object_ids': object_ids,
            'ambiguous_object_ids': [],
            'panorama_id_count': len(object_ids),
            'face_detection_count': len(face_detections),
            'deduplicated_face_detection_count': len(deduplicated_faces),
        }

    panorama_directions = np.asarray([
        panorama_by_id[object_id]['direction'] for object_id in object_ids
    ])
    face_directions = np.asarray([
        detection['direction'] for detection in deduplicated_faces
    ])
    dots = np.clip(panorama_directions @ face_directions.T, -1.0, 1.0)
    distances = np.degrees(np.arccos(dots))
    within_radius = distances <= float(max_distance_degrees)

    panorama_nearest = {}
    ambiguous = set()
    for row, object_id in enumerate(object_ids):
        candidates = np.flatnonzero(within_radius[row])
        if not len(candidates):
            continue
        ordered = candidates[np.argsort(distances[row, candidates])]
        nearest = int(ordered[0])
        panorama_nearest[object_id] = nearest
        if len(ordered) > 1:
            margin = distances[row, ordered[1]] - distances[row, nearest]
            if margin < float(ambiguity_margin_degrees):
                ambiguous.add(object_id)

    face_nearest = {}
    for column in range(len(deduplicated_faces)):
        candidates = np.flatnonzero(within_radius[:, column])
        if not len(candidates):
            continue
        ordered = candidates[np.argsort(distances[candidates, column])]
        nearest = int(ordered[0])
        face_nearest[column] = object_ids[nearest]
        if len(ordered) > 1:
            margin = distances[ordered[1], column] - distances[nearest, column]
            if margin < float(ambiguity_margin_degrees):
                ambiguous.update(object_ids[index] for index in ordered)

    matches = {}
    for object_id, face_index in panorama_nearest.items():
        if object_id in ambiguous:
            continue
        if face_nearest.get(face_index) != object_id:
            continue
        matches[object_id] = {
            'face_detection': deduplicated_faces[face_index],
            'angular_distance_degrees': float(
                distances[object_ids.index(object_id), face_index]
            ),
        }

    lost = [
        object_id for object_id in object_ids
        if object_id not in matches and object_id not in ambiguous
    ]
    return {
        'matches': matches,
        'lost_object_ids': lost,
        'ambiguous_object_ids': sorted(ambiguous),
        'panorama_id_count': len(object_ids),
        'face_detection_count': len(face_detections),
        'deduplicated_face_detection_count': len(deduplicated_faces),
    }


