"""Rules for resolving overlaps between bright point-source detections."""
import math


def detections_overlap(first, second, minimum_separation_pixels=4.0, padding_pixels=2.0):
    """Return whether two detections describe the same image region."""
    if first is None or second is None:
        return False

    delta_x = float(first['pixel_x']) - float(second['pixel_x'])
    delta_y = float(first['pixel_y']) - float(second['pixel_y'])
    distance = math.hypot(delta_x, delta_y)
    first_radius = max(0.0, float(first.get('radius', 0.0)))
    second_radius = max(0.0, float(second.get('radius', 0.0)))
    allowed_distance = max(
        float(minimum_separation_pixels),
        first_radius + second_radius + float(padding_pixels),
    )
    return distance <= allowed_distance