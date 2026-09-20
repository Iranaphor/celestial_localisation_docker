from celestial_detector.point_source_classification import detections_overlap


def test_same_saturated_region_is_classified_as_overlap():
    sun = {'pixel_x': 100.0, 'pixel_y': 200.0, 'radius': 2.0}
    moon = {'pixel_x': 100.5, 'pixel_y': 200.5, 'radius': 1.5}

    assert detections_overlap(sun, moon)


def test_separated_bodies_are_not_classified_as_overlap():
    sun = {'pixel_x': 100.0, 'pixel_y': 200.0, 'radius': 2.0}
    moon = {'pixel_x': 110.0, 'pixel_y': 200.0, 'radius': 1.5}

    assert not detections_overlap(sun, moon)


def test_missing_detection_cannot_overlap():
    assert not detections_overlap(None, {'pixel_x': 0.0, 'pixel_y': 0.0})