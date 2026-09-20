from celestial_interfaces.msg import CelestialObservation

from celestial_detector.celestial_detector_node import _marker_color


def test_unknown_star_marker_is_red():
    assert _marker_color(CelestialObservation.STAR, 'UNKNOWN') == (0, 0, 255)


def test_identified_star_marker_stays_green():
    assert _marker_color(CelestialObservation.STAR, 'HIP_91262') == (0, 255, 0)