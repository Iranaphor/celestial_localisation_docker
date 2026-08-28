#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from celestial_interfaces.msg import CelestialObservation, CelestialObservationArray

from celestial_detector.angular_projection import pixel_to_az_el
from celestial_detector.star_detector import detect_stars
from celestial_detector.star_identifier import identify_stars
from celestial_detector.sun_detector import detect_sun
from celestial_detector.moon_detector import detect_moon
from celestial_detector.transient_detector import classify_point_sources


class CelestialDetectorNode(Node):
    def __init__(self):
        super().__init__('celestial_detector_node')

        self.declare_parameter('input_topic', '/sky_map')
        self.declare_parameter('output_topic', '/celestial_observations')
        self.declare_parameter('detect_stars', True)
        self.declare_parameter('detect_sun', True)
        self.declare_parameter('detect_moon', True)
        self.declare_parameter('star_detection_threshold', 5.0)
        self.declare_parameter('minimum_confidence', 0.5)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.do_stars = self.get_parameter('detect_stars').value
        self.do_sun = self.get_parameter('detect_sun').value
        self.do_moon = self.get_parameter('detect_moon').value
        self.star_threshold = self.get_parameter('star_detection_threshold').value
        self.min_confidence = self.get_parameter('minimum_confidence').value

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, input_topic, self._on_sky_map, 10)
        self.pub = self.create_publisher(CelestialObservationArray, output_topic, 10)

        self.get_logger().info(
            f"celestial_detector listening on {input_topic}, publishing on {output_topic}"
        )

    def _on_sky_map(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        observations = []

        sun = detect_sun(gray) if self.do_sun else None
        if sun and sun['confidence'] >= self.min_confidence:
            observations.append(self._build_observation(sun, width, height, CelestialObservation.SUN, 'SUN'))

        moon = detect_moon(gray) if self.do_moon else None
        if moon and moon['confidence'] >= self.min_confidence:
            observations.append(self._build_observation(moon, width, height, CelestialObservation.MOON, 'MOON'))

        if self.do_stars:
            stars = detect_stars(gray, self.star_threshold)
            stars = classify_point_sources(stars)
            stars = identify_stars(stars)
            for star in stars:
                if star['confidence'] < self.min_confidence:
                    continue
                observations.append(
                    self._build_observation(star, width, height, CelestialObservation.STAR, star['object_id'])
                )

        out = CelestialObservationArray()
        out.header = msg.header
        out.observations = observations
        self.pub.publish(out)

    def _build_observation(self, detection, width, height, object_type, object_id):
        azimuth, elevation = pixel_to_az_el(detection['pixel_x'], detection['pixel_y'], width, height)
        obs = CelestialObservation()
        obs.object_type = object_type
        obs.object_id = object_id
        obs.azimuth = azimuth
        obs.elevation = elevation
        obs.angular_uncertainty = 0.5
        obs.confidence = detection['confidence']
        obs.pixel_x = detection['pixel_x']
        obs.pixel_y = detection['pixel_y']
        obs.brightness = detection['brightness']
        return obs


def main(args=None):
    rclpy.init(args=args)
    node = CelestialDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
