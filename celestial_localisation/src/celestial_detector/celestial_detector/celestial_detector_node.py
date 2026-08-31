#!/usr/bin/env python3
import json
from pathlib import Path

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

_MARKER_COLOR_BGR = {
    'SUN': (0, 215, 255),
    'MOON': (220, 220, 220),
    'STAR': (0, 255, 0),
}

_TYPE_NAMES = {
    CelestialObservation.SUN: 'SUN',
    CelestialObservation.MOON: 'MOON',
    CelestialObservation.STAR: 'STAR',
}


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
        self.declare_parameter('debug_output_dir', '')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.do_stars = self.get_parameter('detect_stars').value
        self.do_sun = self.get_parameter('detect_sun').value
        self.do_moon = self.get_parameter('detect_moon').value
        self.star_threshold = self.get_parameter('star_detection_threshold').value
        self.min_confidence = self.get_parameter('minimum_confidence').value

        debug_output_dir = self.get_parameter('debug_output_dir').value
        self.debug_dir = Path(debug_output_dir) if debug_output_dir else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, input_topic, self._on_sky_map, 10)
        self.pub = self.create_publisher(CelestialObservationArray, output_topic, 10)

        self.get_logger().info(
            f"celestial_detector listening on {input_topic}, publishing on {output_topic}"
            + (f", saving debug output to {self.debug_dir}" if self.debug_dir else "")
        )

    def _on_sky_map(self, msg):
        self.get_logger().info(f"received sky map ({msg.width}x{msg.height})")
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

        for obs in observations:
            type_name = _TYPE_NAMES.get(obs.object_type, 'UNKNOWN')
            self.get_logger().info(
                f"detected {type_name} '{obs.object_id}' pixel=({obs.pixel_x:.0f},{obs.pixel_y:.0f}) "
                f"az={obs.azimuth:.2f} el={obs.elevation:.2f} confidence={obs.confidence:.2f} "
                f"brightness={obs.brightness:.1f}"
            )

        sun_count = sum(1 for o in observations if o.object_type == CelestialObservation.SUN)
        moon_count = sum(1 for o in observations if o.object_type == CelestialObservation.MOON)
        star_count = sum(1 for o in observations if o.object_type == CelestialObservation.STAR)

        out = CelestialObservationArray()
        out.header = msg.header
        out.observations = observations
        self.pub.publish(out)
        self.get_logger().info(
            f"published {len(observations)} observations (sun={sun_count}, moon={moon_count}, "
            f"star={star_count}) to celestial_localizer"
        )

        self._save_debug_output(image, observations, msg.header)

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

    def _save_debug_output(self, image_bgr, observations, header):
        if self.debug_dir is None:
            return

        stamp = f"{header.stamp.sec}_{header.stamp.nanosec:09d}"

        annotated = image_bgr.copy()
        for obs in observations:
            type_name = _TYPE_NAMES.get(obs.object_type, 'UNKNOWN')
            color = _MARKER_COLOR_BGR.get(type_name, (255, 255, 255))
            center = (int(obs.pixel_x), int(obs.pixel_y))
            cv2.circle(annotated, center, 10, color, 2)
            cv2.putText(
                annotated, f"{obs.object_id} {obs.confidence:.2f}",
                (center[0] + 12, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        image_path = self.debug_dir / f"sky_map_{stamp}.png"
        cv2.imwrite(str(image_path), annotated)

        observations_payload = [
            {
                'object_type': _TYPE_NAMES.get(obs.object_type, 'UNKNOWN'),
                'object_id': obs.object_id,
                'azimuth': obs.azimuth,
                'elevation': obs.elevation,
                'confidence': obs.confidence,
                'pixel_x': obs.pixel_x,
                'pixel_y': obs.pixel_y,
                'brightness': obs.brightness,
            }
            for obs in observations
        ]
        observations_path = self.debug_dir / f"observations_{stamp}.json"
        with observations_path.open('w', encoding='utf-8') as stream:
            json.dump(observations_payload, stream, indent=2)

        self.get_logger().info(f"saved annotated image and observations to {self.debug_dir}")


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
