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
from celestial_detector.star_identifier import StarIdentifier, identify_stars
from celestial_detector.sun_detector import detect_sun
from celestial_detector.moon_detector import detect_moon
from celestial_detector.point_source_classification import detections_overlap
from celestial_detector.transient_detector import classify_point_sources
from celestial_detector.gmm_classifier import classify_live_observations, load_gmm_boundaries

_MARKER_COLOR_BGR = {
    'SUN': (0, 215, 255),
    'MOON': (220, 220, 220),
    'STAR': (0, 255, 0),
}
_UNKNOWN_STAR_COLOR_BGR = (0, 0, 255)

_TYPE_NAMES = {
    CelestialObservation.SUN: 'SUN',
    CelestialObservation.MOON: 'MOON',
    CelestialObservation.STAR: 'STAR',
}


def _marker_color(object_type, object_id):
    type_name = _TYPE_NAMES.get(object_type, 'UNKNOWN')
    if type_name == 'STAR' and str(object_id).strip().upper() == 'UNKNOWN':
        return _UNKNOWN_STAR_COLOR_BGR
    return _MARKER_COLOR_BGR.get(type_name, (255, 255, 255))


class CelestialDetectorNode(Node):
    def __init__(self):
        super().__init__('celestial_detector_node')

        self.declare_parameter('input_topic', '/sky_map')
        self.declare_parameter('output_topic', '/celestial_observations')
        self.declare_parameter('detect_stars', True)
        self.declare_parameter('detect_sun', True)
        self.declare_parameter('detect_moon', True)
        self.declare_parameter('star_detection_threshold', 5.0)
        self.declare_parameter('star_min_elevation_degrees', 5.0)
        self.declare_parameter('star_max_candidates', 80)
        self.declare_parameter('star_database_path', '')
        self.declare_parameter('star_fov_degrees', 30.0)
        self.declare_parameter('star_fov_max_error_degrees', 3.0)
        self.declare_parameter('star_tile_size', 512)
        self.declare_parameter('star_match_radius', 0.02)
        self.declare_parameter('star_match_threshold', 0.001)
        self.declare_parameter('star_min_matches', 4)
        self.declare_parameter('minimum_confidence', 0.5)
        self.declare_parameter('debug_output_dir', '')
        self.declare_parameter('gmm_boundaries_filename', 'gmm_boundaries.json')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.do_stars = self.get_parameter('detect_stars').value
        self.do_sun = self.get_parameter('detect_sun').value
        self.do_moon = self.get_parameter('detect_moon').value
        self.star_threshold = self.get_parameter('star_detection_threshold').value
        self.star_min_elevation = self.get_parameter('star_min_elevation_degrees').value
        self.star_max_candidates = self.get_parameter('star_max_candidates').value
        self.min_confidence = self.get_parameter('minimum_confidence').value
        self.star_identifier = StarIdentifier(
            database_path=self.get_parameter('star_database_path').value,
            fov_degrees=self.get_parameter('star_fov_degrees').value,
            fov_max_error_degrees=self.get_parameter('star_fov_max_error_degrees').value,
            tile_size=self.get_parameter('star_tile_size').value,
            match_radius=self.get_parameter('star_match_radius').value,
            match_threshold=self.get_parameter('star_match_threshold').value,
            min_matches=self.get_parameter('star_min_matches').value,
        )

        debug_output_dir = self.get_parameter('debug_output_dir').value
        self.debug_dir = Path(debug_output_dir) if debug_output_dir else None
        if self.debug_dir:
            try:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                self.get_logger().error(f"cannot create debug output directory {self.debug_dir}: {error}")
                self.debug_dir = None

        self.gmm_boundaries_path = None
        self.gmm_model = None
        if self.debug_dir:
            configured_boundaries = Path(
                self.get_parameter('gmm_boundaries_filename').value
            )
            self.gmm_boundaries_path = (
                configured_boundaries
                if configured_boundaries.is_absolute()
                else self.debug_dir / configured_boundaries
            )
            if self.gmm_boundaries_path.exists():
                try:
                    self.gmm_model = load_gmm_boundaries(self.gmm_boundaries_path)
                    self.get_logger().info(
                        f"loaded GMM boundaries from {self.gmm_boundaries_path}"
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                    self.get_logger().warning(
                        f"could not load GMM boundaries from {self.gmm_boundaries_path}: {error}"
                    )
            else:
                self.get_logger().warning(
                    f"GMM boundaries not found at {self.gmm_boundaries_path}; "
                    "live samples will remain unclassified"
                )

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, input_topic, self._on_sky_map, 10)
        self.pub = self.create_publisher(CelestialObservationArray, output_topic, 10)

        self.get_logger().info(
            f"celestial_detector listening on {input_topic}, publishing on {output_topic}"
            + (f", saving debug output to {self.debug_dir}" if self.debug_dir else "")
        )
        if self.star_identifier.enabled:
            self.get_logger().info(
                f"offline star identification enabled using {self.star_identifier.catalogue_name}"
            )
        else:
            self.get_logger().warning(
                f"offline star identification disabled: {self.star_identifier.error or 'no solver'}"
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
        if (
            sun
            and moon
            and sun['confidence'] >= self.min_confidence
            and detections_overlap(sun, moon)
        ):
            self.get_logger().warning(
                'moon detection overlaps the accepted sun detection; suppressing duplicate moon observation'
            )
            moon = None
        if moon and moon['confidence'] >= self.min_confidence:
            observations.append(self._build_observation(moon, width, height, CelestialObservation.MOON, 'MOON'))

        if self.do_stars:
            stars = detect_stars(
                gray,
                self.star_threshold,
                [detection for detection in (sun, moon) if detection],
                self.star_min_elevation,
                self.star_max_candidates,
            )
            stars = classify_point_sources(stars)
            stars = [star for star in stars if star['confidence'] >= self.min_confidence]
            stars = identify_stars(stars, (width, height), self.star_identifier)
            for star in stars:
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

    def _load_gmm_model_if_available(self):
        if self.gmm_model is not None or self.gmm_boundaries_path is None:
            return
        if not self.gmm_boundaries_path.exists():
            return
        try:
            self.gmm_model = load_gmm_boundaries(self.gmm_boundaries_path)
            self.get_logger().info(
                f"loaded GMM boundaries from {self.gmm_boundaries_path}"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(
                f"could not load GMM boundaries from {self.gmm_boundaries_path}: {error}"
            )

    def _save_debug_output(self, image_bgr, observations, header):
        if self.debug_dir is None:
            return

        try:
            self._load_gmm_model_if_available()
            annotated = image_bgr.copy()
            for obs in observations:
                color = _marker_color(obs.object_type, obs.object_id)
                center = (int(obs.pixel_x), int(obs.pixel_y))
                cv2.circle(annotated, center, 10, color, 2)
                if obs.object_id != 'UNKNOWN':
                    cv2.putText(
                        annotated, f"{obs.object_id} {obs.confidence:.2f}",
                        (center[0] + 12, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                    )

            provisional_cluster_id = None
            if self.gmm_model is not None:
                provisional_cluster_id = classify_live_observations(
                    self.gmm_model,
                    len(observations),
                )
            image_path = self.debug_dir / "sky_map.png"
            if not cv2.imwrite(str(image_path), annotated):
                raise OSError(f"failed to write {image_path}")
            if provisional_cluster_id is not None:
                cluster_image_path = self.debug_dir / f"cluster{provisional_cluster_id}.png"
                if not cv2.imwrite(str(cluster_image_path), annotated):
                    raise OSError(f"failed to write {cluster_image_path}")

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
            observations_path = self.debug_dir / "observations.json"
            with observations_path.open('w', encoding='utf-8') as stream:
                json.dump(observations_payload, stream, indent=2)

            classification_payload = {
                'observation_timestamp': {
                    'sec': int(header.stamp.sec),
                    'nanosec': int(header.stamp.nanosec),
                },
                'identified_objects_used': len(observations),
                'provisional_cluster_id': provisional_cluster_id,
                'classification': (
                    'observation-marginal'
                    if provisional_cluster_id is not None
                    else 'unclassified'
                ),
            }
            classification_path = self.debug_dir / "sky_map_classification.json"
            with classification_path.open('w', encoding='utf-8') as stream:
                json.dump(classification_payload, stream, indent=2)
        except OSError as error:
            self.get_logger().error(f"disabling debug output after write failure: {error}")
            self.debug_dir = None
            return

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
