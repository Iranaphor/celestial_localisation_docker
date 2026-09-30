#!/usr/bin/env python3
import json
import math
from pathlib import Path
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from celestial_interfaces.msg import (
    CelestialFaceArray,
    CelestialObservation,
    CelestialObservationArray,
)

from celestial_detector.angular_projection import pixel_to_az_el
from celestial_detector.star_detector import detect_stars
from celestial_detector.star_identifier import StarIdentifier, identify_stars
from celestial_detector.sun_detector import detect_sun
from celestial_detector.moon_detector import detect_moon
from celestial_detector.point_source_classification import detections_overlap
from celestial_detector.transient_detector import classify_point_sources
from celestial_detector.gmm_classifier import (
    classify_live_observations,
    cluster_filename,
    load_gmm_boundaries,
)
from celestial_detector.face_direction import (
    direction_to_azimuth_elevation,
    face_pixel_to_direction,
    match_face_detections,
)

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
        self.declare_parameter('face_input_topic', '/celestial_faces')
        self.declare_parameter('direction_source', 'panorama')
        self.declare_parameter('face_match_radius_degrees', 2.0)
        self.declare_parameter('face_match_ambiguity_margin_degrees', 0.25)
        self.declare_parameter('face_duplicate_radius_degrees', 0.15)
        self.declare_parameter('face_sync_timeout_seconds', 1.0)
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
        self.declare_parameter('star_benefit_filter_enabled', True)
        self.declare_parameter(
            'star_benefit_metrics_filename',
            'simulated_location_filter_samples*.csv',
        )
        self.declare_parameter('star_benefit_min_score', -0.1)
        self.declare_parameter('star_benefit_min_present_count', 3)
        self.declare_parameter('star_benefit_min_absent_count', 2)
        self.declare_parameter('star_benefit_use_high_error_group', True)
        self.declare_parameter('star_benefit_use_medium_error_group', False)

        input_topic = self.get_parameter('input_topic').value
        face_input_topic = self.get_parameter('face_input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.direction_source = str(self.get_parameter('direction_source').value).lower()
        if self.direction_source not in ('panorama', 'original_face'):
            raise ValueError("direction_source must be 'panorama' or 'original_face'")
        self.face_match_radius_degrees = float(
            self.get_parameter('face_match_radius_degrees').value
        )
        self.face_match_ambiguity_margin_degrees = float(
            self.get_parameter('face_match_ambiguity_margin_degrees').value
        )
        self.face_duplicate_radius_degrees = float(
            self.get_parameter('face_duplicate_radius_degrees').value
        )
        self.face_sync_timeout_seconds = float(
            self.get_parameter('face_sync_timeout_seconds').value
        )
        self.do_stars = self.get_parameter('detect_stars').value
        self.do_sun = self.get_parameter('detect_sun').value
        self.do_moon = self.get_parameter('detect_moon').value
        self.star_threshold = self.get_parameter('star_detection_threshold').value
        self.star_min_elevation = self.get_parameter('star_min_elevation_degrees').value
        self.star_max_candidates = self.get_parameter('star_max_candidates').value
        self.min_confidence = self.get_parameter('minimum_confidence').value
        debug_output_dir = self.get_parameter('debug_output_dir').value
        configured_metrics_path = Path(
            self.get_parameter('star_benefit_metrics_filename').value
        )
        if self.get_parameter('star_benefit_filter_enabled').value:
            self.star_benefit_metrics_path = (
                configured_metrics_path
                if configured_metrics_path.is_absolute() or not debug_output_dir
                else Path(debug_output_dir) / configured_metrics_path
            )
        else:
            self.star_benefit_metrics_path = None
        self.star_identifier = StarIdentifier(
            database_path=self.get_parameter('star_database_path').value,
            fov_degrees=self.get_parameter('star_fov_degrees').value,
            fov_max_error_degrees=self.get_parameter('star_fov_max_error_degrees').value,
            tile_size=self.get_parameter('star_tile_size').value,
            match_radius=self.get_parameter('star_match_radius').value,
            match_threshold=self.get_parameter('star_match_threshold').value,
            min_matches=self.get_parameter('star_min_matches').value,
            star_benefit_metrics_path=(
                str(self.star_benefit_metrics_path)
                if self.star_benefit_metrics_path is not None
                else ''
            ),
            star_benefit_min_score=self.get_parameter('star_benefit_min_score').value,
            star_benefit_min_present_count=self.get_parameter(
                'star_benefit_min_present_count'
            ).value,
            star_benefit_min_absent_count=self.get_parameter(
                'star_benefit_min_absent_count'
            ).value,
            star_benefit_use_high_error_group=self.get_parameter(
                'star_benefit_use_high_error_group'
            ).value,
            star_benefit_use_medium_error_group=self.get_parameter(
                'star_benefit_use_medium_error_group'
            ).value,
        )

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
        self.face_sub = None
        self.face_timer = None
        self.face_bundles = {}
        self.pending_sky_maps = {}
        if self.direction_source == 'original_face':
            if not face_input_topic:
                raise ValueError('face_input_topic is required for original_face direction source')
            self.face_sub = self.create_subscription(
                CelestialFaceArray,
                face_input_topic,
                self._on_face_bundle,
                10,
            )
            self.face_timer = self.create_timer(0.05, self._expire_pending_sky_maps)

        self.get_logger().info(
            f"celestial_detector listening on {input_topic}, publishing on {output_topic}"
            f", direction source={self.direction_source}"
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
        if self.star_identifier.star_benefit_error:
            self.get_logger().warning(self.star_identifier.star_benefit_error)
        elif self.star_identifier.excluded_star_ids:
            excluded_star_ids = ', '.join(sorted(self.star_identifier.excluded_star_ids))
            source_paths = ', '.join(
                str(path) for path in self.star_identifier.star_benefit_source_paths
            )
            self.get_logger().info(
                f"star benefit filter excluded {len(self.star_identifier.excluded_star_ids)} "
                f"catalogue stars using {len(self.star_identifier.star_benefit_source_paths)} "
                f"files ({source_paths}): {excluded_star_ids}"
            )
        else:
            source_paths = ', '.join(
                str(path) for path in self.star_identifier.star_benefit_source_paths
            ) or str(self.star_benefit_metrics_path)
            self.get_logger().info(
                f"star benefit filter found no exclusions in {source_paths}"
            )

    @staticmethod
    def _stamp_key(header):
        return int(header.stamp.sec), int(header.stamp.nanosec)

    def _on_face_bundle(self, msg):
        key = self._stamp_key(msg.header)
        self.face_bundles[key] = msg
        pending = self.pending_sky_maps.pop(key, None)
        if pending is not None:
            self._process_sky_map(pending[0], msg)

    def _expire_pending_sky_maps(self):
        now = time.monotonic()
        expired = [
            key for key, (_, received_at) in self.pending_sky_maps.items()
            if now - received_at >= self.face_sync_timeout_seconds
        ]
        for key in expired:
            msg, _ = self.pending_sky_maps.pop(key)
            self.get_logger().error(
                f'no face bundle arrived for sky map timestamp {key}; '
                'dropping original-face observation'
            )
            self._process_sky_map(msg, None)

    def _on_sky_map(self, msg):
        if self.direction_source != 'original_face':
            self._process_sky_map(msg, None)
            return

        key = self._stamp_key(msg.header)
        face_bundle = self.face_bundles.pop(key, None)
        if face_bundle is None:
            self.pending_sky_maps[key] = (msg, time.monotonic())
            return
        self._process_sky_map(msg, face_bundle)

    def _process_sky_map(self, msg, face_bundle):
        self.get_logger().info(f"received sky map ({msg.width}x{msg.height})")
        if self.direction_source == 'original_face' and face_bundle is None:
            return
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        observations = []

        face_sources = None
        if self.direction_source == 'original_face':
            face_sources = self._detect_face_sources(face_bundle)
            self.get_logger().info(
                f"received {len(face_bundle.faces)} face images; "
                f"face detections stars={len(face_sources['stars'])}, "
                f"fov={face_bundle.field_of_view_degrees:.3f}"
            )

        sun = detect_sun(gray) if self.do_sun else None
        if sun and sun['confidence'] >= self.min_confidence:
            observations.append(self._build_observation(
                sun, width, height, CelestialObservation.SUN, 'SUN'
            ))

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
            observations.append(self._build_observation(
                moon, width, height, CelestialObservation.MOON, 'MOON'
            ))

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
            face_matching = None
            if self.direction_source == 'original_face':
                face_detections = face_sources['stars']
                face_matching = match_face_detections(
                    stars,
                    face_detections,
                    (width, height),
                    max_distance_degrees=self.face_match_radius_degrees,
                    ambiguity_margin_degrees=self.face_match_ambiguity_margin_degrees,
                    duplicate_radius_degrees=self.face_duplicate_radius_degrees,
                )
                self.get_logger().info(
                    f"face matching accepted {len(face_matching['matches'])}/"
                    f"{face_matching['panorama_id_count']} panorama star IDs; "
                    f"lost={len(face_matching['lost_object_ids'])}, "
                    f"ambiguous={len(face_matching['ambiguous_object_ids'])}"
                )
            for star in stars:
                direction_override = None
                if face_matching is not None:
                    match = face_matching['matches'].get(star.get('object_id'))
                    if match is None:
                        continue
                    direction_override = match['face_detection']['direction']
                observations.append(
                    self._build_observation(
                        star,
                        width,
                        height,
                        CelestialObservation.STAR,
                        star['object_id'],
                        direction_override=direction_override,
                    )
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

    def _build_observation(
        self,
        detection,
        width,
        height,
        object_type,
        object_id,
        direction_override=None,
    ):
        if direction_override is None:
            azimuth, elevation = pixel_to_az_el(
                detection['pixel_x'], detection['pixel_y'], width, height
            )
        else:
            azimuth, elevation = direction_to_azimuth_elevation(direction_override)
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

    def _detect_face_sources(self, face_bundle):
        stars = []
        for face in face_bundle.faces:
            image = self.bridge.imgmsg_to_cv2(face.image, desired_encoding='bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sun = detect_sun(gray) if self.do_sun else None
            moon = detect_moon(gray) if self.do_moon else None
            if (
                sun
                and moon
                and sun['confidence'] >= self.min_confidence
                and detections_overlap(sun, moon)
            ):
                moon = None
            face_stars = detect_stars(
                gray,
                self.star_threshold,
                [detection for detection in (sun, moon) if detection],
                minimum_elevation_degrees=-90.0,
                max_candidates=self.star_max_candidates,
            )
            face_stars = classify_point_sources(face_stars)
            for detection in face_stars:
                if detection['confidence'] < self.min_confidence:
                    continue
                direction = face_pixel_to_direction(
                    face,
                    detection['pixel_x'],
                    detection['pixel_y'],
                    image.shape[1],
                    image.shape[0],
                    face_bundle.field_of_view_degrees,
                )
                _, elevation = direction_to_azimuth_elevation(direction)
                if elevation < self.star_min_elevation:
                    continue
                stars.append({
                    **detection,
                    'face_name': face.name,
                    'direction': direction,
                })
        return {'stars': stars}

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
                cluster_image_path = self.debug_dir / cluster_filename(provisional_cluster_id)
                if not cv2.imwrite(str(cluster_image_path), annotated):
                    raise OSError(f"failed to write {cluster_image_path}")
                legacy_cluster_path = self.debug_dir / f"cluster{provisional_cluster_id}.png"
                try:
                    legacy_cluster_path.unlink()
                except FileNotFoundError:
                    pass

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
