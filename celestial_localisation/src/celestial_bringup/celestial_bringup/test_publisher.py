#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from celestial_interfaces.msg import CelestialObservation, CelestialObservationArray
from celestial_interfaces.srv import PublishImageFile


class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')

        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('sky_map_topic', '/sky_map')
        self.declare_parameter('observations_topic', '/celestial_observations')
        self.declare_parameter('sample_observations_file', '')
        self.declare_parameter('camera_image_file', '')
        self.declare_parameter('sky_map_image_file', '')
        self.declare_parameter('sky_map_image_dir', '')
        self.declare_parameter('capture_timestamp', '')

        self.camera_pub = self.create_publisher(
            Image, self.get_parameter('camera_topic').value, 10
        )
        self.sky_map_pub = self.create_publisher(
            Image, self.get_parameter('sky_map_topic').value, 10
        )
        self.observations_pub = self.create_publisher(
            CelestialObservationArray, self.get_parameter('observations_topic').value, 10
        )
        self.bridge = CvBridge()
        self.sample_file = self._resolve_sample_file()

        self.create_service(Trigger, '/test/publish_camera_image', self._publish_camera)
        self.create_service(PublishImageFile, '/test/publish_sky_map', self._publish_sky_map)
        self.create_service(Trigger, '/test/publish_observations', self._publish_observations)
        self.create_service(Trigger, '/test/publish_all', self._publish_all)

        self.get_logger().info(
            'test_publisher ready: publish_camera_image, publish_sky_map, '
            'publish_observations, publish_all'
        )

    def _resolve_sample_file(self):
        configured = self.get_parameter('sample_observations_file').value
        if configured:
            return Path(configured)
        return Path(get_package_share_directory('celestial_bringup')) / 'data' / 'sample_observations.json'

    def _header(self):
        stamp = self.get_clock().now().to_msg()
        return stamp

    def _publish_image(self, image, publisher, stamp):
        message = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        message.header.stamp = stamp
        publisher.publish(message)

    def _load_image(self, image_file):
        if not image_file:
            return None, 'no image file configured'
        path = Path(image_file)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None, f"could not read image file '{path}'"
        return image, None

    def _resolve_sky_map_path(self, filename):
        if filename:
            image_dir = self.get_parameter('sky_map_image_dir').value
            return str(Path(image_dir) / filename) if image_dir else filename
        return self.get_parameter('sky_map_image_file').value

    def _resolve_stamp(self):
        configured = self.get_parameter('capture_timestamp').value
        if configured:
            return self._parse_timestamp(configured)
        return self._header()

    def _publish_camera(self, request, response):
        image, error = self._load_image(self.get_parameter('camera_image_file').value)
        if image is None:
            response.success = False
            response.message = f'could not publish camera image: {error}'
            return response

        stamp = self._resolve_stamp()
        self._publish_image(image, self.camera_pub, stamp)
        self.get_logger().info(
            f"published camera image ({image.shape[1]}x{image.shape[0]}) "
            f"on {self.camera_pub.topic_name} with stamp sec={stamp.sec}"
        )
        response.success = True
        response.message = 'test camera image published'
        return response

    def _publish_sky_map(self, request, response):
        image, error = self._load_image(self._resolve_sky_map_path(request.filename))
        if image is None:
            response.success = False
            response.message = f'could not publish sky map: {error}'
            return response

        stamp = self._resolve_stamp()
        self._publish_image(image, self.sky_map_pub, stamp)
        self.get_logger().info(
            f"published sky map ({image.shape[1]}x{image.shape[0]}) "
            f"on {self.sky_map_pub.topic_name} with stamp sec={stamp.sec}"
        )
        response.success = True
        response.message = 'test sky map published'
        return response

    def _publish_observations(self, request, response):
        try:
            with self.sample_file.open(encoding='utf-8') as stream:
                sample = json.load(stream)
            message = self._observation_message(sample)
        except (OSError, ValueError, KeyError) as error:
            response.success = False
            response.message = f'could not load observations: {error}'
            return response

        self.observations_pub.publish(message)
        self.get_logger().info(
            f"published {len(message.observations)} test observations on {self.observations_pub.topic_name}"
        )
        response.success = True
        response.message = f'{len(message.observations)} test observations published'
        return response

    def _publish_all(self, request, response):
        self._publish_camera(request, Trigger.Response())
        self._publish_sky_map(PublishImageFile.Request(), PublishImageFile.Response())
        observation_response = self._publish_observations(request, Trigger.Response())
        response.success = observation_response.success
        response.message = 'test camera image, sky map, and observations published'
        if not response.success:
            response.message = observation_response.message
        return response

    def _observation_message(self, sample):
        message = CelestialObservationArray()
        message.header.stamp = self._parse_timestamp(sample.get('timestamp'))
        for item in sample['observations']:
            observation = CelestialObservation()
            observation.object_type = getattr(CelestialObservation, item['object_type'])
            observation.object_id = item['object_id']
            observation.azimuth = float(item['azimuth'])
            observation.elevation = float(item['elevation'])
            observation.angular_uncertainty = float(item.get('angular_uncertainty', 0.5))
            observation.confidence = float(item.get('confidence', 1.0))
            observation.pixel_x = float(item.get('pixel_x', 0.0))
            observation.pixel_y = float(item.get('pixel_y', 0.0))
            observation.brightness = float(item.get('brightness', 0.0))
            message.observations.append(observation)
        return message

    @staticmethod
    def _parse_timestamp(value):
        if not value:
            return Time()
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        timestamp_ns = int(parsed.timestamp() * 1_000_000_000)
        result = Time()
        result.sec = timestamp_ns // 1_000_000_000
        result.nanosec = timestamp_ns % 1_000_000_000
        return result


def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()