#!/usr/bin/env python3
import math
import os
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from celestial_interfaces.srv import LoadGps

from celestial_simulation.renderer import StellariumRenderer
from celestial_simulation.stellarium_browser import RendererError


class CelestialSimulationNode(Node):
    def __init__(self):
        super().__init__('celestial_simulation_node')

        self.declare_parameter('output_topic', '/sky_map')
        self.declare_parameter('load_gps_service', '/test/load_gps')
        self.declare_parameter('frame_id', 'sky_map')
        self.declare_parameter('panorama_width', 2048)
        self.declare_parameter('panorama_height', 1024)
        self.declare_parameter('face_size', 512)
        self.declare_parameter('face_field_of_view', 95.0)
        self.declare_parameter('render_timeout_seconds', 60.0)
        self.declare_parameter(
            'debug_output_dir',
            os.environ.get('CELESTIAL_DEBUG_OUTPUT_DIR', ''),
        )
        self.declare_parameter(
            'engine_js',
            os.environ.get(
                'CELESTIAL_STELLARIUM_ENGINE_JS',
                '/opt/stellarium/stellarium-web-engine.js',
            ),
        )
        self.declare_parameter(
            'engine_wasm',
            os.environ.get(
                'CELESTIAL_STELLARIUM_ENGINE_WASM',
                '/opt/stellarium/stellarium-web-engine.wasm',
            ),
        )
        self.declare_parameter(
            'data_root',
            os.environ.get('CELESTIAL_STELLARIUM_DATA_ROOT', '/opt/stellarium/data'),
        )
        self.declare_parameter('browser_executable', os.environ.get('CELESTIAL_BROWSER_EXECUTABLE', ''))

        output_topic = self.get_parameter('output_topic').value
        service_name = self.get_parameter('load_gps_service').value
        self.frame_id = self.get_parameter('frame_id').value
        debug_output_dir = self.get_parameter('debug_output_dir').value
        self.debug_dir = Path(debug_output_dir) if debug_output_dir else None
        if self.debug_dir:
            try:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                self.get_logger().error(
                    f'cannot create debug output directory {self.debug_dir}: {error}'
                )
                self.debug_dir = None

        self.publisher = self.create_publisher(Image, output_topic, 10)
        self.service = self.create_service(LoadGps, service_name, self._load_gps)
        self.renderer = StellariumRenderer(
            engine_js=self.get_parameter('engine_js').value,
            engine_wasm=self.get_parameter('engine_wasm').value,
            data_root=self.get_parameter('data_root').value,
            browser_executable=self.get_parameter('browser_executable').value,
            face_size=self.get_parameter('face_size').value,
            timeout_seconds=self.get_parameter('render_timeout_seconds').value,
            panorama_width=self.get_parameter('panorama_width').value,
            panorama_height=self.get_parameter('panorama_height').value,
            face_field_of_view=self.get_parameter('face_field_of_view').value,
        )
        self.get_logger().info(
            f'celestial_simulation publishing generated sky maps on {output_topic}; '
            f'GPS service is {service_name}'
            + (f', saving debug output to {self.debug_dir}' if self.debug_dir else '')
        )

    def _load_gps(self, request, response):
        error = self._validate_request(request)
        if error:
            response.success = False
            response.message = error
            return response

        timestamp_ms = request.timestamp.sec * 1000.0 + request.timestamp.nanosec / 1e6
        try:
            image = self.renderer.render(
                latitude=request.latitude,
                longitude=request.longitude,
                altitude=request.altitude,
                timestamp_ms=timestamp_ms,
            )
        except (RendererError, RuntimeError, OSError, ValueError) as error:
            self.get_logger().error(f'failed to render requested sky map: {error}')
            response.success = False
            response.message = f'failed to render sky map: {error}'
            return response
        except Exception as error:
            self.get_logger().error(f'unexpected simulator rendering failure: {error}')
            response.success = False
            response.message = f'unexpected rendering failure: {error}'
            return response

        message = Image()
        message.header.stamp = request.timestamp
        message.header.frame_id = self.frame_id
        message.height = image.shape[0]
        message.width = image.shape[1]
        message.encoding = 'bgr8'
        message.is_bigendian = False
        message.step = image.shape[1] * 3
        message.data = image.tobytes()
        self.publisher.publish(message)
        self._save_debug_output(image, message.header)

        response.success = True
        response.message = (
            f'published {message.width}x{message.height} sky map for '
            f'latitude={request.latitude:.6f}, longitude={request.longitude:.6f}, '
            f'altitude={request.altitude:.2f} m'
        )
        return response

    def _save_debug_output(self, image, header):
        if self.debug_dir is None:
            return

        stamp = f'{header.stamp.sec}_{header.stamp.nanosec:09d}'
        image_path = self.debug_dir / f'simulated_sky_map_{stamp}.png'
        try:
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f'failed to write {image_path}')
        except (OSError, cv2.error) as error:
            self.get_logger().error(f'disabling debug output after write failure: {error}')
            self.debug_dir = None
            return

        self.get_logger().info(f'saved simulated sky map to {image_path}')

    @staticmethod
    def _validate_request(request):
        if not math.isfinite(request.latitude) or not -90.0 <= request.latitude <= 90.0:
            return 'latitude must be finite and between -90 and 90 degrees'
        if not math.isfinite(request.longitude) or not -180.0 <= request.longitude <= 180.0:
            return 'longitude must be finite and between -180 and 180 degrees'
        if not math.isfinite(request.altitude):
            return 'altitude must be finite'
        if request.timestamp.nanosec < 0 or request.timestamp.nanosec >= 1_000_000_000:
            return 'timestamp nanosec must be between 0 and 999999999'
        return ''

    def destroy_node(self):
        self.renderer.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CelestialSimulationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
