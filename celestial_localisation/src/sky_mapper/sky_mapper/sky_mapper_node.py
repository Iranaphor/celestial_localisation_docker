#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from sky_mapper.calibration import Calibration
from sky_mapper.panorama import build_panorama


class SkyMapperNode(Node):
    def __init__(self):
        super().__init__('sky_mapper_node')

        self.declare_parameter('input_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('output_topic', '/sky_map')
        self.declare_parameter('panorama_width', 2048)
        self.declare_parameter('panorama_height', 1024)
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('publish_continuously', True)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.width = self.get_parameter('panorama_width').value
        self.height = self.get_parameter('panorama_height').value
        self.publish_continuously = self.get_parameter('publish_continuously').value

        self.bridge = CvBridge()
        self.calibration = Calibration(self.get_parameter('calibration_file').value)
        self.latest_frame = None

        self.sub = self.create_subscription(Image, input_topic, self._on_image, 10)
        self.pub = self.create_publisher(Image, output_topic, 10)
        self.capture_srv = self.create_service(Trigger, 'capture_sky_map', self._on_capture)

        self.get_logger().info(
            f"sky_mapper listening on {input_topic}, publishing on {output_topic}"
        )

    def _on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = self.calibration.undistort(frame)
        self.latest_frame = (frame, msg.header)
        if self.publish_continuously:
            self._publish_sky_map()

    def _publish_sky_map(self):
        if self.latest_frame is None:
            return
        frame, header = self.latest_frame
        panorama = build_panorama(frame, self.width, self.height)
        out_msg = self.bridge.cv2_to_imgmsg(panorama, encoding='bgr8')
        out_msg.header = header
        self.pub.publish(out_msg)

    def _on_capture(self, request, response):
        if self.latest_frame is None:
            response.success = False
            response.message = 'no frame received yet'
            return response
        self._publish_sky_map()
        response.success = True
        response.message = 'sky map published'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SkyMapperNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
