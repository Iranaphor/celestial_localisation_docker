#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from sensor_msgs.msg import NavSatFix
from tf2_ros import TransformBroadcaster

from celestial_interfaces.msg import CelestialObservation, CelestialObservationArray

from celestial_localizer.ephemeris import EphemerisProvider
from celestial_localizer.pose_solver import solve
from celestial_localizer.uncertainty import estimate_covariance

_OBJECT_ID_BY_TYPE = {
    CelestialObservation.SUN: 'sun',
    CelestialObservation.MOON: 'moon',
}


class CelestialLocalizerNode(Node):
    def __init__(self):
        super().__init__('celestial_localizer_node')

        self.declare_parameter('input_topic', '/celestial_observations')
        self.declare_parameter('pose_topic', '/celestial_pose')
        self.declare_parameter('fix_topic', '/celestial_fix')
        self.declare_parameter('use_sun', True)
        self.declare_parameter('use_moon', True)
        self.declare_parameter('use_stars', True)
        self.declare_parameter('fixed_altitude', 0.0)
        self.declare_parameter('initial_latitude', 51.5)
        self.declare_parameter('initial_longitude', -0.1)
        self.declare_parameter('robust_loss', 'soft_l1')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        input_topic = self.get_parameter('input_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        fix_topic = self.get_parameter('fix_topic').value
        self.use_sun = self.get_parameter('use_sun').value
        self.use_moon = self.get_parameter('use_moon').value
        self.use_stars = self.get_parameter('use_stars').value
        self.fixed_altitude = self.get_parameter('fixed_altitude').value
        self.robust_loss = self.get_parameter('robust_loss').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.state = [
            self.get_parameter('initial_latitude').value,
            self.get_parameter('initial_longitude').value,
            0.0,
        ]

        self.ephemeris = EphemerisProvider()

        self.sub = self.create_subscription(CelestialObservationArray, input_topic, self._on_observations, 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, pose_topic, 10)
        self.fix_pub = self.create_publisher(NavSatFix, fix_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(f"celestial_localizer listening on {input_topic}")

    def _on_observations(self, msg):
        usable = []
        for obs in msg.observations:
            if obs.object_type == CelestialObservation.SUN and not self.use_sun:
                continue
            if obs.object_type == CelestialObservation.MOON and not self.use_moon:
                continue
            if obs.object_type == CelestialObservation.STAR and not self.use_stars:
                continue

            object_id = _OBJECT_ID_BY_TYPE.get(obs.object_type)
            if object_id is None:
                # Unidentified stars / rejection classes are not yet ephemeris-predictable.
                continue

            usable.append({
                'object_id': object_id,
                'azimuth': obs.azimuth,
                'elevation': obs.elevation,
                'confidence': max(obs.confidence, 1e-3),
            })

        timestamp = RclpyTime.from_msg(msg.header.stamp).nanoseconds / 1e9
        if timestamp <= 0:
            timestamp = self.get_clock().now().nanoseconds / 1e9

        result = solve(usable, timestamp, self.ephemeris, self.state, self.robust_loss)
        self.state = list(result.x)
        covariance = estimate_covariance(result, len(usable))

        self._publish_pose(msg.header, covariance)
        self._publish_fix(msg.header, covariance)
        if self.publish_tf:
            self._publish_tf(msg.header)

    def _publish_pose(self, header, covariance):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = self.map_frame
        pose_msg.pose.pose.position.x = self.state[1]  # longitude as x (local-tangent approximation)
        pose_msg.pose.pose.position.y = self.state[0]  # latitude as y
        pose_msg.pose.pose.position.z = self.fixed_altitude
        pose_msg.pose.pose.orientation.z = 0.0
        pose_msg.pose.pose.orientation.w = 1.0
        cov = [0.0] * 36
        cov[0] = covariance[0][0]
        cov[7] = covariance[1][1]
        cov[35] = covariance[2][2]
        pose_msg.pose.covariance = cov
        self.pose_pub.publish(pose_msg)

    def _publish_fix(self, header, covariance):
        fix_msg = NavSatFix()
        fix_msg.header = header
        fix_msg.latitude = self.state[0]
        fix_msg.longitude = self.state[1]
        fix_msg.altitude = self.fixed_altitude
        fix_msg.position_covariance = [
            covariance[0][0], 0.0, 0.0,
            0.0, covariance[1][1], 0.0,
            0.0, 0.0, 1.0,
        ]
        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
        self.fix_pub.publish(fix_msg)

    def _publish_tf(self, header):
        transform = TransformStamped()
        transform.header = header
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = self.state[1]
        transform.transform.translation.y = self.state[0]
        transform.transform.translation.z = self.fixed_altitude
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = CelestialLocalizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
