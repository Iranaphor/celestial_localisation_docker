"""Brings up sky_mapper, celestial_detector and celestial_localizer together.

Topic names and key parameters are overridable via environment variables so
the same launch file can be reused across docker-compose deployments,
mirroring the motion_extractor pattern of env-driven configuration.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def _env(name, default):
    return os.environ.get(name, default)


def _package_file(*parts):
    return os.path.join(get_package_share_directory('celestial_bringup'), *parts)


def generate_launch_description():
    camera_input_topic = _env('CELESTIAL_CAMERA_TOPIC', '/camera/camera/color/image_raw')
    sky_map_topic = _env('CELESTIAL_SKY_MAP_TOPIC', '/sky_map')
    observations_topic = _env('CELESTIAL_OBSERVATIONS_TOPIC', '/celestial_observations')
    debug_output_dir = _env('CELESTIAL_DEBUG_OUTPUT_DIR', '')

    sky_mapper_node = Node(
        package='sky_mapper',
        executable='sky_mapper_node',
        name='sky_mapper_node',
        output='screen',
        parameters=[{
            'input_topic': camera_input_topic,
            'output_topic': sky_map_topic,
            'panorama_width': int(_env('CELESTIAL_PANORAMA_WIDTH', '2048')),
            'panorama_height': int(_env('CELESTIAL_PANORAMA_HEIGHT', '1024')),
            'calibration_file': _env('CELESTIAL_CALIBRATION_FILE', ''),
            'debug_output_dir': debug_output_dir,
        }],
    )

    celestial_detector_node = Node(
        package='celestial_detector',
        executable='celestial_detector_node',
        name='celestial_detector_node',
        output='screen',
        parameters=[{
            'input_topic': sky_map_topic,
            'output_topic': observations_topic,
            'detect_stars': _env('CELESTIAL_DETECT_STARS', 'true') == 'true',
            'detect_sun': _env('CELESTIAL_DETECT_SUN', 'true') == 'true',
            'detect_moon': _env('CELESTIAL_DETECT_MOON', 'true') == 'true',
            'star_detection_threshold': float(_env('CELESTIAL_STAR_THRESHOLD', '5.0')),
            'minimum_confidence': float(_env('CELESTIAL_MIN_CONFIDENCE', '0.5')),
            'debug_output_dir': debug_output_dir,
        }],
    )

    celestial_localizer_node = Node(
        package='celestial_localizer',
        executable='celestial_localizer_node',
        name='celestial_localizer_node',
        output='screen',
        parameters=[{
            'input_topic': observations_topic,
            'pose_topic': _env('CELESTIAL_POSE_TOPIC', '/celestial_pose'),
            'fix_topic': _env('CELESTIAL_FIX_TOPIC', '/celestial_fix'),
            'use_sun': _env('CELESTIAL_USE_SUN', 'true') == 'true',
            'use_moon': _env('CELESTIAL_USE_MOON', 'true') == 'true',
            'use_stars': _env('CELESTIAL_USE_STARS', 'true') == 'true',
            'fixed_altitude': float(_env('CELESTIAL_FIXED_ALTITUDE', '0.0')),
            'initial_latitude': float(_env('CELESTIAL_INITIAL_LATITUDE', '51.5')),
            'initial_longitude': float(_env('CELESTIAL_INITIAL_LONGITUDE', '-0.1')),
            'publish_tf': _env('CELESTIAL_PUBLISH_TF', 'true') == 'true',
            'map_frame': _env('CELESTIAL_MAP_FRAME', 'map'),
            'base_frame': _env('CELESTIAL_BASE_FRAME', 'base_link'),
            'debug_output_dir': debug_output_dir,
        }],
    )

    test_publisher_node = Node(
        package='celestial_bringup',
        executable='test_publisher',
        name='test_publisher',
        output='screen',
        parameters=[{
            'camera_topic': camera_input_topic,
            'sky_map_topic': sky_map_topic,
            'observations_topic': observations_topic,
            'sample_observations_file': _package_file('data', 'sample_observations.json'),
            'camera_image_file': _env('CELESTIAL_TEST_CAMERA_IMAGE_FILE', ''),
            'sky_map_image_file': _env('CELESTIAL_TEST_SKY_MAP_IMAGE_FILE', ''),
            'sky_map_image_dir': _env('CELESTIAL_TEST_SKY_MAP_IMAGE_DIR', ''),
            'capture_timestamp': _env('CELESTIAL_TEST_CAPTURE_TIMESTAMP', ''),
        }],
    )

    return LaunchDescription([
        sky_mapper_node,
        celestial_detector_node,
        celestial_localizer_node,
        test_publisher_node,
    ])
