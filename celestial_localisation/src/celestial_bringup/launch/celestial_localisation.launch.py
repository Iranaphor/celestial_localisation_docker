"""Brings up the celestial pipeline and its optional simulation service.

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

    celestial_simulation_node = Node(
        package='celestial_simulation',
        executable='celestial_simulation_node',
        name='celestial_simulation_node',
        output='screen',
        parameters=[{
            'output_topic': sky_map_topic,
            'load_gps_service': _env('CELESTIAL_LOAD_GPS_SERVICE', '/test/load_gps'),
            'frame_id': _env('CELESTIAL_SKY_MAP_FRAME', 'sky_map'),
            'panorama_width': int(_env('CELESTIAL_PANORAMA_WIDTH', '2048')),
            'panorama_height': int(_env('CELESTIAL_PANORAMA_HEIGHT', '1024')),
            'face_size': int(_env('CELESTIAL_SIMULATION_FACE_SIZE', '512')),
            'face_field_of_view': float(_env('CELESTIAL_SIMULATION_FACE_FOV', '95.0')),
            'render_timeout_seconds': float(_env('CELESTIAL_SIMULATION_RENDER_TIMEOUT', '60.0')),
            'show_object_labels': _env('CELESTIAL_SIMULATION_SHOW_LABELS', 'false') == 'true',
            'enable_landscape': _env('CELESTIAL_SIMULATION_ENABLE_LANDSCAPE', 'true') == 'true',
            'landscape_key': _env('CELESTIAL_SIMULATION_LANDSCAPE_KEY', 'guereins'),
            'debug_output_dir': debug_output_dir,
            'engine_js': _env(
                'CELESTIAL_STELLARIUM_ENGINE_JS',
                '/opt/stellarium/stellarium-web-engine.js',
            ),
            'engine_wasm': _env(
                'CELESTIAL_STELLARIUM_ENGINE_WASM',
                '/opt/stellarium/stellarium-web-engine.wasm',
            ),
            'data_root': _env('CELESTIAL_STELLARIUM_DATA_ROOT', '/opt/stellarium/data'),
            'browser_executable': _env('CELESTIAL_BROWSER_EXECUTABLE', ''),
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
            'star_min_elevation_degrees': float(_env('CELESTIAL_STAR_MIN_ELEVATION_DEGREES', '25.0')),
            'star_max_candidates': int(_env('CELESTIAL_STAR_MAX_CANDIDATES', '64')),
            'star_database_path': _env('CELESTIAL_STAR_DATABASE_PATH', ''),
            'star_fov_degrees': float(_env('CELESTIAL_STAR_FOV_DEGREES', '30.0')),
            'star_fov_max_error_degrees': float(_env('CELESTIAL_STAR_FOV_MAX_ERROR_DEGREES', '3.0')),
            'star_tile_size': int(_env('CELESTIAL_STAR_TILE_SIZE', '512')),
            'star_match_radius': float(_env('CELESTIAL_STAR_MATCH_RADIUS', '0.02')),
            'star_match_threshold': float(_env('CELESTIAL_STAR_MATCH_THRESHOLD', '0.001')),
            'star_min_matches': int(_env('CELESTIAL_STAR_MIN_MATCHES', '4')),
            'minimum_confidence': float(_env('CELESTIAL_MIN_CONFIDENCE', '0.5')),
            'debug_output_dir': debug_output_dir,
            'gmm_boundaries_filename': _env(
                'CELESTIAL_GMM_BOUNDARIES_FILENAME',
                'gmm_boundaries.json',
            ),
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
            'star_database_path': _env('CELESTIAL_STAR_DATABASE_PATH', ''),
            'fixed_altitude': float(_env('CELESTIAL_FIXED_ALTITUDE', '0.0')),
            'initial_latitude': float(_env('CELESTIAL_INITIAL_LATITUDE', '51.5')),
            'initial_longitude': float(_env('CELESTIAL_INITIAL_LONGITUDE', '-0.1')),
            'solver_max_nfev': int(_env('CELESTIAL_SOLVER_MAX_NFEV', '30')),
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

    random_localisation_evaluator_node = Node(
        package='celestial_bringup',
        executable='random_localisation_evaluator',
        name='random_localisation_evaluator',
        output='screen',
        parameters=[{
            'load_gps_service': _env('CELESTIAL_LOAD_GPS_SERVICE', '/test/load_gps'),
            'evaluation_service': _env(
                'CELESTIAL_RANDOM_EVALUATION_SERVICE',
                '/test/run_random_evaluation',
            ),
            'debug_output_dir': debug_output_dir,
            'pose_filename': _env('CELESTIAL_POSE_DEBUG_FILENAME', 'pose.json'),
            'metrics_filename': _env(
                'CELESTIAL_RANDOM_METRICS_FILENAME',
                'random_localisation_metrics.csv',
            ),
            'gmm_boundaries_filename': _env(
                'CELESTIAL_GMM_BOUNDARIES_FILENAME',
                'gmm_boundaries.json',
            ),
            'fixed_altitude': float(_env('CELESTIAL_FIXED_ALTITUDE', '0.0')),
            'result_timeout_seconds': float(
                _env('CELESTIAL_RANDOM_EVALUATION_TIMEOUT', '180.0')
            ),
            'poll_interval_seconds': float(
                _env('CELESTIAL_RANDOM_EVALUATION_POLL_INTERVAL', '0.25')
            ),
        }],
    )

    return LaunchDescription([
        sky_mapper_node,
        celestial_simulation_node,
        celestial_detector_node,
        celestial_localizer_node,
        test_publisher_node,
        random_localisation_evaluator_node,
    ])
