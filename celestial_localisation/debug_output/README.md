# Debug output

This tracked file ensures the directory exists before Docker Compose starts.
Otherwise, Docker may create the bind-mount source as `root`, preventing the
non-root `ros` user from writing diagnostic images and JSON files.


# Command to run:

```sh
docker exec celestial_localisation_docker-celestial_localisation_service-1 bash -lc 'source /opt/ros/$ROS_DISTRO/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 service call /test/load_gps celestial_interfaces/srv/LoadGps "{latitude: 51.5, longitude: -0.1, altitude: 30.0, timestamp: {sec: $(date -u +%s), nanosec: 0}}"'
```