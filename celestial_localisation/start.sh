#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
cd /home/ros/ros2_ws
colcon build --symlink-install
source install/setup.bash

exec ros2 launch celestial_bringup celestial_localisation.launch.py
