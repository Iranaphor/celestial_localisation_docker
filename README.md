# Celestial Localisation

Docker Compose workspace for running the ROS 2 Humble celestial localisation
pipeline with an Intel RealSense camera and optional RViz visualisation.

## What is included

The workspace is split into three Compose services:

- `realsense_service` runs the RealSense ROS driver, camera topics, static
  camera transform, and the depth point cloud node.
- `celestial_localisation_service` builds and runs the ROS 2 workspace in
  `celestial_localisation/src`. It launches `sky_mapper`,
  `celestial_simulation`, `celestial_detector`, and `celestial_localizer`,
  together with the test publisher.
- `rviz_service` opens RViz with the configuration in `rviz/all.rviz`.

The two runtime services use host networking and the same `ROS_DOMAIN_ID` so
ROS 2 DDS discovery works between the containers and the host.

## Prerequisites

- Docker Engine with the Compose plugin
- An Intel RealSense camera connected over USB
- A host with an X server when using RViz
- Permission to access the camera and `/tmp/.X11-unix`

The RealSense container is privileged and receives `/dev/bus/usb`. The
celestial container installs its Python and ROS dependencies while building,
including NumPy, SciPy, OpenCV, Astropy, Skyfield, Photutils, and
`py360convert`.

## Quick start

Create a local environment file from the non-sensitive template and set the
camera serial number if a specific device should be selected:

```bash
cp example.env .env
# Edit .env and set REALSENSE_SERIAL_NUMBER when required.
```

Start the camera and celestial pipeline from this directory:

```bash
docker compose up --build realsense_service celestial_localisation_service
```

To include RViz, allow the local container to connect to the display and
start the additional service:

```bash
xhost +local:
docker compose up --build realsense_service celestial_localisation_service rviz_service
```

Stop the services with `Ctrl-C`. The first build can take some time because
both images install dependencies and build their ROS packages with `colcon`.

For the details of USB access, X11, networking, and service-specific
overrides, see [Docker operations](docs/docker.md).

## Configuration

Compose reads `.env` from the repository root. `example.env` lists the
available settings without including a device identifier or other local
values. The most commonly changed settings are:

- `REALSENSE_SERIAL_NUMBER` selects a specific camera.
- `REALSENSE_TF_*` configures the camera transform relative to the robot.
- `CELESTIAL_CAMERA_TOPIC` selects the image topic consumed by the pipeline.
- `CELESTIAL_PANORAMA_WIDTH` and `CELESTIAL_PANORAMA_HEIGHT` set the sky map size.
- `CELESTIAL_CALIBRATION_FILE` optionally supplies camera calibration.
- `CELESTIAL_INITIAL_LATITUDE` and `CELESTIAL_INITIAL_LONGITUDE` set the
  localiser starting estimate.
- `CELESTIAL_STELLARIUM_ASSET_DIR` supplies the mounted engine/data bundle.
- `CELESTIAL_STELLARIUM_ENGINE_JS` / `CELESTIAL_STELLARIUM_ENGINE_WASM` /
  `CELESTIAL_STELLARIUM_DATA_ROOT` select assets inside the container.
- `CELESTIAL_BROWSER_EXECUTABLE` optionally selects a custom Chromium executable;
  leave it blank to use the Playwright-managed browser in the image.
- `CELESTIAL_RANDOM_EVALUATION_SERVICE` selects the random evaluation service.
- `CELESTIAL_RANDOM_METRICS_FILENAME` selects the CSV filename under the debug
  output directory.
- `ROS_DOMAIN_ID` must match for all ROS 2 services.

The default data flow is:

```text
/camera/camera/color/image_raw
    -> /sky_map
    -> /celestial_observations
    -> /celestial_pose and /celestial_fix
```

The bringup launch file also publishes the configured TF between `map` and
`base_link` when `CELESTIAL_PUBLISH_TF=true`. Topic names and detector,
localiser, and panorama parameters are passed from environment variables.

The bringup launch file also starts the `test_publisher` node and exposes
`/test/publish_camera_image`,
`/test/publish_sky_map`, `/test/publish_observations`, and `/test/publish_all`
services. Those services publish sample messages onto the configured camera,
sky-map, and celestial-observation topics.

The simulator service is always available. Call `/test/load_gps` to render a
map from an observer position and timestamp:

```bash
ros2 service call /test/load_gps celestial_interfaces/srv/LoadGps \
  "{latitude: 51.5, longitude: -0.1, altitude: 30.0, timestamp: {sec: $(date -u +%s), nanosec: 0}}"
```

The request timestamp is UTC. The simulator publishes one `bgr8`
equirectangular image on `/sky_map` with that timestamp. The engine JavaScript,
WebAssembly, and external sky-data assets are intentionally mounted rather
than copied into the repository; Stellarium Web Engine is AGPL-3.0 or
commercially licensed.

The `random_localisation_evaluator` node is also started with the pipeline. It
waits idle until `/test/run_random_evaluation` is called, then performs the
requested number of random surface-position simulations and appends the
ground-truth coordinates, estimated coordinates, identified-object counts,
and error metrics to
`debug_output/random_localisation_metrics.csv`:

```bash
ros2 service call /test/run_random_evaluation \
  celestial_interfaces/srv/RunRandomEvaluation "{repetitions: 100}"
```

## Source layout

```text
.
|-- compose.yml
|-- example.env
|-- celestial_localisation/
|   |-- Dockerfile
|   `-- src/
|       |-- celestial_bringup/
|       |-- celestial_detector/
|       |-- celestial_interfaces/
|       |-- celestial_localiser/
|       |-- celestial_simulation/
|       `-- sky_mapper/
|-- realsense_docker/
|   |-- Dockerfile
|   `-- realsense_bringup/
`-- rviz/
  |-- Dockerfile
  `-- all.rviz
```

The source tree is bind-mounted into the celestial container. After editing
Python or ROS package files, restart or rebuild the service so its startup
script runs `colcon build --symlink-install` again.

## Development notes

See [Future directions](docs/future-directions.md) for planned developments.
