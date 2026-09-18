# Celestial localisation

This ROS 2 Humble docker service hosts the full GNSS-independent celestial
localisation pipeline described in the repository [README](../README.md).

A single container builds and runs a colcon workspace containing six
packages under `src/`:

- `celestial_interfaces` — custom `CelestialObservation`/`CelestialObservationArray` messages.
- `sky_mapper` — builds a calibrated equirectangular sky map from the camera feed.
- `celestial_detector` — detects stars (photutils), sun and moon (OpenCV) in the sky map.
- `celestial_localizer` — matches observations against Astropy ephemeris predictions
  and solves for latitude/longitude/heading with SciPy, publishing pose/fix/TF.
- `celestial_simulation` — renders Stellarium Web Engine views into an
  equirectangular sky map after a GPS/time request.
- `celestial_bringup` — launch file that starts the pipeline nodes together.

Star identification (catalogue matching), aircraft/satellite rejection, and
IMU fusion are stubbed as documented in the relevant source files — they are
the natural next steps described in the design document.

## Run with RealSense

From the repository root:

```bash
docker compose up --build realsense_service celestial_localisation_service
```

Both services use host networking and must share the same `ROS_DOMAIN_ID`,
so the localizer can subscribe directly to the RealSense image topic.

## Configuration

Copy `example.env` to the repository root as `.env`, or export any of these
variables before running Compose:

- `CELESTIAL_CAMERA_TOPIC`
- `CELESTIAL_SKY_MAP_TOPIC`
- `CELESTIAL_OBSERVATIONS_TOPIC`
- `CELESTIAL_POSE_TOPIC` / `CELESTIAL_FIX_TOPIC`
- `CELESTIAL_PANORAMA_WIDTH` / `CELESTIAL_PANORAMA_HEIGHT`
- `CELESTIAL_CALIBRATION_FILE`
- `CELESTIAL_DETECT_STARS` / `CELESTIAL_DETECT_SUN` / `CELESTIAL_DETECT_MOON`
- `CELESTIAL_STAR_THRESHOLD` / `CELESTIAL_MIN_CONFIDENCE`
- `CELESTIAL_USE_SUN` / `CELESTIAL_USE_MOON` / `CELESTIAL_USE_STARS`
- `CELESTIAL_FIXED_ALTITUDE`
- `CELESTIAL_INITIAL_LATITUDE` / `CELESTIAL_INITIAL_LONGITUDE`
- `CELESTIAL_PUBLISH_TF` / `CELESTIAL_MAP_FRAME` / `CELESTIAL_BASE_FRAME`

## Editing packages

The `./src` directory is bind-mounted read/write into the container, so
package sources can be edited on the host and rebuilt inside the container
with `colcon build --symlink-install`.

The default bringup includes the in-container `test_publisher`, which exposes
`/test/publish_camera_image`, `/test/publish_sky_map`,
`/test/publish_observations`, and `/test/publish_all` services.

## Run the Stellarium simulator

Provide a local Stellarium Web Engine asset bundle in
`celestial_localisation/stellarium_assets/`. The directory is
mounted at `/opt/stellarium` by Compose. It should contain:

```text
stellarium_assets/
|-- stellarium-web-engine.js
|-- stellarium-web-engine.wasm
`-- data/
    |-- stars/
    |-- skycultures/western/
    |-- dso/
    `-- surveys/
        |-- milkyway/
        `-- sso/{sun,moon}/
```

The engine and sky-data files are not bundled in this repository. Stellarium
Web Engine is AGPL-3.0 or commercially licensed; obtain and distribute those
assets under the license that applies to your use.

The simulator node is launched alongside the live camera mapper and test
publisher, but it does not initialize the browser until `/test/load_gps` is
called. It exposes:

```bash
ros2 service call /test/load_gps celestial_interfaces/srv/LoadGps \
  "{latitude: 51.5, longitude: -0.1, altitude: 30.0, timestamp: {sec: $(date -u +%s), nanosec: 0}}"
```

The timestamp is UTC and becomes the image header timestamp as well as the
Stellarium observer time. A successful request renders and publishes one map;
`/celestial_fix` remains a localizer output and is not used as simulator input.
