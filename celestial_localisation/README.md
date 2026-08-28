# Celestial localisation

This ROS 2 Humble docker service hosts the full GNSS-independent celestial
localisation pipeline described in the repository [README](../README.md).

A single container builds and runs a colcon workspace containing five
packages under `src/`:

- `celestial_interfaces` — custom `CelestialObservation`/`CelestialObservationArray` messages.
- `sky_mapper` — builds a calibrated equirectangular sky map from the camera feed.
- `celestial_detector` — detects stars (photutils), sun and moon (OpenCV) in the sky map.
- `celestial_localizer` — matches observations against Astropy ephemeris predictions
  and solves for latitude/longitude/heading with SciPy, publishing pose/fix/TF.
- `celestial_bringup` — launch file that starts all three nodes together.

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
