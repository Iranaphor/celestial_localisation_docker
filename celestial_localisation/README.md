# Celestial localisation

This ROS 2 Humble docker service hosts the full GNSS-independent celestial
localisation pipeline described in the repository [README](../README.md).

A single container builds and runs a colcon workspace containing six
packages under `src/`:

- `celestial_interfaces` — custom `CelestialObservation`/`CelestialObservationArray` messages.
- `sky_mapper` — builds a calibrated equirectangular sky map from the camera feed.
- `celestial_detector` — detects stars (photutils), sun and moon (OpenCV), then
  optionally identifies stars with the bundled offline tetra3 database.
- `celestial_localizer` — matches observations against Astropy ephemeris predictions
  and solves for latitude/longitude/heading with SciPy, publishing pose/fix/TF.
- `celestial_simulation` — renders Stellarium Web Engine views into an
  equirectangular sky map after a GPS/time request.
- `celestial_bringup` — launch file that starts the pipeline nodes together.

Catalogue matching is now implemented for equirectangular maps through virtual
perspective tiles. Identified catalogue stars can enter the localizer's
Astropy ephemeris path. Aircraft/satellite rejection and IMU fusion remain
future work.

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
- `CELESTIAL_STAR_DATABASE_PATH` (empty uses tetra3's bundled offline database)
- `CELESTIAL_STAR_FOV_DEGREES` / `CELESTIAL_STAR_FOV_MAX_ERROR_DEGREES`
- `CELESTIAL_STAR_TILE_SIZE` / `CELESTIAL_STAR_MATCH_RADIUS`
- `CELESTIAL_STAR_MATCH_THRESHOLD` / `CELESTIAL_STAR_MIN_MATCHES`
- `CELESTIAL_USE_SUN` / `CELESTIAL_USE_MOON` / `CELESTIAL_USE_STARS`
- `CELESTIAL_FIXED_ALTITUDE`
- `CELESTIAL_INITIAL_LATITUDE` / `CELESTIAL_INITIAL_LONGITUDE`
- `CELESTIAL_PUBLISH_TF` / `CELESTIAL_MAP_FRAME` / `CELESTIAL_BASE_FRAME`
- `CELESTIAL_SIMULATION_SHOW_LABELS` hides or shows Stellarium object names.
- `CELESTIAL_SIMULATION_ENABLE_LANDSCAPE` enables the ground landscape.
- `CELESTIAL_SIMULATION_LANDSCAPE_KEY` selects the landscape data source.

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
    |-- landscapes/guereins/
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
  "{latitude: 51.5, longitude: -0.1, altitude: 30.0, yaw: 0.0, timestamp: {sec: $(date -u +%s), nanosec: 0}}"
```

The timestamp is UTC and becomes the image header timestamp as well as the
Stellarium observer time. A successful request renders and publishes one map;
`/celestial_fix` remains a localizer output and is not used as simulator input.
Object labels are disabled and the `guereins` ground landscape is enabled by
default. The landscape metadata points to the Stellarium HIPS service, so
rendering it requires network access from the container.

## Random localisation evaluation

The `random_localisation_evaluator` node is launched with the pipeline but
does no work until its service is called. It samples positions uniformly over
the Earth's surface, perturbs each sample for the requested repetitions, calls
the simulator for each repetition, averages the inlier poses, and appends one
tagged result for every repetition plus one tagged summary per sample to
`debug_output/random_localisation_metrics.csv`:

```bash
ros2 service call /test/run_random_evaluation \
  celestial_interfaces/srv/RunRandomEvaluation \
  "{samples: 5, reps: 10, var_time: 1800.0, var_xy: 200.0, var_yaw: 20.0}"
```

`var_time` is the maximum timestamp variation in seconds, `var_xy` is the
maximum radial position variation in metres, and `var_yaw` is the maximum
absolute yaw variation in degrees. Repetitions whose estimated position is a
spatial outlier are excluded before the sample average is calculated.

The CSV includes ground-truth and estimated latitude/longitude, the number of
identified objects used by the solver, the JSON-encoded list of identified
catalogue star IDs for each sample, signed latitude/longitude errors in degrees
and metres, haversine error distance in metres, and the cumulative mean error.
A second evaluation appends more rows to the same CSV. Each group shares a
`sample_id`; `record_type` is `step` for an individual repetition and `summary`
for the inlier average, while `repetition_index` and `is_outlier` tag the
individual steps. The Summary view has a small `Summaries` / `Steps` toggle for
switching between those records. The service response is returned only after
all requested samples and repetitions finish or one request fails. The summary
page uses the star list to compare each star's mean error when present with its
mean error when absent; positive benefit scores indicate lower error when that
star is present.
