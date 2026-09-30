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
- `CELESTIAL_FACE_TOPIC` selects the timestamped simulator face-bundle topic.
- `CELESTIAL_DIRECTION_SOURCE` selects `original_face` (the simulated default)
  or `panorama` (the comparison baseline) for
  `original_face` for face-centroid directions. Original-face mode keeps the
  panorama's Tetra3 catalogue IDs, then matches face detections to observed
  panorama directions with mutual-nearest and ambiguity rejection; it does
  not use ephemeris or ground truth for correspondence.
- `CELESTIAL_FACE_MATCH_RADIUS_DEGREES`,
  `CELESTIAL_FACE_MATCH_AMBIGUITY_MARGIN_DEGREES`,
  `CELESTIAL_FACE_DUPLICATE_RADIUS_DEGREES` tune that observation-only match.
- The bundled simulated Compose pipeline defaults to `original_face` with
  `CELESTIAL_SOLVER_MAX_NFEV=60`, `CELESTIAL_SOLVER_MAX_STARTS=5`, and global
  search enabled. Set `CELESTIAL_DIRECTION_SOURCE=panorama` to reproduce the
  panorama comparison baseline; camera-only operation should use that override
  unless timestamped simulator face bundles are available.
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
- `CELESTIAL_STAR_BENEFIT_METRICS_FILENAME` selects the historical metrics CSV or filename pattern used to filter stars; the default is `simulated_location_filter_samples*.csv`.
- `CELESTIAL_STAR_BENEFIT_MIN_SCORE` / `CELESTIAL_STAR_BENEFIT_MIN_PRESENT_COUNT` / `CELESTIAL_STAR_BENEFIT_MIN_ABSENT_COUNT` tune harmful-star filtering.
- `CELESTIAL_STAR_BENEFIT_USE_HIGH_ERROR_GROUP` enables analysis of GMM group 3.
- `CELESTIAL_STAR_BENEFIT_USE_MEDIUM_ERROR_GROUP` enables analysis of GMM group 2.
- `CELESTIAL_USE_SUN` / `CELESTIAL_USE_MOON` / `CELESTIAL_USE_STARS`
- `CELESTIAL_FIXED_ALTITUDE`
- `CELESTIAL_INITIAL_LATITUDE` / `CELESTIAL_INITIAL_LONGITUDE`
- `CELESTIAL_PUBLISH_TF` / `CELESTIAL_MAP_FRAME` / `CELESTIAL_BASE_FRAME`
- `CELESTIAL_SIMULATION_SHOW_LABELS` hides or shows Stellarium object names.
- `CELESTIAL_SIMULATION_ENABLE_LANDSCAPE` enables the ground landscape.
- `CELESTIAL_SIMULATION_LANDSCAPE_KEY` selects the landscape data source.

At detector startup, the star-benefit filter expands the configured metrics
pattern and combines all matching CSV files. It uses `step` rows to compare
mean error with each catalogue star present versus absent within the enabled
GMM groups. By default, only the high-error group (group 3) is enabled; the
medium-error group (group 2) can be enabled independently. With the default
threshold, stars scoring at or below `-0.1` are published as `UNKNOWN` and
therefore do not contribute to localisation. Missing or invalid metrics leave
identification unchanged. Restart the pipeline after replacing the snapshot so
new data is not mixed into an active filter.

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

When `CELESTIAL_DIRECTION_SOURCE=original_face`, the simulator publishes the
six original face images on `CELESTIAL_FACE_TOPIC` with the same ROS timestamp
as the panorama. The detector uses those images only to replace accepted star
directions; the panorama remains the source of star identities and the default
`panorama` mode does not publish the extra face bundle.

Face centroids are projected from pixel centers rather than image edges. This
matches the renderer/compositor sampling convention and is part of the
simulated original-face measurement path.

The controlled fixed-scene comparison can be run from the repository root:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_simulation:/home/ros/ros2_ws/src/celestial_detector:/home/ros/ros2_ws/src/celestial_localiser; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/compare_face_panorama_campaign.py \
     --scenes /tmp/celestial_scripts/fixed_face_panorama_scenes.json \
     --output /home/ros/debug_output/face_panorama_campaign.json \
     --csv-output /home/ros/debug_output/face_panorama_campaign.csv'
```

The manifest contains 24 fixed scenes spanning locations, UTC times and
headings. Each pair shares the panorama-identified star IDs, initial state,
solver limits and global-search settings. The JSON and CSV report geographic
error, angular observation error, solver residuals, failures, runtimes, and
face stars lost or rejected as ambiguous during observation-only matching.

To separate source-face measurement resolution from re-identification, use the
canonical-ID transfer diagnostic:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  -v "$PWD/celestial_localisation/debug_output:/home/ros/debug_output:rw" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_simulation:/home/ros/ros2_ws/src/celestial_detector:/home/ros/ros2_ws/src/celestial_localiser; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/compare_face_resolution_transfer.py \
     --scenes /tmp/celestial_scripts/fixed_face_panorama_scenes.json \
     --scene-id london_00_h000 new_york_00_h045 quito_00_h030 \
     --face-sizes 512 1024 2048 \
     --output /home/ros/debug_output/face_resolution_transfer.json'
```

This identifies stars once from the 512-face panorama, transfers those IDs to
each face resolution using observed-direction matching and ambiguity rejection,
then solves only the strict common-ID intersection. Higher-resolution runs that
lose catalogue IDs are reported as identification-limited rather than treated
as measurement failures.

The saved New York observation diagnostic is available at
`scripts/analyze_new_york_saved_observations.py`. It reports signed per-star
errors, leave-one-out solves, every optimizer start and cost, and the objective
at the known scene state for post-hoc diagnosis. The current saved transfer
artifact contains exact solver observations only for the 512-face case; its
1024/2048 entries contain IDs and error summaries, not directional observation
vectors, so higher-resolution signed-error analysis requires another capture.
The subsequent New York resolution capture stores those per-size directions and
crops under `debug_output/new_york_resolution_capture_20260929`. `HIP_26241`
is a merged bright component at all three sizes, with its footprint growing
from 20 thresholded pixels at 512 to 170 at 2048 and a face assignment change
at the boundary. It is retained; this single scene is insufficient evidence
for a catalogue-star blacklist.

The saved per-resolution New York diagnostic is generated by
`scripts/analyze_new_york_resolution_saved.py`. It keeps the same 17-star
intersection and reports signed errors, leave-one-out influence, all-start
solutions, objective-at-truth, and nearby detections on overlapping faces.
It shows `HIP_26241` has a consistent positive elevation bias across 512/1024/
2048 and is influential, but removing it does not eliminate the 2048 geographic
degradation. The higher-resolution result therefore remains a combined
measurement/model issue rather than an identity-specific failure.

The saved multi-resolution observation diagnostic is generated by
`scripts/analyze_new_york_resolution_saved.py`:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  -v "$PWD/celestial_localisation/debug_output:/home/ros/debug_output:rw" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_localiser:/tmp/celestial_scripts; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/analyze_new_york_resolution_saved.py \
     --transfer /home/ros/debug_output/new_york_resolution_transfer_20260929.json \
     --output /home/ros/debug_output/new_york_resolution_subsets_20260929.json'
```

The report compares signed azimuth, cross-track and elevation errors at 512,
1024 and 2048 pixels, preserves signed errors for every overlapping-face
candidate, and labels observations as `clean_isolated`, `overlapping_faces`, or
`suspected_blended` using candidate neighborhoods. These labels are diagnostic
only; they do not reject or reweight observations in production. The current
New York report shows clean-isolated angular error improving while the selected
geographic error worsens, with the persistent HIP_26241 elevation bias as a
prominent blended-measurement case.
The geographic errors in the resolution rows are solver results from all 17
stars common to the three resolutions; they are not computed from the clean
subset. The subset summaries are post-hoc annotations only.
The same report records the production matcher trace. For HIP_26241, the
panorama direction itself is about `0.470` degrees from the ephemeris truth.
The matcher therefore selects the candidate `0.070` to `0.084` degrees from
that biased panorama direction, while a nearby candidate with roughly `0.45`
to `0.47` degrees panorama distance has near-zero truth error. The overlapping
near pair is collapsed by the `0.15` degree duplicate radius, leaving a
`0.37` to `0.40` degree panorama margin above the `0.25` degree ambiguity
threshold. The result passes mutual-nearest checks; this is a specific
panorama-bias matching failure, not evidence that the operational matcher may
use truth to choose the alternate.

Independent face-only identification can be tested on the same saved images
with `scripts/diagnose_independent_face_identification.py`. It runs the
offline Tetra3 matcher in overlapping virtual perspective crops and compares
its correspondence with the panorama-transferred identity without using truth
for identification or selection:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  -v "$PWD/celestial_localisation/debug_output:/home/ros/debug_output:rw" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_localiser:/tmp/celestial_scripts; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/diagnose_independent_face_identification.py \
     --transfer /home/ros/debug_output/new_york_resolution_transfer_20260929.json \
     --crop-fov 45 \
     --output /home/ros/debug_output/independent_face_identification_45deg_20260929.json'
```

On the saved New York scene, the 1024-pixel/45-degree crop run independently
identified HIP_26241 on the alternate zenith detection with 16 matches and
false probability `3.1e-17`; its post-hoc angular error was `0.0086` degrees.
Replacing only that correspondence in the all-17-star solve changed the
geographic error from `3.53 km` to `1.56 km`. The 512-pixel/30-degree run
instead identified the biased near-duplicate, and no independent HIP_26241
assignment was obtained at 2048 in these settings. This is evidence from one
scene, not a production matcher or filter proposal.

The fixed-setting multi-scene campaign is generated by
`scripts/evaluate_independent_face_campaign.py`:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  -v "$PWD/celestial_localisation/debug_output:/home/ros/debug_output:rw" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_localiser:/tmp/celestial_scripts; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/evaluate_independent_face_campaign.py \
     --captures /home/ros/debug_output/quality_heldout_captures_20260929 \
     --crop-fov 45 \
     --output /home/ros/debug_output/independent_face_campaign_45deg_20260929.json'
```

The campaign uses each capture's saved face observations for both the
panorama-transferred baseline and the independent hybrid solve. Rerun matching
is used only to report source agreement; truth is post-hoc annotation and never
selects an identity. Across 24 development scenes, the baseline was valid on
21 and the independent hybrid on 22. Their median geographic errors were
`1.509 km` and `1.420 km`, respectively. Among 21 paired-valid scenes, the
hybrid improved 13 and worsened 8, with median delta `-10 m` and mean delta
`-201 m`. Independent selection changed the saved baseline face source for
`19/402` selected observations; exact rerun-candidate comparison changed
`21/402`. The saved captures do not retain every baseline pixel coordinate, so
the former is the primary coarse correspondence statistic. This is mixed
evidence, not a consistently improving policy, so production remains unchanged.

The consolidated per-scene comparison is stored in
`debug_output/independent_face_campaign_45deg_summary_20260929.json` with a
flat CSV companion. Baseline versus hybrid valid-fix rates are:

- `<=1 km`: `5/21` (`23.8%`) versus `7/22` (`31.8%`)
- `<=5 km`: `17/21` (`81.0%`) versus `19/22` (`86.4%`)
- `<=10 km`: `21/21` (`100%`) versus `22/22` (`100%`)

Worst valid errors were `8.560 km` for the baseline and `8.542 km` for the
hybrid. The largest paired improvement was Rio-00 (`5.589 -> 0.311 km`); the
largest regression was Seoul-06 (`5.532 -> 6.023 km`). Changed-correspondence
lists are diagnostic and do not provide causal attribution: some geographic
deltas occur without a detected source change, and a changed star may interact
with the full solver geometry. The 24 scenes are development evidence, not an
untouched evaluation set.

Ground-truth-free quality rules can be evaluated on the saved development
captures with `scripts/evaluate_face_quality_rules.py`. The diagnostic compares
the unweighted baseline with continuous observable weighting and conservative
rejection using face-boundary margin, observed face/panorama agreement,
overlapping-face disagreement, local crowding and crop saturation/shape. Truth
is used only for post-hoc error reporting; no identity-specific blacklist or
quality rule is enabled in production.
The fresh quality-heldout evaluation is stored in
`debug_output/quality_heldout_rules_20260929.json`. On 24 new scenes, baseline
median error was `1.53 km`, full weighting was `1.74 km`, and agreement-only
weighting was `1.72 km`; agreement-only also produced a multi-thousand-kilometre
outlier. Neither weighting strategy generalized as a reliable improvement, so
both remain diagnostic-only.
The weighting investigation is frozen at this point: production keeps the
unweighted baseline, and no fit-quality gate is enabled. The proposed
diagnostic residual profile is `balanced_residual` (`RMS <= 0.15` degrees,
`P90 <= 0.15` degrees, maximum `<= 0.75` degrees). On the 24 quality-heldout
development captures it rejected `0/21` valid unweighted baseline fixes; the
other three attempts were already invalid (`insufficient_observations` or
non-convergence). Those captures were used to choose the diagnostic profile,
so they are not independent validation data.

The exact agreement-only Melbourne failure is retained as the compact solver
fixture `src/celestial_localiser/test/data/melbourne_agreement_only_fixture.json`.
Its regression test verifies that the unweighted baseline remains a valid
low-residual fix, while agreement-only weighting produces five bad basins and
selects the lowest finite cost with a large residual. The saved report is
`debug_output/fit_quality_gate_agreement_only_20260929.json`. Any future
production gate must first be evaluated on an independently collected set and
must report valid-baseline rejection before being enabled.
New campaign JSON reports also include `replay_cases` with the exact solver
observations for each scene. Replay original-face failures without rendering:

```bash
python3 scripts/replay_face_failure_cases.py \
  --campaign celestial_localisation/debug_output/face_panorama_campaign.json \
  --max-nfev 30 60 120 240 \
  --max-starts 5 13 25 \
  --output celestial_localisation/debug_output/face_failure_replay.json
```

Each configuration records every attempted start's initial state, final state,
residual vector, cost, convergence status and geographic error. Campaigns
created before `replay_cases` was added cannot be replayed exactly without a
new render; the replay command fails explicitly for those artifacts.

To create fresh captures of the seven historical face-failure scene
definitions, use the dedicated capture command. These are new captures of the
same scene definitions, not exact replays of the earlier render:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  -v "$PWD/celestial_localisation/debug_output:/home/ros/debug_output:rw" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_simulation:/home/ros/ros2_ws/src/celestial_detector:/home/ros/ros2_ws/src/celestial_localiser; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/capture_face_failure_scenes.py \
     --output-dir /home/ros/debug_output/face_failure_captures'
```

Each capture stores all six face images, the composed panorama, solver-ready
observations, resolved settings, and a reference to the historical result.
Replay all captured scenes across larger iteration and initialization budgets:

```bash
docker compose run --rm --no-deps --entrypoint bash \
  -v "$PWD/scripts:/tmp/celestial_scripts:ro" \
  -v "$PWD/celestial_localisation/debug_output:/home/ros/debug_output:rw" \
  celestial_localisation_service -lc \
  'export PYTHONDONTWRITEBYTECODE=1 \
   PYTHONPATH=/home/ros/ros2_ws/src/celestial_localiser:/tmp/celestial_scripts; \
   source /opt/ros/humble/setup.bash && source /home/ros/ros2_ws/install/setup.bash && \
   python3 /tmp/celestial_scripts/replay_face_failure_cases.py \
     --captures /home/ros/debug_output/face_failure_captures \
     --max-nfev 30 60 120 240 \
     --max-starts 5 13 25 \
     --output /home/ros/debug_output/face_failure_replay.json'
```

The replay report groups distinct final solutions, records every optimizer
start, and separately annotates the lowest-cost finite start, lowest-cost
converged start, and operationally selected start with geographic error for
diagnosis. Ground truth is not used by the operational solver selection.

The solver now ranks finite candidates by cost first and uses convergence as
an acceptance gate. A lower-cost unfinished candidate is therefore returned as
`optimizer_not_converged` rather than being replaced by a higher-cost
converged local minimum; the localizer does not publish that unresolved result.

The completed 24-scene 60-evaluation/five-start benchmark is stored in
`debug_output/face_panorama_campaign_20260929_nfev60_starts5.json` and its CSV
companion. It is a development benchmark result, not a production default
change.

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
  "{samples: 5, reps: 10, var_time: 1800.0, var_xy: 200.0, var_yaw: 20.0, random_seed: 20260928}"
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
Failed attempts are retained as `record_type=failure` rows, and the summary
aggregation does not use ground truth to select inliers. Use
`python3 scripts/summarize_localisation_metrics.py` for explicit denominators,
percentiles and threshold fractions.
