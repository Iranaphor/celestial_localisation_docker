# Debug output

This tracked file ensures the directory exists before Docker Compose starts.
Otherwise, Docker may create the bind-mount source as `root`, preventing the
non-root `ros` user from writing diagnostic images and JSON files.


# Command to run:

```sh
docker exec celestial_localisation_docker-celestial_localisation_service-1 bash -lc 'source /opt/ros/$ROS_DISTRO/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 service call /test/load_gps celestial_interfaces/srv/LoadGps "{latitude: 51.5, longitude: -0.1, altitude: 30.0, timestamp: {sec: $(date -u +%s), nanosec: 0}}"'
```

To run repeated random-position accuracy measurements, call:

```sh
docker exec celestial_localisation_docker-celestial_localisation_service-1 bash -lc 'source /opt/ros/$ROS_DISTRO/setup.bash && source ~/ros2_ws/install/setup.bash && ros2 service call /test/run_random_evaluation celestial_interfaces/srv/RunRandomEvaluation "{samples: 5, reps: 10, var_time: 1800.0, var_xy: 200.0, var_yaw: 20.0}"'
```

Every repetition and one summary measurement per sample are appended to
`random_localisation_metrics.csv` in this directory. Rows share a `sample_id`
and are tagged with `record_type=step` or `record_type=summary`; step rows also
carry their `repetition_index` and `is_outlier` status.

New evaluation rows include `identified_star_ids`, a JSON list of catalogue
star IDs used by the averaged pose. Existing rows from before that field was
added remain valid but have no star-list data; the evaluator migrates the CSV
header when the next sample is written.

Generate or refresh the seeded Gaussian-mixture model from those measurements
with:

```sh
python3 scripts/compare_localisation_clustering.py
```

This writes `gmm_boundaries.json` and adds `cluster_id` to existing CSV rows.
The detector writes the current `sky_map.png` and a live named cluster image
such as `cluster1_lowerror.png`; completed random evaluations replace that group
image with the definitive error-aware GMM assignment and add the error plus
ground-truth and estimated latitude/longitude to the lower-left corner.
`sky_map_classification.json` records the live observation-side classification
and timestamp.