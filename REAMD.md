# ROS 2 Celestial Localisation System

## 1. Project Overview

This project will develop a ROS 2 system for **GNSS-independent celestial localisation** using an upward-facing 360° camera.

The system will observe the sky and use known celestial objects, including:

- Stars
- Sun
- Moon

to estimate the approximate global position and orientation of a robot.

The initial implementation should operate while the robot is stationary and assume that accurate UTC date/time is available.

The system should be designed so that it can later be extended with:

- Cloud detection
- Cloud tracking
- Atmospheric visual odometry
- Satellite detection and identification
- Aircraft rejection
- Temporal sky observations
- IMU fusion
- Continuous localisation

The initial development should therefore focus on a modular **celestial localisation pipeline** rather than implementing the later atmospheric odometry system.

---

# 2. High-Level Architecture

The initial system should contain three primary ROS 2 nodes:

```text
┌─────────────────────────┐
│      360° Camera        │
└────────────┬────────────┘
             │
             │ Image frames
             ▼
┌─────────────────────────┐
│      sky_mapper         │
│                         │
│ Construct calibrated    │
│ sky panorama / skybox   │
└────────────┬────────────┘
             │
             │ /sky_map
             ▼
┌─────────────────────────┐
│   celestial_detector    │
│                         │
│ Detect and identify     │
│ celestial observations │
└────────────┬────────────┘
             │
             │ /celestial_observations
             ▼
┌─────────────────────────┐
│  celestial_localizer    │
│                         │
│ Ephemeris matching      │
│ + position optimisation │
└────────────┬────────────┘
             │
             ├── PoseWithCovarianceStamped
             ├── NavSatFix
             └── TF
```

The modules should be kept independent so that individual implementations can be replaced later.

---

# 3. Proposed ROS 2 Package Structure

Create a ROS 2 Python workspace/package structure similar to:

```text
celestial_localisation_ws/
│
└── src/
    │
    ├── sky_mapper/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── resource/
    │   ├── config/
    │   │   └── sky_mapper.yaml
    │   └── sky_mapper/
    │       ├── __init__.py
    │       ├── sky_mapper_node.py
    │       ├── panorama.py
    │       ├── projection.py
    │       └── calibration.py
    │
    ├── celestial_detector/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── config/
    │   │   └── celestial_detector.yaml
    │   └── celestial_detector/
    │       ├── __init__.py
    │       ├── celestial_detector_node.py
    │       ├── star_detector.py
    │       ├── star_identifier.py
    │       ├── sun_detector.py
    │       ├── moon_detector.py
    │       ├── transient_detector.py
    │       └── angular_projection.py
    │
    ├── celestial_localizer/
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── config/
    │   │   └── celestial_localizer.yaml
    │   └── celestial_localizer/
    │       ├── __init__.py
    │       ├── celestial_localizer_node.py
    │       ├── ephemeris.py
    │       ├── coordinate_transforms.py
    │       ├── pose_solver.py
    │       └── uncertainty.py
    │
    ├── celestial_interfaces/
    │   ├── package.xml
    │   ├── CMakeLists.txt
    │   └── msg/
    │       ├── CelestialObservation.msg
    │       └── CelestialObservationArray.msg
    │
    └── celestial_bringup/
        ├── package.xml
        ├── setup.py
        ├── launch/
        │   └── celestial_localisation.launch.py
        └── config/
            └── celestial_localisation.yaml
```

The exact package structure may be adjusted where appropriate, but image construction, celestial detection and localisation should remain logically separated.

---

# 4. Node 1 — `sky_mapper`

## Purpose

Construct a geometrically calibrated representation of the visible sky from the camera.

The preferred representation is an **equirectangular spherical panorama**.

The output should represent:

```text
horizontal axis → azimuth
vertical axis   → elevation
```

Conceptually:

```text
x = 0 ------------------------------ image_width
    0°                              360°
                AZIMUTH

y
│
│ +90° zenith
│
│
│   0° horizon
▼
```

Exact mapping should be defined through camera calibration rather than assuming perfect geometry.

---

## Triggering

The node should support a request/trigger telling it to begin constructing a sky snapshot.

Possible interface:

```text
/std_msgs/Bool
```

or preferably a ROS 2 service such as:

```text
/capture_sky_map
```

A service is preferred because sky-map generation represents a discrete operation rather than a continuous command.

---

## Inputs

Potential input:

```text
/camera/image_raw
```

Type:

```text
sensor_msgs/msg/Image
```

Camera calibration may additionally come from:

```text
/camera/camera_info
```

---

## Output

```text
/sky_map
```

Type:

```text
sensor_msgs/msg/Image
```

The output should preferably contain a complete calibrated equirectangular panorama.

---

# 5. Sky Mapping Libraries

## OpenCV

Primary image-processing dependency:

```text
opencv-python
```

Potential functionality:

- Camera calibration
- Image undistortion
- Feature detection
- Feature matching
- Homography estimation
- Image warping
- Panorama construction
- Image blending

Useful OpenCV components include:

```python
cv2.Stitcher
cv2.remap
cv2.warpPerspective
cv2.findHomography
```

For an initial prototype, `cv2.Stitcher` may be sufficient.

For later versions, lower-level control over projection and stitching will likely be preferable because angular accuracy is more important than visual panorama quality.

---

## py360convert

Use:

```text
py360convert
```

for conversions between:

- Equirectangular images
- Cubemaps
- Perspective projections

This is particularly useful because star-identification algorithms generally expect conventional perspective images rather than heavily distorted equirectangular images.

Example pipeline:

```text
360° sky panorama
       │
       ▼
py360convert
       │
       ├── North perspective
       ├── East perspective
       ├── South perspective
       ├── West perspective
       └── Zenith perspective
```

Detections can subsequently be transformed back into the global sky coordinate representation.

---

# 6. Camera Calibration

Angular accuracy is critical.

Do NOT assume that:

```text
pixel coordinate = perfect azimuth/elevation
```

unless the camera projection has been calibrated.

The system should eventually account for:

- Lens distortion
- Fisheye distortion
- Stitching distortion
- Camera orientation
- Camera-to-robot transform
- Panorama seam geometry
- Camera mounting error

The calibration module should expose functions similar to:

```python
pixel_to_unit_vector(x, y)

unit_vector_to_az_el(vector)

pixel_to_az_el(x, y)
```

---

# 7. Node 2 — `celestial_detector`

## Purpose

Process a completed sky snapshot and generate structured observations of celestial objects.

Initial categories:

```text
STAR
SUN
MOON
```

Additional classifications should be anticipated:

```text
SATELLITE
AIRCRAFT
UNKNOWN_POINT_SOURCE
CLOUD
```

Satellites and aircraft should initially primarily be treated as contaminants/rejection classes rather than localisation landmarks.

---

# 8. Star Detection

Do NOT initially use a general object detector such as YOLO for stars.

Stars are astronomical point sources and should initially use established astronomical source-detection techniques.

Recommended library:

```text
photutils
```

Primary detector:

```python
from photutils.detection import DAOStarFinder
```

Pipeline:

```text
sky image
    │
    ▼
grayscale / intensity image
    │
    ▼
background estimation
    │
    ▼
point-source detection
    │
    ▼
subpixel centroid estimation
    │
    ▼
star candidate list
```

Store:

```text
pixel_x
pixel_y
brightness
centroid uncertainty
detection confidence
```

Subpixel centroid estimation is desirable because celestial localisation accuracy is directly related to angular measurement accuracy.

---

# 9. Star Identification

Recommended library:

```text
tetra3
```

ESA's Tetra3 implementation provides lost-in-space star-pattern recognition.

Conceptually:

```text
Detected star centroids
          │
          ▼
Geometric star pattern
          │
          ▼
Tetra3 catalogue matching
          │
          ▼
Known star identities
          +
celestial orientation
```

Because Tetra3 is designed around conventional camera views, do not necessarily run it directly against the complete equirectangular panorama.

Preferred architecture:

```text
Equirectangular sky map
          │
          ▼
    py360convert
          │
    perspective views
          │
          ▼
        tetra3
          │
          ▼
 identified stars
          │
          ▼
convert detections back into
global sky azimuth/elevation
```

Alternative/future implementation:

```text
tetra3rs
```

may also be investigated.

---

# 10. Sun Detection

The initial Sun detector should use classical computer vision.

Possible techniques:

- Intensity thresholding
- Connected-component detection
- Contour extraction
- Circular fitting
- Hough circle detection
- Limb fitting

Useful OpenCV functionality:

```python
cv2.threshold
cv2.findContours
cv2.HoughCircles
cv2.moments
```

The objective is to estimate the **centre of the solar disc**, not simply the brightest pixel.

Output should contain:

```text
SUN
azimuth
elevation
angular uncertainty
confidence
```

Care should be taken around:

- Sensor saturation
- Bloom
- Clouds
- Lens flare
- Reflections

---

# 11. Moon Detection

Moon detection may similarly initially use classical computer vision.

Potential methods:

- Bright-region detection
- Circular/elliptical limb fitting
- Hough circle detection
- Contour analysis

Moon phase means that simple centroid detection may not correspond exactly to the centre of the lunar disc.

Later implementations should therefore investigate **lunar limb fitting**.

Output:

```text
MOON
azimuth
elevation
angular uncertainty
confidence
```

---

# 12. Learned Detection

Machine-learning detection should be considered an optional extension.

Potential framework:

```text
Ultralytics YOLO
```

Possible segmentation/detection classes:

```text
SUN
MOON
CLOUD
AIRCRAFT
```

Stars should remain handled separately using astronomical point-source detection.

---

# 13. Aircraft and Satellite Rejection

Aircraft and satellites may appear as point sources and could potentially be confused with stars.

Temporal observations can help distinguish these objects.

Conceptually:

```text
point source
    │
    ├── follows celestial star field → STAR
    │
    ├── independent continuous motion → SATELLITE
    │
    ├── aircraft morphology/flashing → AIRCRAFT
    │
    └── uncertain → UNKNOWN_POINT_SOURCE
```

Potential OpenCV functionality:

```python
cv2.calcOpticalFlowPyrLK
```

Initially, unidentified moving objects should simply be rejected from the localisation solver.

Future versions may use identified satellites as additional known celestial references.

---

# 14. Pixel-to-Celestial Conversion

Every valid detection must eventually be represented as an angular observation.

Do NOT publish only azimuth.

Each observation should contain at least:

```text
azimuth
elevation
```

Preferably represent internally as a 3D unit vector as well:

```text
[x, y, z]
```

This avoids problems associated with angular wraparound.

Conceptually:

```text
detected pixel
      │
      ▼
camera calibration
      │
      ▼
camera-frame ray
      │
      ▼
robot/celestial frame
      │
      ▼
azimuth + elevation
```

---

# 15. Custom ROS Messages

Create:

```text
CelestialObservation.msg
```

Suggested contents:

```text
std_msgs/Header header

uint8 UNKNOWN=0
uint8 STAR=1
uint8 SUN=2
uint8 MOON=3
uint8 SATELLITE=4
uint8 AIRCRAFT=5

uint8 object_type

string object_id

float64 azimuth
float64 elevation

float64 angular_uncertainty
float64 confidence

float64 pixel_x
float64 pixel_y

float64 brightness
```

For stars:

```text
object_id
```

could contain a catalogue identifier such as:

```text
HIP_32349
```

For the Sun and Moon:

```text
SUN
MOON
```

---

Create:

```text
CelestialObservationArray.msg
```

Suggested contents:

```text
std_msgs/Header header
CelestialObservation[] observations
```

Publish on:

```text
/celestial_observations
```

---

# 16. Node 3 — `celestial_localizer`

## Purpose

Determine the terrestrial position and orientation of the robot by comparing observed celestial locations against predicted celestial locations.

Inputs:

```text
/celestial_observations
```

plus:

```text
UTC date/time
```

and optionally:

```text
IMU/gravity orientation
initial position estimate
altitude estimate
```

---

# 17. Astronomy Libraries

## Astropy

Primary astronomy/coordinate dependency:

```text
astropy
```

Important modules:

```python
from astropy.time import Time

from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    AltAz,
    get_body
)
```

Astropy should be used for:

- Astronomical time handling
- Celestial coordinate frames
- ICRS transformations
- Altitude/azimuth calculations
- Earth observer locations
- Solar-System body coordinates
- Coordinate transformations

---

# 18. Skyfield

Also investigate/use:

```text
skyfield
```

Skyfield is particularly useful for ephemeris prediction.

Potential applications:

- Sun position
- Moon position
- Planetary positions
- Star positions
- Satellite positions
- JPL ephemerides

The architecture should isolate ephemeris calculation behind an interface so that Astropy and Skyfield implementations can be compared or interchanged.

For example:

```python
class EphemerisProvider:

    def predict(
        self,
        object_id,
        timestamp,
        latitude,
        longitude,
        altitude
    ):
        ...
```

---

# 19. Ephemeris Matching

Given:

```text
UTC time
candidate latitude
candidate longitude
candidate altitude
```

the ephemeris module should predict:

```text
expected azimuth
expected elevation
```

for every identified celestial object.

Example:

```text
Observed:

Sirius
Az = 212.47°
El = 31.34°

Moon
Az = 145.92°
El = 42.11°

Sun
Az = 251.84°
El = 18.42°
```

For a candidate terrestrial location, calculate predicted values and compare them with observations.

---

# 20. Localisation State

Initial optimization state:

\[
x =
[
latitude,
longitude,
yaw
]
\]

Potential future state:

\[
x =
[
latitude,
longitude,
altitude,
roll,
pitch,
yaw
]
\]

For the first implementation, altitude may be assumed known or fixed.

Roll and pitch should preferably come from the robot IMU/gravity vector.

This leaves the main unknowns:

```text
latitude
longitude
heading/yaw
```

---

# 21. Position Solver

Recommended library:

```text
scipy
```

Use:

```python
from scipy.optimize import least_squares
```

The optimizer should minimize angular error between:

```text
observed celestial direction
```

and:

```text
predicted celestial direction
```

for all observations.

Conceptually:

\[
\hat{x}
=
\arg\min_x
\sum_i
w_i
d(
u_{observed,i},
u_{predicted,i}(x,t)
)^2
\]

where:

- \(x\) is the candidate robot state
- \(t\) is UTC
- \(u\) is a celestial unit vector
- \(d\) is angular distance
- \(w_i\) represents confidence/uncertainty weighting

---

# 22. Prefer Vector Residuals

Where practical, compare celestial unit vectors rather than directly subtracting azimuth/elevation.

This avoids issues such as:

```text
359.9° - 0.1°
```

incorrectly appearing to produce a ~360° residual.

Represent celestial observations as:

\[
u =
[x,y,z]
\]

and compute angular separation using:

\[
\theta =
\arccos(
u_{observed}
\cdot
u_{predicted}
)
\]

Clamp the dot product numerically into:

```text
[-1, 1]
```

before applying `acos`.

---

# 23. Robust Optimization

Use confidence and uncertainty to weight measurements.

For example:

```text
high-confidence star → strong weight
partially obscured Moon → lower weight
saturated Sun → lower weight
uncertain star match → low weight/reject
```

Investigate robust losses such as:

```python
loss="soft_l1"
```

with SciPy `least_squares`.

This will reduce the effect of incorrect celestial identifications.

---

# 24. Initial Search

Do not assume the nonlinear optimizer will always converge from an arbitrary position on Earth.

Implement a coarse-to-fine strategy.

Potential first implementation:

```text
Coarse global grid
        │
        ▼
evaluate celestial residual
        │
        ▼
best N candidate locations
        │
        ▼
SciPy nonlinear optimization
        │
        ▼
final position
```

If prior location information is available, allow the global search to be constrained.

Example ROS parameters:

```yaml
localisation:
  global_search: true

  initial_latitude: null
  initial_longitude: null

  search_radius_km: null
```

---

# 25. Time

Accurate time is essential.

Use ROS timestamps where appropriate, but astronomy calculations should explicitly convert timestamps into appropriate astronomical time representations using Astropy/Skyfield.

The system should not assume that ROS system time is automatically suitable without conversion.

Store the timestamp associated with the **actual image exposure**, rather than simply the time at which Node 3 receives the observation.

---

# 26. IMU Integration

An IMU should eventually provide:

- Gravity direction
- Roll
- Pitch
- Approximate heading where available

Gravity is particularly valuable because celestial localisation depends on the relationship between celestial observations and the local vertical.

Potential input:

```text
/imu/data
```

Type:

```text
sensor_msgs/msg/Imu
```

The system should allow IMU use to be enabled/disabled through parameters.

---

# 27. Localisation Outputs

Primary output:

```text
/celestial_pose
```

Type:

```text
geometry_msgs/msg/PoseWithCovarianceStamped
```

The localisation system should also publish a global geographic position where useful:

```text
/celestial_fix
```

Type:

```text
sensor_msgs/msg/NavSatFix
```

This does NOT imply GNSS was used.

`NavSatFix` simply provides a convenient standard representation for:

```text
latitude
longitude
altitude
covariance
```

---

# 28. TF Output

Publish an appropriate TF representing the resulting localisation.

Potential frames:

```text
earth
  │
  ▼
map
  │
  ▼
odom
  │
  ▼
base_link
```

The celestial localizer should NOT unnecessarily replace normal ROS localisation architecture.

Instead, treat celestial localisation as a **global absolute positioning source**, conceptually similar to GNSS.

The exact TF ownership should be configurable so the system can later integrate with:

```text
robot_localisation
Nav2
```

---

# 29. Uncertainty

Do not output a position without uncertainty.

Sources of uncertainty include:

- Star centroid error
- Star identification confidence
- Sun centre uncertainty
- Moon centre uncertainty
- Camera calibration uncertainty
- IMU orientation uncertainty
- Timestamp uncertainty
- Atmospheric refraction
- Ephemeris uncertainty
- Optimization residual

The first implementation may use approximate covariance estimation.

Later implementations should propagate measurement uncertainty into the final pose covariance more rigorously.

---

# 30. Configuration

All important values should be ROS parameters rather than hard-coded constants.

Example:

```yaml
sky_mapper:
  ros__parameters:

    panorama_width: 4096
    panorama_height: 2048

    projection: "equirectangular"

    calibration_file: ""

celestial_detector:
  ros__parameters:

    detect_stars: true
    detect_sun: true
    detect_moon: true

    reject_aircraft: true
    reject_satellites: true

    star_detection_threshold: 5.0

    minimum_confidence: 0.5

celestial_localizer:
  ros__parameters:

    use_stars: true
    use_sun: true
    use_moon: true

    use_imu: true

    fixed_altitude: 0.0

    global_search: true

    robust_loss: "soft_l1"

    publish_tf: true
```

---

# 31. Python Dependencies

Initial dependencies should include approximately:

```text
numpy
scipy
opencv-python
astropy
skyfield
photutils
py360convert
tetra3
```

Optional:

```text
ultralytics
```

ROS dependencies:

```text
rclpy
sensor_msgs
geometry_msgs
std_msgs
tf2_ros
builtin_interfaces
```

Custom interfaces:

```text
celestial_interfaces
```

---

# 32. Recommended Processing Pipeline

The complete first-stage pipeline should resemble:

```text
              ┌─────────────────┐
              │  360° CAMERA    │
              └────────┬────────┘
                       │
                       ▼
               ┌───────────────┐
               │  sky_mapper   │
               └───────┬───────┘
                       │
               Equirectangular
                  sky map
                       │
                       ▼
              ┌──────────────────┐
              │celestial_detector│
              └────────┬─────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Photutils      OpenCV       OpenCV
       STARS           SUN          MOON
          │            │            │
          ▼            │            │
        Tetra3         │            │
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
             angular observations
                       │
                       ▼
             ┌────────────────────┐
             │celestial_localizer │
             └─────────┬──────────┘
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          Astropy            Skyfield
       coordinates          ephemerides
              │                 │
              └────────┬────────┘
                       │
                       ▼
                SciPy solver
                       │
                       ▼
             latitude/longitude
                  + heading
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
         NavSatFix    Pose        TF
```

---

# 33. Development Stages

## Stage 1 — Synthetic Ephemeris Solver

Before implementing camera perception, test whether terrestrial location can be recovered from perfect synthetic celestial observations.

Create a Python/ROS test that:

1. Defines a known latitude/longitude.
2. Defines a known UTC timestamp.
3. Uses Astropy/Skyfield to generate synthetic celestial observations.
4. Passes these observations to the localisation solver.
5. Starts the solver from an incorrect location.
6. Attempts to recover the original location.
7. Reports position error in metres.

This should be implemented FIRST.

It isolates the fundamental localisation mathematics from camera/perception problems.

---

## Stage 2 — Add Measurement Noise

Introduce controlled angular noise:

```text
0.01°
0.005°
0.001°
10 arcsec
5 arcsec
1 arcsec
```

Measure resulting localisation accuracy.

Produce:

```text
angular measurement error
        versus
position localisation error
```

This experiment will establish the required camera/angular accuracy.

---

## Stage 3 — Sun and Moon Solver

Test localisation using synthetic:

```text
Sun
Moon
```

observations.

Determine observability and accuracy.

---

## Stage 4 — Star Solver

Add synthetic identified stars.

Investigate:

```text
number of stars
star geometry
angular uncertainty
localisation accuracy
```

---

## Stage 5 — Real Star Images

Implement:

```text
Photutils
+
Tetra3
```

and validate against known astronomical images.

---

## Stage 6 — Real Sky Camera

Connect the 360° camera.

Implement:

```text
camera
→ sky_mapper
→ celestial_detector
```

Validate measured celestial angles against expected Astropy/Skyfield positions from a known location.

---

## Stage 7 — Full Celestial Localisation

Run:

```text
camera
→ sky map
→ celestial detection
→ ephemeris matching
→ terrestrial position solver
```

Compare estimated position against known ground truth.

---

## Stage 8 — ROS Localisation Integration

Integrate output with:

```text
robot_localisation
```

and potentially:

```text
Nav2
```

Treat the celestial position estimate as an intermittent global absolute localisation measurement.

---

# 34. Future Stage — Atmospheric Visual Odometry

The architecture should anticipate a later second research component:

**Atmospheric Visual Odometry / Cloud-Based Visual Odometry**

This should NOT initially be implemented as part of the celestial localizer.

Future architecture:

```text
              SKY CAMERA
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
     Celestial         Atmospheric
    Localisation       Odometry
          │                │
 absolute global      relative motion
     position             estimate
          │                │
          └───────┬────────┘
                  ▼
            localisation
               fusion
                  │
                  ▼
            robot pose
```

The atmospheric system may investigate:

- Cloud segmentation
- Cloud classification
- Optical flow
- Feature tracking
- Cloud altitude estimation
- Cloud velocity estimation
- Cloud deformation
- Wind information
- Weather-station information
- Separation of robot ego-motion from cloud motion

Conceptually:

```text
observed cloud motion
        =
robot ego-motion
        +
cloud translation
        +
cloud deformation
```

---

# 35. Design Principles

When generating the project, follow these principles.

### Modularity

Do not tightly couple perception and localisation.

Each detector should be replaceable.

---

### ROS-native interfaces

Prefer standard ROS messages wherever suitable.

Use custom messages only for celestial observations that do not have an appropriate standard representation.

---

### Calibration over assumptions

Do not assume perfect pixel-to-angle mapping.

All angular observations should pass through calibration.

---

### Preserve raw measurements

Do not discard:

```text
pixel position
confidence
uncertainty
timestamp
```

after calculating azimuth/elevation.

These may be required later for debugging and uncertainty estimation.

---

### Unit vectors internally

Prefer 3D normalized vectors for celestial direction calculations.

Convert to azimuth/elevation primarily for human readability and ROS messages.

---

### Uncertainty-aware localisation

Every observation should have confidence/uncertainty.

The localisation solver should weight observations appropriately.

---

### Simulation before perception

Develop and validate the celestial position solver using synthetic perfect observations before introducing camera perception.

This allows localisation mathematics and perception errors to be evaluated independently.

---

### Offline operation

The eventual system should be capable of operating without Internet access.

Required:

- Star catalogues
- Ephemerides
- Calibration
- Models

should be available locally.

This is important for GNSS-denied and remote robotic applications.

---

# 36. Initial Scope

The minimum viable implementation should achieve:

```text
KNOWN:
    UTC
    camera calibration
    robot stationary
    approximate/fixed altitude
    gravity orientation

OBSERVED:
    identified celestial objects
    azimuth
    elevation

SOLVE:
    latitude
    longitude
    heading

OUTPUT:
    global position
    pose
    covariance
    TF
```

Cloud-based localisation is explicitly outside the minimum viable implementation.

---

# 37. Initial Success Criterion

Given synthetic observations generated at a known:

```text
latitude
longitude
UTC timestamp
```

the system must successfully recover the original terrestrial location.

After this has been demonstrated, progressively introduce:

```text
angular noise
→ imperfect star measurements
→ real astronomical imagery
→ 360° camera observations
→ outdoor robotic testing
```

This staged approach should be maintained throughout development so that failures can be attributed separately to:

```text
celestial geometry
ephemeris calculation
optimization
camera calibration
object detection
star identification
or ROS integration
```

rather than debugging the entire pipeline simultaneously.