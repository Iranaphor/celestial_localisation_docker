# Future Work: Multi-Layer Sky-Based Localization

## Overview

This project investigates **sky-based localization for mobile robots**, using celestial and atmospheric observations as alternatives or complementary sources to conventional GNSS-based positioning.

The current implementation establishes the core **celestial localization pipeline**: an upward-facing camera is converted into a calibrated spherical representation of the sky, celestial objects are detected, their observed angular positions are compared against astronomical ephemerides, and the resulting observations are used to estimate the robot's global pose.

The longer-term research direction is to extend this from a purely celestial system into a **multi-layer sky localization framework**.

The central hypothesis is that different visible layers of the sky provide different localization information because they:

* exist at very different effective distances;
* exhibit different types of motion;
* have different levels of predictability;
* become available under different environmental conditions;
* provide different constraints on absolute position, orientation, and relative motion.

Potential observation layers include:

```text
Stars
   ↓
Sun / Moon
   ↓
Satellites
   ↓
Aircraft
   ↓
High-altitude clouds
   ↓
Mid-altitude clouds
   ↓
Low-altitude clouds
   ↓
Horizon / local terrain
   ↓
Robot
```

The intention is not that every layer must always be available. Instead, the eventual system could dynamically use whichever sky observations are currently reliable.

---

# Current Implementation

The following components have already been implemented.

## Calibrated Equirectangular Sky-Map Generation

Camera imagery is transformed into a calibrated equirectangular representation of the visible sky.

The sky map provides a consistent spherical image representation in which observations can ultimately be related to:

* azimuth;
* elevation;
* camera-frame rays;
* robot-frame directions.

Conceptually:

```text
Camera Images
      │
      ▼
Camera Calibration
      │
      ▼
Spherical Projection / Stitching
      │
      ▼
Calibrated Equirectangular Sky Map
```

Maintaining a calibrated projection is important because the localization system ultimately depends on **angular measurements**, rather than merely detecting whether an object is present in an image.

---

## Sun, Moon and Raw Star Detection

The current perception system detects:

* the Sun;
* the Moon;
* raw stellar point sources.

These detections are converted from image coordinates into observations within the calibrated sky representation.

Raw star detection currently identifies candidate stellar sources but does **not yet perform catalogue-based star identification**.

Therefore the current distinction is:

```text
Star detection      → IMPLEMENTED
Star identification → FUTURE WORK
```

---

## Celestial Observation ROS Interfaces

A ROS 2 topic pipeline has been implemented for passing celestial observations between the perception and localization components.

The architecture follows the general structure:

```text
Sky Camera
    │
    ▼
Sky-Map Generation
    │
    ▼
Celestial Detection
    │
    ▼
Celestial Observation Messages
    │
    ▼
Ephemeris Matching
    │
    ▼
Pose Estimation
```

The observation interface allows celestial measurements to be separated from the localization implementation.

This modularity is intended to allow future observation sources such as identified stars, satellites and aircraft to be introduced without redesigning the complete localization pipeline.

---

## Astropy Ephemeris Matching

The current localization system uses **Astropy** for astronomical coordinate and ephemeris calculations.

Observed celestial directions can be compared against the expected positions of known astronomical objects for a candidate:

* latitude;
* longitude;
* timestamp;
* observer orientation.

This provides the connection between image-space observations and terrestrial global position.

---

## SciPy Pose Solver

The current system uses **SciPy optimization** to estimate the robot pose that best explains the observed celestial measurements.

Conceptually:

$$
\hat{x}
=
\arg\min_x
\sum_i
w_i
d
\left(
u_{observed,i},
u_{predicted,i}(x,t)
\right)^2
$$

where:

* \(x\) represents the robot pose parameters;
* \(t\) represents observation time;
* \(u_{observed}\) is an observed celestial direction;
* \(u_{predicted}\) is the corresponding predicted direction;
* \(d\) represents angular disagreement;
* \(w_i\) allows observations to be weighted according to confidence.

This establishes the core:

```text
OBSERVE SKY
     ↓
PREDICT SKY
     ↓
COMPARE
     ↓
OPTIMISE
     ↓
GLOBAL POSE
```

pipeline.

---

# Current Development Status

| Component                                     | Status                        |
| --------------------------------------------- | ----------------------------- |
| Calibrated equirectangular sky-map generation | **Implemented**               |
| Sun detection                                 | **Implemented**               |
| Moon detection                                | **Implemented**               |
| Raw star detection                            | **Implemented**               |
| Pixel-to-angular observation conversion       | **Implemented**               |
| Celestial ROS message interfaces              | **Implemented**               |
| ROS topic pipeline                            | **Implemented**               |
| Astropy ephemeris calculations                | **Implemented**               |
| SciPy pose optimization                       | **Implemented**               |
| Catalogue-based star identification           | **Future Work**               |
| Aircraft detection/rejection                  | **Future Work**               |
| Satellite detection/rejection                 | **Future Work**               |
| Cloud classification                          | **Future Work**               |
| Cloud tracking                                | **Future Work**               |
| Atmospheric visual odometry                   | **Future Work**               |
| Aircraft-assisted differential parallax       | **Future Work / Exploratory** |
| Satellite-assisted localization               | **Future Work / Exploratory** |
| IMU fusion                                    | **Future Work**               |
| Condition-adaptive sensor fusion              | **Future Work / Exploratory** |

---

# Future Work 1: Catalogue-Based Star Identification

The current system detects stellar point sources but does not associate them with known stars.

A future extension should perform **lost-in-space star identification**, matching geometric arrangements of detected stars against an astronomical catalogue.

Potential software includes:

* `tetra3`;
* `tetra3rs`;
* custom catalogue matching using Astropy-compatible star catalogues.

The intended pipeline is:

```text
Detected Star Centroids
         │
         ▼
Geometric Pattern Extraction
         │
         ▼
Star Catalogue Matching
         │
         ▼
Identified Stars
         │
         ▼
Known Celestial Coordinates
```

An identified observation could subsequently contain a catalogue identifier such as:

```text
Object Type: STAR
Object ID: HIP_32349
Azimuth: ...
Elevation: ...
Confidence: ...
Angular Uncertainty: ...
```

This would allow individual stellar observations to participate directly in the existing ephemeris/localization solver.

Stars are particularly useful because their effectively infinite distance makes them extremely stable **inertial orientation references**.

---

# Future Work 2: Aircraft and Satellite Detection

Moving point sources currently risk contaminating stellar observations.

Two particularly important categories are:

* aircraft;
* artificial satellites.

Future perception should identify and reject these sources before star catalogue matching.

Temporal observations are particularly valuable:

```text
Point Source
     │
     ├── Consistent with celestial rotation
     │       └── STAR
     │
     ├── Smooth independent trajectory
     │       └── SATELLITE candidate
     │
     ├── Aircraft morphology / lighting
     │       └── AIRCRAFT
     │
     └── Otherwise
             └── UNKNOWN
```

Rather than treating these objects permanently as contaminants, later research may investigate using them as **localization references themselves**.

---

# Future Work 3: Satellite-Assisted Localization

Artificial satellites occupy an interesting intermediate localization layer.

Unlike stars, they are relatively close to Earth and move rapidly across the sky. However, their orbital trajectories can often be predicted from known orbital information.

Conceptually:

```text
Satellite Catalogue / Orbit
             +
             UTC
             +
     Candidate Robot Position
             │
             ▼
 Predicted Satellite Direction
             │
             ↕
 Observed Satellite Direction
```

This potentially provides an additional absolute positioning constraint.

Satellite localization should initially be treated as exploratory work rather than a dependency of the core celestial localization system.

---

# Future Work 4: Aircraft as Opportunistic References

Aircraft generally operate at approximately kilometre-scale ranges, with commercial aircraft commonly cruising around the 8–12 km altitude region.

Their trajectories are substantially more predictable over short intervals than cloud motion.

An aircraft observed over a sliding temporal window may therefore provide a useful intermediate-distance reference.

For example:

```text
t - 30 s                           t

Plane:    •───•───•───•───•───•───•

Cloud:   █████ → █████ → █████

Robot:        ●──────→
```

The aircraft trajectory should **not** be treated as a static landmark.

Observed aircraft motion contains contributions from:

$$
\text{Observed Aircraft Motion}
=
\text{Aircraft Motion}
+
\text{Robot Motion}
+
\text{Camera Rotation}
$$

Instead, a temporal trajectory model can estimate the aircraft's approximately smooth motion.

Where external aircraft information is available, for example ADS-B information, the aircraft could potentially become a much stronger known moving reference.

Such information may include:

* aircraft latitude/longitude;
* altitude;
* velocity;
* heading;
* timestamp.

This transforms the aircraft from an unknown moving object into an **opportunistic moving landmark**.

The localization system should nevertheless remain capable of operating without Internet or external aircraft data.

---

# Future Work 5: Multiple Atmospheric Cloud Layers

Clouds should not be considered a single atmospheric layer.

Different cloud types occupy different characteristic altitude ranges and exhibit different:

* wind velocities;
* deformation rates;
* persistence;
* visual textures;
* optical-flow characteristics.

A simplified atmospheric representation might include:

```text
HIGH CLOUDS
~5–12+ km
Cirrus
Cirrostratus
Cirrocumulus

        ↓

MID-LEVEL CLOUDS
~2–7 km
Altostratus
Altocumulus

        ↓

LOW CLOUDS
~0–2 km
Stratus
Stratocumulus
Cumulus

        ↓

VERTICALLY DEVELOPING CLOUDS
Potentially spanning multiple layers
Towering Cumulus
Cumulonimbus
```

Exact cloud altitude and velocity vary substantially with meteorological conditions and location.

Consequently, altitude should ultimately be represented as an **estimated state or probability distribution**, rather than a fixed lookup value based only on cloud class.

---

# Future Work 6: Cloud Classification

A future cloud perception node should estimate:

```text
Sky Image
    │
    ▼
Cloud Segmentation
    │
    ▼
Cloud Regions
    │
    ▼
Cloud Classification
    │
    ▼
Approximate Atmospheric Layer
```

Possible information associated with each cloud observation could include:

```text
cloud_class
segmentation_mask
estimated_altitude
altitude_uncertainty
optical_flow
estimated_velocity
deformation_score
tracking_confidence
```

The purpose of classification is not simply meteorological labelling.

Cloud type provides a prior on:

* likely altitude;
* likely persistence;
* likely deformation;
* suitability for visual tracking.

---

# Future Work 7: Atmospheric Visual Odometry

Clouds cannot normally provide an absolute global position because their positions are unknown and continuously changing.

They may, however, provide **relative motion information**.

This creates a distinction between the two major parts of the proposed system:

```text
CELESTIAL OBSERVATIONS
Stars / Sun / Moon
        │
        ▼
Absolute Pose Information


ATMOSPHERIC OBSERVATIONS
Clouds
        │
        ▼
Relative Motion / Odometry
```

Cloud motion observed by the camera can conceptually be decomposed as:

$$
f_{observed}
=
f_{robot}
+
f_{atmosphere}
+
f_{deformation}
$$

where:

* \(f_{robot}\) represents apparent motion caused by robot ego-motion;
* \(f_{atmosphere}\) represents physical cloud movement;
* \(f_{deformation}\) represents changing cloud geometry.

The research problem becomes estimating these components sufficiently well to recover useful robot motion.

---

# Future Work 8: Differential Sky Parallax

A major exploratory direction is to exploit the different effective distances of sky objects.

Robot translation causes stronger apparent parallax for nearby objects than distant objects.

For a simplified geometry:

$$
\theta \approx \frac{b}{d}
$$

where:

* \(b\) is the robot translation baseline;
* \(d\) is object distance;
* \(\theta\) is apparent angular displacement.

Consequently:

```text
STARS                     CLOUDS

effectively ∞             ~1 km

      *                     ███
      │                      ╲
      │                       ╲
      │                        ╲
Robot ●────────────→●          ●────→●
      very little             measurable
      parallax                parallax
```

This creates the possibility of using multiple atmospheric/celestial depth layers to constrain robot translation.

---

# Future Work 9: Aircraft-Assisted Cloud Parallax

Aircraft may be particularly useful because they occupy a substantially different distance layer from many clouds.

Consider:

```text
Aircraft:       ~10 km
                   ✈
                  /
                 /

High cloud:      ~8 km
              ─────────

Low cloud:       ~1 km
            █████████████


Robot:          ●──────→
```

The same robot translation produces different angular parallax for each layer.

A possible pipeline is therefore:

```text
             Sliding Sky Image Buffer
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Aircraft Tracking        Cloud Tracking
          │                       │
          ▼                       ▼
Trajectory Estimation      Dense Optical Flow
          │                       │
          ▼                       ▼
Altitude / Motion Prior    Cloud-Layer Estimate
          │                       │
          └───────────┬───────────┘
                      ▼
             Rotation Compensation
                      │
                      ▼
             Differential Parallax
                      │
                      ▼
          Robot Translation Estimate
```

This should currently be regarded as an **experimental research hypothesis**, not an established localization technique.

---

# Future Work 10: Spherical Trajectory Representation

Aircraft, satellites and cloud features should not be tracked as straight lines directly in equirectangular pixel coordinates.

Equirectangular projection introduces substantial geometric distortion.

Observations should instead be converted:

```text
Image Pixel
    │
    ▼
Calibrated Camera Ray
    │
    ▼
Azimuth / Elevation
    │
    ▼
3D Unit Vector
```

with:

$$
u =
[x,y,z]
$$

Temporal trajectories can then be fitted on the sphere.

This allows tracking algorithms to remain independent of the particular visualization/projection used for the sky map.

---

# Future Work 11: Sliding Temporal Window

Atmospheric localization requires observations over time.

A useful initial design would retain approximately the previous **30 seconds** of sky observations.

However, processing should not necessarily assume that the entire 30-second period represents a single rigid interval.

Instead:

```text
                   30 s history

t-30 ────────────────────────────────── t

      [──── 10 s ────]
            [──── 10 s ────]
                  [──── 10 s ────]
                        [──── 10 s ────]
```

Overlapping windows may provide a better trade-off between:

### Longer windows

* Greater robot baseline
* Larger measurable parallax
* Better aircraft trajectory estimation

### Shorter windows

* Less cloud deformation
* More reliable optical flow
* Lower probability of losing tracked features

The window duration could eventually be dynamically selected according to robot velocity and atmospheric conditions.

---

# Future Work 12: IMU Fusion

IMU integration remains unimplemented.

This is an important future addition because image motion produced by camera rotation can be much larger than translational sky parallax.

An IMU can provide:

* gravity direction;
* roll;
* pitch;
* rotational velocity;
* short-term rotational odometry.

The atmospheric pipeline could therefore perform:

```text
Observed Sky Motion
        │
        ├── IMU Rotation Estimate
        │
        ▼
Remove Camera Rotation
        │
        ▼
Residual Sky Motion
        │
        ▼
Translation + Atmospheric Motion
```

Celestial observations may also provide an independent absolute orientation reference.

---

# Future Work 13: Polarized Sunlight

Another potential sky-derived orientation source is the polarization pattern of daylight.

Scattered sunlight produces a structured polarization field across the sky related to the position of the Sun.

With appropriate polarization-sensitive sensing, this could potentially provide:

* Sun-direction information;
* absolute heading constraints;
* orientation information when the solar disc itself is obscured.

This would primarily contribute to **absolute orientation**, rather than directly providing translational odometry.

It should therefore be treated as another complementary sky observation modality.

---

# Absolute Localization vs Odometry

A key distinction for future development is whether each observation contributes to **global pose** or **relative motion**.

| Observation                    |  Absolute Translation  |  Absolute Orientation  |    Relative Odometry   |
| ------------------------------ | :--------------------: | :--------------------: | :--------------------: |
| Stars                          |       Potentially      |       **Strong**       |         Limited        |
| Sun                            |       Potentially      |       **Strong**       |         Limited        |
| Moon                           |       **Useful**       |       **Useful**       |         Limited        |
| Satellites                     |   Potentially useful   |         Useful         |   Potentially useful   |
| Aircraft without external data |          Weak          |          Weak          | **Potentially useful** |
| Aircraft + ADS-B               | **Potentially useful** |         Useful         |       **Useful**       |
| High clouds                    |           No           |           No           | **Potentially useful** |
| Mid-level clouds               |           No           |           No           | **Potentially useful** |
| Low clouds                     |           No           |           No           | **Potentially strong** |
| Weather information            |      Context/prior     |      Context/prior     |   Improves modelling   |
| Polarized sunlight             |    No/directly weak    | **Potentially strong** |         Limited        |
| Horizon / terrain              |    Not used globally   |    Local constraint    |   Local geometric cue  |

These contributions should be treated as research hypotheses until experimentally validated.

---

# Horizon and Terrain

The horizon and terrain may provide useful **local geometric information**, but this project does not propose treating arbitrary terrain as a database of globally known visual landmarks.

Doing so would require storing and matching large-scale visual/topological representations of the Earth's surface, which conflicts with the intended lightweight and globally applicable nature of the approach.

Furthermore, distant landmarks:

* may not be visible;
* may be occluded;
* vary strongly with location;
* depend on atmospheric visibility;
* may require extremely large reference datasets.

Therefore:

```text
Horizon / Terrain

NOT:
    globally stored landmark database

POTENTIALLY:
    horizon geometry
    local slope
    skyline motion
    occlusion information
    local visual odometry
```

---

# Environmental Availability

No single sky observation is universally available.

The eventual localization framework should therefore be **condition adaptive**.

## Clear Night

Likely useful:

```text
Stars          ★★★★★
Moon           ★★★★
Satellites     ★★★
Aircraft       ★★★
Clouds         condition-dependent
```

## Clear Day

Likely useful:

```text
Sun            ★★★★★
Aircraft       ★★★★
Clouds         ★★★★
Polarization   ★★★★
Moon           condition-dependent
```

## Overcast

Likely useful:

```text
Clouds         ★★★★★
Aircraft       condition-dependent
Sun            weak/obscured
Stars          unavailable
```

## Broken Cloud

Potentially particularly interesting:

```text
Sun / Moon / Stars
        +
multiple cloud layers
        +
aircraft/satellites
```

because several depth/reference layers may be visible simultaneously.

## Rain / Poor Visibility

Celestial observations may become unavailable.

Atmospheric motion and external meteorological information may still provide contextual information, although visual localization performance is expected to deteriorate.

---

# External Meteorological Information

Internet or locally received weather information could optionally improve atmospheric modelling.

Potential inputs include:

* wind speed;
* wind direction;
* cloud-base altitude;
* cloud-layer estimates;
* pressure;
* frontal systems;
* precipitation;
* storm movement.

These data should be treated as **priors**, not ground truth.

The system should ideally support:

```text
Offline Mode
    │
    └── camera + ephemerides + IMU

Enhanced Mode
    │
    ├── weather observations
    ├── ADS-B
    └── updated satellite/orbital information
```

External connectivity should improve localization rather than being required for basic operation.

---

# Long-Term Multi-Layer Architecture

The eventual system could therefore contain several parallel estimators:

```text
                         SKY OBSERVATIONS
                               │
        ┌──────────────┬───────┼─────────┬──────────────┐
        │              │       │         │              │
        ▼              ▼       ▼         ▼              ▼
    Celestial       Satellite Aircraft   Cloud       Polarization
    Localization    Tracking  Tracking   Tracking      Compass
        │              │       │         │              │
        ▼              ▼       ▼         ▼              ▼
   Absolute Pose      Moving Reference   Atmospheric   Absolute
   / Orientation       Constraints        Odometry    Orientation
        │              │       │         │              │
        └──────────────┴───────┼─────────┴──────────────┘
                               ▼
                      State Estimation / Fusion
                               │
                        ┌──────┴──────┐
                        │             │
                        ▼             ▼
                      IMU        Robot Odometry
                        │             │
                        └──────┬──────┘
                               ▼
                          ROBOT POSE
```

A factor-graph or probabilistic state-estimation architecture may ultimately be more appropriate than directly combining all measurements in a single celestial pose solver.

---

# Core Research Hypothesis

The broader research hypothesis can be summarized as:

> **The sky should not be treated as a single visual surface. It consists of multiple observable layers at dramatically different distances, with different motion models, predictability and environmental availability. These differences may provide complementary constraints for estimating robot position, orientation and relative motion.**

The current celestial localization implementation establishes the first component of this framework.

Future research will investigate whether:

1. catalogue-identified stars improve absolute celestial localization;
2. satellites can provide additional predictable moving references;
3. aircraft can act as opportunistic intermediate-range references;
4. multiple cloud layers can provide useful visual odometry;
5. differential parallax between atmospheric layers can reveal robot translation;
6. IMU measurements can isolate rotational image motion;
7. weather and ADS-B information can improve otherwise ambiguous observations;
8. polarization patterns can provide additional absolute orientation information;
9. dynamically fusing whichever layers are currently observable can provide a more resilient localization system than relying on any individual sky phenomenon.

---

# Development Roadmap

```text
CURRENT
  │
  ├── ✓ Calibrated sky-map generation
  ├── ✓ Sun detection
  ├── ✓ Moon detection
  ├── ✓ Raw star detection
  ├── ✓ Celestial observation interfaces
  ├── ✓ ROS topic pipeline
  ├── ✓ Astropy ephemeris matching
  └── ✓ SciPy celestial pose solver
  │
  ▼
NEXT
  │
  ├── Catalogue-based star identification
  ├── Star-pattern validation
  ├── Aircraft/satellite rejection
  └── IMU integration
  │
  ▼
ATMOSPHERIC PERCEPTION
  │
  ├── Cloud segmentation
  ├── Cloud classification
  ├── Cloud-layer estimation
  ├── Optical flow
  └── Temporal cloud tracking
  │
  ▼
EXPERIMENTAL SKY ODOMETRY
  │
  ├── Rotation compensation
  ├── Cloud motion modelling
  ├── Multi-layer differential parallax
  └── Relative robot motion estimation
  │
  ▼
OPPORTUNISTIC REFERENCES
  │
  ├── Aircraft trajectory estimation
  ├── ADS-B association
  ├── Satellite identification
  └── Multi-depth sky constraints
  │
  ▼
LONG-TERM
  │
  ├── Polarization sensing
  ├── Weather-informed estimation
  ├── Condition-adaptive observation selection
  ├── Multi-modal state estimation
  └── Continuous sky-based localization
```

---

# Summary

The current project provides the foundation for **celestial absolute localization** from an upward-facing camera.

The immediate priority is to complete the conventional celestial pipeline through catalogue-based star identification and IMU integration.

Beyond this, the project proposes a broader interpretation of the sky as a set of **distinct localization layers**:

```text
Far / Predictable
        │
        Stars
        Sun / Moon
        Satellites
        │
        Aircraft
        │
        High Clouds
        Mid Clouds
        Low Clouds
        │
Near / Dynamic
```

Far celestial objects provide stable absolute references.

Intermediate known or semi-known objects such as satellites and aircraft may provide additional geometric constraints.

Nearby atmospheric structures provide stronger parallax but have increasingly complex independent motion and deformation.

The long-term objective is therefore not to derive localization from one particular sky feature, but to build a **condition-adaptive sky perception system that combines absolute celestial references with relative atmospheric motion cues to estimate robot pose when conventional positioning systems are unavailable or unreliable.**
