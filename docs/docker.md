# Docker operations

## Compose files

Run commands from the repository root. The root `compose.yml` combines the
three services and is the normal entry point. The component Compose files can
also be run independently when developing one container:

```bash
docker compose -f realsense_docker/compose.yml up --build
docker compose -f celestial_localisation/compose.yml up --build
```

The `compose.how_to_use.yml` files contain example overrides for each
component. They are useful for testing explicit camera and localisation
settings without changing the base Compose files.

## RealSense access

`realsense_service` requires a connected camera, USB device access, the
`video` group, and privileged container mode. Set `REALSENSE_SERIAL_NUMBER`
when more than one camera is connected or a particular device is required.
The startup script prefixes the serial with `_` for the RealSense launch
argument when necessary.

## RViz and X11

RViz uses host networking and mounts `/tmp/.X11-unix`. Before starting
`rviz_service`, permit local container clients to use the current X server:

```bash
xhost +local:
docker compose up --build rviz_service
```

Revoke that permission after the session when appropriate:

```bash
xhost -local:
```

Set `DISPLAY` in the shell when the display is not the default. The root
Compose file passes it through to RViz.

## Rebuilding after source changes

Both the RealSense and celestial startup scripts source ROS 2 Humble and run
`colcon build --symlink-install`. Recreate the affected service after source
or dependency changes:

```bash
docker compose up --build --force-recreate celestial_localisation_service
```

The celestial source directory is bind-mounted read/write, so host edits are
visible in the container. The image build still copies the source tree and
installs system and Python dependencies, so use `--build` after changing a
Dockerfile or package dependency.
