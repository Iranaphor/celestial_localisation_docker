FROM ros:humble

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=ros
ARG UID=1000
ARG GID=1000

ENV USERNAME=${USERNAME}
ENV ROS_WS=/home/${USERNAME}/ros2_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    build-essential \
    python3-colcon-common-extensions \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-launch \
    ros-humble-launch-ros \
    && rm -rf /var/lib/apt/lists/*

# opencv/numpy/scipy are installed via pip (not apt) so their binary wheels
# share a consistent NumPy ABI. NumPy is pinned below 2.x because
# ros-humble-cv-bridge (apt) is compiled against NumPy 1.x and crashes with
# "_ARRAY_API not found" under NumPy 2.x.
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    scipy \
    opencv-python-headless \
    astropy \
    skyfield \
    photutils \
    py360convert

RUN addgroup --gid ${GID} ${USERNAME} \
    && adduser --uid ${UID} --gid ${GID} --disabled-password --gecos "" ${USERNAME} \
    && usermod -a -G dialout ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p ${ROS_WS}/src \
    && chown -R ${USERNAME}:${USERNAME} ${ROS_WS}

COPY start.sh /home/${USERNAME}/start.sh
RUN chmod +x /home/${USERNAME}/start.sh

COPY src/ ${ROS_WS}/src/

RUN /bin/bash -lc "source /opt/ros/humble/setup.bash && cd ${ROS_WS} && colcon build --symlink-install"
RUN chown -R ${USERNAME}:${USERNAME} ${ROS_WS}

USER ${USERNAME}
WORKDIR ${ROS_WS}

RUN echo 'source /opt/ros/humble/setup.bash' >> /home/${USERNAME}/.bashrc \
    && echo "source ${ROS_WS}/install/setup.bash" >> /home/${USERNAME}/.bashrc

CMD ["/bin/bash"]
