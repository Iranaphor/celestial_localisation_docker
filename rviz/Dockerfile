FROM ros:humble

ARG USERNAME=ros
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid ${GID} ${USERNAME} \
    && adduser --uid ${UID} --gid ${GID} --disabled-password --gecos "" ${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

CMD ["/bin/bash"]