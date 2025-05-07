FROM osrf/ros:noetic-desktop-full

# This docker is intended to run on a development machine.
# No CUDA (for now), but with simulation and without sensor drivers.

# Copy scripts folder
COPY scripts/ /root/scripts/
WORKDIR /root/
RUN chmod a+x -R /root/scripts

# Env variables
ENV SCRIPTS_PATH=/root/scripts
ENV HAND_DEP_WS=/root/hand_dep_ws
ENV ROS_DISTRO=noetic

# Run the general dep installation
RUN scripts/install_sys_deps.sh

# Run the ROS workspac  & dep installation
RUN scripts/install_ros_deps.sh

# Install python packages
RUN scripts/install_python_deps.sh

# Build all the stuff
RUN scripts/build_ros.sh
