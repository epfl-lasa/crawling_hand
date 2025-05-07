#!/bin/bash

set -o pipefail

# Set up a ROS workspace
mkdir -p $HAND_DEP_WS/src
cd $HAND_DEP_WS
source /opt/ros/${ROS_DISTRO}/setup.bash
catkin init
catkin config --extend /opt/ros/${ROS_DISTRO}
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

# Install ROS packages from apt
apt-get update && apt-get install -y \
	ros-${ROS_DISTRO}-ros-control \
	ros-${ROS_DISTRO}-ros-controllers \
	ros-${ROS_DISTRO}-moveit \
	ros-${ROS_DISTRO}-rosmon \
	ros-${ROS_DISTRO}-pcl-ros \
	ros-${ROS_DISTRO}-tf2-sensor-msgs \
	ros-${ROS_DISTRO}-py-trees \
	ros-${ROS_DISTRO}-py-trees-ros \
	ros-${ROS_DISTRO}-rqt-py-trees \
	ros-${ROS_DISTRO}-joint-state-publisher-gui \
	ros-${ROS_DISTRO}-ddynamic-reconfigure \
	ros-${ROS_DISTRO}-interactive-marker-twist-server \
	ros-${ROS_DISTRO}-ros-numpy \
	ros-${ROS_DISTRO}-smach \
	ros-${ROS_DISTRO}-smach-ros \
	ros-${ROS_DISTRO}-tf-conversions \
	ros-${ROS_DISTRO}-rviz-visual-tools \
    ros-${ROS_DISTRO}-fkie-multimaster \
    ros-${ROS_DISTRO}-fkie-node-manager \
	ros-${ROS_DISTRO}-moveit-resources-panda-description \
	ros-${ROS_DISTRO}-moveit-visual-tools \
	ros-${ROS_DISTRO}-apriltag-ros

apt-get install ros-${ROS_DISTRO}-orocos-kdl
apt-get install ros-${ROS_DISTRO}-python-orocos-kdl

# Clear cache -> keep layer size down
rm -rf /var/lib/apt/lists/*

