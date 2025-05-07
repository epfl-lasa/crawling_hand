#!/bin/bash
set -o pipefail

# This script builds all the installed ROS packages and sets up the bashrc.
cd $HAND_DEP_WS
catkin build -c

# Add sourcing of the repo to the ~/.bashrc
echo "source $HAND_DEP_WS/devel/setup.bash" >> ~/.bashrc
echo 'ROSBASH=/root/hand_ws/devel/setup.bash && [ -e "$ROSBASH" ] && source $ROSBASH' >> ~/.bashrc
