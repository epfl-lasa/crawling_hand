#!/bin/bash

set -o pipefail

# Update Ubuntu packages to latest.
apt-get -qq update && apt-get -qq upgrade

# update git and set to always point to https
apt-get -qq update && apt-get install -y curl apt-utils tmux tmuxp usbutils
apt-get -qq update && apt-get install -y git git-lfs
apt-get install -y tree
git config --global url.https://github.com/.insteadOf git@github.com:
git config --global advice.detachedHead false

# get install tools
apt-get -qq update && apt-get install -y python3-catkin-tools python3-vcstool python3-pip python-is-python3

# Clear cache -> keep layer size down
rm -rf /var/lib/apt/lists/*