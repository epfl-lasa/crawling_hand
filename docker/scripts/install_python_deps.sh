#!/bin/bash

sudo apt-get update -y
sudo apt-get install python3-tk -y

pip install jupyter
pip install numpy scipy matplotlib numba tqdm
pip install opencv-python

pip install urdf-parser-py quaternion mujoco sympy
pip install pygad==3.2.0

# Clear cache -> keep layer size down
rm -rf /var/lib/apt/lists/*
