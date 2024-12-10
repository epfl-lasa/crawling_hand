# Crawling Hand Project

## System requirements
 - Ubuntu 20.04 with `ROS-noetic`

# Installation guide
- Create a `conda` environment with `python==3.8.18`, `pygad=3.2.0`
- Install `Jupyter notebook`
- Install mujoco by `pip install mujoco` under the `conda` environment. The MuJoCo version is tested on `3.0.0`
- Several minutes should be enough for a normal computer.

### Kinematics-related packages
- `conda install -c conda-forge python-orocos-kdl` for PyKDL from https://github.com/orocos/orocos_kinematics_dynamics \
- `pip install urdf-parser-py`\
- `conda install -c conda-forge matplotlib`\
- `conda install -c conda-forge quaternion`  `conda install numpy scipy numba`

# Visulization
- Run `crawling_and_grasp_test04.ipynb`, a robotic hand with 5 fingers will be loaded. The robot can crawl to an object and pick it up.
- In mujoco conda env, `python -m mujoco.viewer --mjcf=five_finger_hand_ssss.xml`
- With ROS, `roslaunch urdf_tutorial display.launch model:=URDF_finger_ssss.urdf`


# Explanation of code
 - `gait_test/GA_crawling_full_test_new.py`, the optimal structures achieved by optimizing 
the link lengths and finger placements for grasping one, two, or three objects, respectively.
 - `*_replay_*.py` replay the crawling in simulation.


# Notes
 - Under folder `singer_finger/`, `URDF_finger_xxxx.urdf `is the urdf file for a single finger.
 - For six fingers, see `descriptions/six_fingers_llll.xml`, which makes 6 copies of the single finger.
- To use [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics), modify the function `build_chain_from_mjcf`
in `mjcf.py` by `m = mujoco.MjModel.from_xml_path(data)` (line 75)
- Load the model from .xml by 
```
xml_path = 'descriptions/five_finger_hand_ssss.xml'
chain = pk.build_chain_from_mjcf(xml_path, 'hand')
```

## Dynamixel motor control (without ROS)
- Current-based position control https://github.com/leap-hand/LEAP_Hand_API  8ms for reading position only and sending position commands
- https://github.com/AlanSunHR/DynamixelController has full access to all parameters of motors.
- A wrapper is written in the `control_utils/control_interface_v3.py`


## Grasping stability in MuJoCo
The following parameters are important:
```   <option noslip_iterations="5"> </option>
    <option>
        <flag gravity="enable" multiccd="enable" />
    </option>
 <option cone="elliptic" impratio="10"/>
```
we also need to set the friction coefficients

## Steps to connect motors
- Give access to USB port `/dev/ttyUSB0` and `/dev/ttyUSB0` by `sudo chmod -R 777 /dev/ttyUSB0`
- Open Dynamixel Wizard and scan motors
- Set the correct USB port number and baudrate
- run `detach_mechanism/key_command_tester.py` to control the attachment/detachment mechanism
