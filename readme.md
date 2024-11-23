# Crawling Hand Project


## Instructions
- This repo is based on [MuJoCo 2.3.6](https://github.com/deepmind/mujoco)
- Create a new Conda env with Python>= 3.8
- Install mujoco by `pip installl mujoco`


## Visulization
- In mujoco conda env, `python -m mujoco.viewer --mjcf=five_finger_hand_ssss.xml`
- With ROS, `roslaunch urdf_tutorial display.launch model:=URDF_finger_ssss.urdf`


## Explanation of code
 - `gait_test/GA_crawling_full_test_new.py`, the optimal structures achieved by optimizing 
the link lengths and finger placements for grasping one, two, or three objects, respectively.
 - `*_replay_*.py` replay the crawling in simulation.



## Notes
 - Under folder `singer_finger/`, `URDF_finger_xxxx.urdf `is the urdf file for a single finger.
 - For six fingers, see `descriptions/six_fingers_llll.xml`, which makes 6 copies of the single finger.
- To use [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics), modify the function `build_chain_from_mjcf`
in `mjcf.py` by `m = mujoco.MjModel.from_xml_path(data)` (line 75)
- Load the model from .xml by 
```
xml_path = 'descriptions/five_finger_hand_ssss.xml'
chain = pk.build_chain_from_mjcf(xml_path, 'hand')
```


## Real robot control (with ROS)
- install ROS2. I use [ros-iron](https://docs.ros.org/en/iron/Installation/Ubuntu-Install-Debians.html)
 - build from source of [dynamixel](https://github.com/dynamixel-community/dynamixel_hardware) with ROS2. 
 - make sure that we can connect the control all motors in `Dynamixel Wizard`. `Scan` and enable `torque` control to test each motor.
 - `source ~/research/lasa/open_source/crawling_robot/dynamixel_new/install/setup.bash` 
 - launch the robot `ros2 launch pantilt_bot_description pantilt_bot.launch.py`. The `usb_port, baud_rate, id` in `ros2_control.xacro` should be the same as `Dynamixel Wizard`
 - to switch controllers, `ros2 control switch_controllers --activate joint_state_broadcaster --activate joint_trajectory_controller --deactivate velocity_controller`
 - run `control_interface_real_robot.py` for an example.
 - For velocity control `ros2 control switch_controllers --deactivate joint_trajectory_controller`

## Dynamixel control (without ROS)
- Current-based position control https://github.com/leap-hand/LEAP_Hand_API  8ms for reading position only and sending position commands
- https://github.com/AlanSunHR/DynamixelController has full access to all parameters of motors.



## KDL
`conda install -c conda-forge python-orocos-kdl` for PyKDL from https://github.com/orocos/orocos_kinematics_dynamics \
`pip install urdf-parser-py`\
`conda install -c conda-forge matplotlib`\
`conda install -c conda-forge quaternion`  `conda install numpy scipy numba`

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
