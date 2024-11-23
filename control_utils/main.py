#!/usr/bin/env python3
import numpy as np

from control_utils.leap_hand_utils.dynamixel_client import *
import control_utils.leap_hand_utils.leap_hand_utils as lhu
import time

#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Each of position, velociy and current costs one sample, so you can sample all three at 30 hz or one at 90hz.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more
#http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Zeros_and_Directions_Setup_Guide I belive the black and white figure (not blue motors) is the zero position, and the + is the correct way around.  LEAP Hand in my videos start at zero position and that looks like that figure.

#LEAP hand conventions:
#180 is flat out for the index, middle, ring, fingers, and positive is closing more and more.

"""


########################################################
class LeapNode:
    def __init__(self, motor_nums=20,current_limit=900):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 1500
        self.kI = 0
        # self.kI = 20
        # self.kD = 200
        self.kD = 2000
        # self.kD = 50
        motors = list(range(motor_nums))
        self.len_motors = motor_nums
        self.curr_lim = 350 * np.ones(self.len_motors)
        self.len_fingers = int(self.len_motors / 4)
        # l1 = 350
        l1 = current_limit
        # l1 = 150
        self.curr_lim = [l1] * 8 + [l1, l1, l1, l1 ] *(self.len_fingers-3) + [l1]*4
        # self.curr_lim = 800

        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(len(motors)))
        first_motors = [0, 4, 8, 12, 16]
        self.kp_all = [self.kP, self.kP, self.kP, self.kP] * self.len_fingers
        # self.kd_all = [100, 100, 100, 100] * 5
        self.kd_all = [self.kD, self.kD, self.kD, self.kD] * self.len_fingers
        self.ki_all = [self.kI, self.kI, self.kI, self.kI] * self.len_fingers

        # You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 3000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 3000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', 3000000)
                self.dxl_client.connect()

                # check the control tabel:  https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/
        # Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness
        self.dxl_client.sync_write(motors, np.array(self.kp_all), 84, 2)  # Pgain stiffness
        # self.dxl_client.sync_write([3, 7, 11, 15, 19], np.ones(5) * (self.kP * 0.5), 84,
        #                            2)  # Pgain stiffness for side to side should be a bit less
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)  # Igain
        self.dxl_client.sync_write(motors, np.array(self.ki_all), 82, 2)  # Igain
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write(motors, np.array(self.kd_all), 80, 2)  # Dgain damping
        # self.dxl_client.sync_write([3,7,11,15,19], np.ones(5) * (self.kD * 0.5), 80, 2) # Dgain damping for side to side should be a bit less
        # self.dxl_client.sync_write(first_motors, np.ones(5) * (self.kP * 2), 84,2)  # Pgain stiffness for side to side should be a bit less
        # self.dxl_client.sync_write(first_motors, np.ones(5) * (self.kD * 2), 80, 2)  # Pgain stiffness for side to side should be a bit less

        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite

        self.dxl_client.sync_write(motors, self.curr_lim, 102, 2)

        # self.q0 = self.read_pos()
        # self.dxl_client.write_desired_pos(self.motors, self.q0)

    # Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # allegro compatibility
    def set_allegro(self, pose, vel=None, acc=None):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

        if vel is not None:
            self.dxl_client.write_vel_profile(self.motors, vel)

        if acc is not None:
            self.dxl_client.write_acc_profile(self.motors, acc)

    # Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # read position
    def read_pos(self):
        return self.dxl_client.read_pos()

    # read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    # read current
    def read_cur(self):
        return self.dxl_client.read_cur()


# init the node
def main(**kwargs):
    leap_hand = LeapNode()
    t0 = time.time()
    while True:
        # leap_hand.set_allegro(np.zeros(20))
        leap_hand.set_allegro(np.ones(20) * 0.2)
        print("Position: " + str(leap_hand.read_pos()))
        time.sleep(0.015)


if __name__ == "__main__":
    main()
