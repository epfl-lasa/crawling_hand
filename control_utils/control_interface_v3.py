"""
Control interface for the 2nd version of the crawling hand

use the LEAP hand api to control the Dynamixel motors by ROS noetic, in ubuntu 20.04
control freq: 100 Hz

Date: 16th July 2024
Author: Xiao GAO
"""
import copy

import numpy as np
import time
from control_utils.main import LeapNode
from kinematics import hand_sym
import control_utils.dynamixel_controller_pkg.dynamixel_controller as dc
import control_utils.dynamixel_controller_pkg.dynamixel_models as dm


class real_hand_v2():
    def __init__(self, path_suffix='', motor_nums=20, finger_num=5, current_limit=900, reversed_motors=None):
        """

        :param path_suffix:
        :param motor_nums:
        :param finger_num:
        :param current_limit: in mA, set the soft limit
        :param reversed_motors:  reversed joints for sending command and receiving command
        """
        if type(motor_nums) == int:
            self.motors = [dm.XM430W210(i) for i in range(motor_nums)]
        else:
            assert type(motor_nums) == list
            self.motors = [dm.XM430W210(i) for i in motor_nums]

        try:
            self.h = dc.DynamixelController("/dev/ttyUSB0", self.motors)
            self.h.activate_controller()
        except Exception:
            try:
                self.h = dc.DynamixelController("/dev/ttyUSB1", self.motors)
                self.h.activate_controller()
            except Exception:
                self.h = dc.DynamixelController("/dev/ttyUSB2", self.motors)
                self.h.activate_controller()

        self.h.torque_off()

        # dynamixel_controller.set_operating_mode_all("position_control")
        self.h.set_operating_mode_all("current_based_position_control")
        # h.set_operating_mode_all("position_control")

        self.h.torque_on()

        # setup KP KD
        self.n = len(self.motors)
        self.h.set_position_p_gain([1500] * self.n)
        self.h.set_position_d_gain([2000] * self.n)

        current_limit = np.round(current_limit * np.ones(self.n)).astype(int)
        self.h.set_goal_current(current_limit)

        self.h.set_return_delay_time(np.zeros(self.n))

        self._q, self._dq, self._ddq, self._pwm = None, None, None, None

        #
        self.dt = 0.01

        # hand fk
        self.hand_kine = hand_sym.Robot(path_suffix=path_suffix,
                                        finger_num=finger_num, version='v2')  # all frames are wrt the `hand_base`
        q_limit = [[-1, 1]] + [[-np.pi / 2, np.pi / 2]] * 3
        self.q_limit = np.array(q_limit)
        self.q_limit = self.q_limit.T

        self.q_cmd = np.zeros(self.n)
        self.reversed_motors = reversed_motors
        if self.reversed_motors is None:
            if self.n == 20:
                self.reversed_motors = [0, 4, 12, 16]
            elif self.n == 24:
                self.reversed_motors = [12, 16, 20]
            elif self.n == 16:
                self.reversed_motors = [0, 4, 12, ]
            else:
                self.reversed_motors = [0, 4, ]

        # self.motor_nums = motor_nums
        self.vel_local = None

        # record current and check if it is out of the output range of the adapter
        self.record_current = []
        self.start_record = False
        self.update_motor_info()
        self.move_to(self.q)

    def move_to(self, q, vel=None, acc=None, update=False):
        """
        send joint cmd directly, be careful. If there is a big difference between q_cmd and self.q, the motor will have a big jump
        use `send_traj` for following a trajectory
        :param q:
        :return:
        """
        assert len(q) == self.n
        q_cmd = copy.deepcopy(q)
        q_cmd[self.reversed_motors] = -q_cmd[self.reversed_motors]  # reverse the joints so that matching the MuJoCo sim

        self.h.set_goal_position_rad(q_cmd)
        self.q_cmd = copy.deepcopy(q)
        if self.start_record:
            self.record_current.append(self.ddq)

        if update:
            self.update_motor_info()

    def send_traj(self, q, q_start=None, vel=0.5, dt=None, print_info=False, update=False):
        """
        linear trajectory interpolation
        :param q:
        :param vel:
        :return:
        """
        if q_start is None:
            q_start = self.q_cmd
        error = self.q - q
        t = np.max(np.abs(error)) / vel
        NTIME = int(t / self.dt)
        if print_info:
            print("Linear interpolation by", NTIME, "joints")
        q_list = np.linspace(q_start, q, NTIME)
        for i in range(NTIME):
            self.move_to(q_list[i, :], update=update)
            if dt is None:
                pass
            else:
                time.sleep(dt)
        if print_info:
            print('Done')
        # return q_list

    def update_motor_info(self):

        self._q, self._dq, self._ddq, self._pwm = self.h.read_info_with_unit(fast_read=True, retry=False)  #
        # position_list, velocity_list, current_list, pwm_list

    @property
    def q(self):
        q0 = copy.deepcopy(self._q)
        q0[self.reversed_motors] = -q0[self.reversed_motors]
        return q0

    @property
    def dq(self):
        """
        some joints are reversed, as shown in `self.q`
        :return:
        """
        dq = copy.deepcopy(self._dq)

        dq[self.reversed_motors] = -dq[self.reversed_motors]
        return dq

    @property
    def ddq(self):
        return self._ddq

    @property
    def pwm(self):
        return self._pwm

    def xh_local(self, q=None) -> list:
        """
        get the fingertip positions, wrt the hand_base
        :return:  [(7,), (7,),...] a list of poses(position and quaternion)
        """
        if q is None:
            q = self.q
        return self.hand_kine.forward_kine(q)

    def jac_local(self, q=None) -> list:
        """
        get the jacobian of fingertips, wrt the hand_base
        :return: jacobian,  [(3,4), (3,4)...]
        """
        if q is None:
            q = self.q
        return self.hand_kine.get_jac(q)


if __name__ == '__main__':
    ### contection test
    h = real_hand_v2(path_suffix='../', finger_num=6, motor_nums=24)
    print('Joint position', h.q)
    print('Joint vel', h.dq)
    print('Joint current', h.ddq)

    ### fk test
    print('fingertip poses', h.xh_local())

    ### ik test
    print("############ik test")
    x_cmd_0 = copy.deepcopy(h.xh_local())
    x_cmd_0[0][2] += 0.01
    q_desired = h.hand_kine.ik(x_cmd_0, q_init=h.q)
    print(q_desired - h.q)
    h.send_traj(q_desired)

    # time.sleep(10)
