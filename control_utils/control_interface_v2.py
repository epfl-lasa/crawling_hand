"""
Control interface for the 2rd version of the crawling hand

use the LEAP hand api to control the Dynamixel motors by ROS noetic, in ubuntu 20.04
control freq: 30Hz

Date: 16th July 2024
Author: Xiao GAO
"""
import copy

import numpy as np
import time
from control_utils.main import LeapNode
from kinematics import hand_sym


class real_hand_v2(LeapNode):
    def __init__(self, path_suffix='', motor_nums=20, finger_num=5, current_limit=900, reversed_motors=None):
        super().__init__(motor_nums=motor_nums, current_limit=current_limit)
        self.n = len(self.motors)
        self.dt = 0.01

        # hand fk
        self.hand_kine = hand_sym.Robot(path_suffix=path_suffix,
                                        finger_num=finger_num, version='v2')  # all frames are wrt the `hand_base`
        q_limit = [[-1, 1]] + [[-np.pi / 2, np.pi / 2]] * 3
        self.q_limit = np.array(q_limit)
        self.q_limit = self.q_limit.T

        self.q_cmd = np.zeros(motor_nums)
        self.reversed_motors = reversed_motors
        if self.reversed_motors is None:
            if motor_nums == 20:
                self.reversed_motors = [0, 4, 12, 16]
            elif motor_nums == 24:
                self.reversed_motors = [12, 16, 20]
            elif motor_nums == 16:
                self.reversed_motors = [0, 4, 12, ]
            else:
                self.reversed_motors = [0, 4,  ]

        self.motor_nums = motor_nums
        self.vel_local = None

        # record current and check if it is out of the output range of the adapter
        self.record_current = []
        self.start_record = False

    def move_to(self, q, vel=None, acc=None):
        """
        send joint cmd directly, be careful. If there is a big difference between q_cmd and self.q, the motor will have a big jump
        use `send_traj` for following a trajectory
        :param q:
        :return:
        """
        assert len(q) == self.n
        q_cmd = copy.deepcopy(q)
        q_cmd[self.reversed_motors] = -q_cmd[self.reversed_motors]  # reverse the joints so that matching the MuJoCo sim
        # q_cmd[8:12] = -q_cmd[8:12]  # the third finger has been reversed.
        # if self.motor_nums == 24:
        #     # q_cmd[20:24] = -q_cmd[20:24]
        #     q_cmd[0:12] = -q_cmd[0:12]

        self.set_allegro(q_cmd, vel=vel, acc=acc)
        self.q_cmd = copy.deepcopy(q)
        if self.start_record:
            self.record_current.append(self.ddq)

    def send_traj(self, q, q_start=None, vel=0.5, dt=None, print_info=False):
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
            self.move_to(q_list[i, :])
            if dt is None:
                pass
            else:
                time.sleep(dt)
        if print_info:
            print('Done')

    @property
    def q(self):
        q0 = self.read_pos() - np.pi
        q0[self.reversed_motors] = -q0[self.reversed_motors]
        q0[8:12] = -q0[8:12]
        if self.motor_nums == 24:
            q0[0:12] = -q0[0:12]
        return q0

    @property
    def dq(self):
        """
        some joints are reversed, as shown in `self.q`
        :return:
        """
        dq = self.read_vel()

        dq[self.reversed_motors] = -dq[self.reversed_motors]
        dq[8:12] = -dq[8:12]
        if self.motor_nums == 24:
            dq[20:24] = -dq[20:24]
        return dq

    @property
    def ddq(self):
        return self.read_cur()

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
