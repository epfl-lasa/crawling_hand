"""
control interface for iiwa and allegro hand
keep sending torque command to iiwa
"""


import time

import rospy
import numpy as np
from functools import partial

import tools.rotations as rot
# import kinematics.allegro_hand_sym as allegro
# # from iiwa_tools.srv import GetIK, GetFK
# from trac_ik_python.trac_ik import IK
from urdf_parser_py.urdf import URDF  # need to install it under py3
import kinematics.kdl_parser as kdl_parser

import PyKDL as kdl
# https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_python/, install it by `pip install -e .` under the cond env


from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

import copy
import signal
import sys


class Robot():
    @staticmethod
    def clean_up(signum, frame):
        if not rospy.is_shutdown():
            rospy.signal_shutdown("User requested shutdown")
        sys.exit()

    def __init__(self, optitrack_frame_names=None, position_control=True, calibration=False, camera=False,
                 camera_object_name=None, ):

        rospy.init_node('iiwa_impedance', anonymous=True)

        self.optitrack_frame_names = optitrack_frame_names

        # recalibration should be done if the relative pose between the marker and iiwa base changes
        # self.iiwa_base2m = np.loadtxt(
        #     'description/config/iwwa_link_0_2_iiwabase7_calibration.txt')  # iiwa_link_0 to iiwa_base7
        self.base2world = None
        self.base2world_b_ = True
        if self.optitrack_frame_names is not None:
            self._x_obj = {}

        # for iiwa
        # self.control_mode =
        self.iiwa_bounds = np.array([[-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                                      -2.09439510239, -3.05432619099],
                                     [2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                                      2.09439510239, 3.05432619099]])
        self._iiwa_js_sub = rospy.Subscriber("/iiwa/joint_states", JointState, self._iiwa_joint_state_cb)

        self._q = np.zeros(7)
        self._dq = np.zeros(7)
        self._effort = np.zeros(7)

        if position_control:
            self.control_mode = "position"
            self._iiwa_position_pub = rospy.Publisher("/iiwa/PositionController/command", Float64MultiArray,
                                                      queue_size=10)
            self._sending_torque = False
        else:
            self.control_mode = "torque"
            self._sending_torque = False
            self._torque_cmd = np.zeros(7)
            self._iiwa_torque_pub = rospy.Publisher("/iiwa/TorqueController/command", Float64MultiArray,
                                                    queue_size=10)
            self._joint_kp = np.array([800, 800, 800, 800, 300, 50, 10.]) * 2
            # self._joint_kp = np.array([400, 400, 400, 400, 200, 50, 10.])
            self._joint_kd = np.array([80, 100, 80, 80, 10, 1, 1.])
            self.q_cmd = None
            # self.x_cmd = None
            self._x_cmd = None

            # for torque control in Cartesian space
            # iiwa_cmd_ = rospy.Publisher('/iiwa_impedance_pose', PoseStamped, queue_size=10)
            rospy.Subscriber('/iiwa_impedance_pose', PoseStamped, self.iiwa_impedance_pose_callback)


        self.fk_service = '/iiwa/iiwa_fk_server'
        # self.get_fk = rospy.ServiceProxy(self.fk_service, GetFK)

        self.freq = 200
        self.dt = 1. / self.freq



        # iiwa ik
        self.iiwa_start_link = "iiwa_link_0"
        self.iiwa_end_link = "iiwa_link_ee"

        # self._urdf_str = rospy.get_param('/robot_description')
        # print(self._urdf_str)
        # self._ik_solver = IK(self.iiwa_start_link, self.iiwa_end_link)
        # lower_bound, upper_bound = self._ik_solver.get_joint_limits()
        # # print(lower_bound, upper_bound)
        # self._ik_solver.set_joint_limits(lower_bound, upper_bound)

        # self._iiwa_urdf = URDF.from_xml_string(self._urdf_str)
        self._iiwa_urdf = URDF.from_xml_file('/home/lasa/research/projects/crawling_hand/crawling_hand/descriptions/iiwa_description/urdf/iiwa14.urdf')

        self._iiwa_urdf_tree = kdl_parser.kdl_tree_from_urdf_model(self._iiwa_urdf)
        self._iiwa_urdf_chain = self._iiwa_urdf_tree.getChain(self.iiwa_start_link, self.iiwa_end_link)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self._iiwa_urdf_chain)
        self.jac_calc = kdl.ChainJntToJacSolver(self._iiwa_urdf_chain)

        time.sleep(1)
        signal.signal(signal.SIGINT, Robot.clean_up)

    def _iiwa_joint_state_cb(self, data):
        self._q = np.copy(np.array(data.position))
        self._dq = np.copy(np.array(data.velocity))
        self._effort = np.copy(np.array(data.effort))

    def iiwa_impedance_pose_callback(self, state: PoseStamped):
        pose = np.array([state.pose.position.x, state.pose.position.y, state.pose.position.z,
                         state.pose.orientation.w, state.pose.orientation.x, state.pose.orientation.y,
                         state.pose.orientation.z])
        self._x_cmd = pose
        print(pose)

    def iiwa_impedance(self, pose: np.ndarray, d_pose=None):
        if d_pose is None:
            d_pose = np.zeros(6)
        kp = np.array([300 * 1.5*2, 40 *2 ])
        kd = np.sqrt(kp) * 2
        # kd[1] = 0.1
        # kd = np.sqrt(kp) * 1
        pos_error = pose[:3] - self.x[:3]
        vel_error = d_pose[:3] - self.dx[:3]
        Fx = kp[0] * (pose[:3] - self.x[:3]) + kd[0] * (d_pose[:3] - self.dx[:3])
        q = self.x[3:]  # [w x y z]
        qd = pose[3:]

        if qd[0] < 0:
            qd = -qd

        # d_theta = (quaternion.from_float_array(qd) * (quaternion.from_float_array(q)).conjugate()).log() * 2
        # d_theta = quaternion.as_float_array(d_theta)[1:]
        axis, angle = rot.quat2axisangle(rot.quat_mul(qd, rot.quat_conjugate(q)))
        d_theta = np.array(axis) * angle
        # if np.linalg.norm(d_theta) >1:
        #     print(d_theta)
        Fr = kp[1] * d_theta + kd[1] * (d_pose[3:] - self.dx[3:6])
        F = np.concatenate([Fx, Fr])
        J = self.J
        impedance_acc_des0 = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), F))
        impedance_acc_des1 = J.T @ F

        # Add stiffness and damping in the null space of the the Jacobian
        nominal_qpos = np.zeros(7)
        null_space_damping = 0.2
        null_space_stiffness = 5
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_control = -null_space_damping * self.dq
        null_space_control += -null_space_stiffness * (
                self.q - nominal_qpos)
        tau_null = projection_matrix.dot(null_space_control)
        tau_null_c = np.clip(tau_null, -5, 5)  # set the torque limit for null space control
        impedance_acc_des = impedance_acc_des1 + tau_null_c

        # self.send_torque(impedance_acc_des + self.C)
        self._send_iiwa_torque(impedance_acc_des)

    def _send_iiwa_torque(self, torques: np.ndarray) -> None:

        iiwa_torque_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(torques)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_torque_cmd.layout = layout

        # Fill torques
        iiwa_torque_cmd.data = torques

        self._iiwa_torque_pub.publish(iiwa_torque_cmd)

    def _iiwa_joint_space_impedance(self, qd, d_qd=None):
        """
        directly sending torque
        :param qd:
        :return:
        """
        if d_qd is None:
            d_qd = np.zeros(7)
        error_q = qd - self.q
        if np.max(np.abs(error_q)) > 0.1:
            print("error")
        assert np.max(np.abs(error_q)) < 0.1
        error_dq = d_qd - self.dq

        qacc_des = self._joint_kp * error_q + self._joint_kd * error_dq

        self._send_iiwa_torque(qacc_des)

    def _iiwa_joint_control(self, qd, vel=0.05):
        """
        joint space control by linear interpolation
        :param qd:
        :param vel:
        :return:
        """
        error = self.q - qd
        t = np.max(np.abs(error)) / vel
        NTIME = int(t / self.dt)
        print("Linear interpolation by", NTIME, "joints")
        q_list = np.linspace(self.q, qd, NTIME)
        for i in range(NTIME):
            self._iiwa_joint_space_impedance(q_list[i, :])
            time.sleep(self.dt)
        self.q_cmd = q_list[-1, :]

    def forward_kine(self, q, quat=True, return_jac=True):
        """
        forward kinematics for all fingers
        :param quat: return quaternion or rotation matrix
        :param q: numpy array  (16,) or (8,)
        :return: x:  pose and jacobian
        """
        assert len(q) == 7

        q_ = kdl_parser.joint_to_kdl_jnt_array(q)
        end_frame = kdl.Frame()
        self.fk_solver.JntToCart(q_, end_frame)
        x = np.array([end_frame.p[0], end_frame.p[1], end_frame.p[2]])
        if quat:
            qua = kdl.Rotation(end_frame.M).GetQuaternion()  # Notice that the quaternion is [x y z w]
            qua = np.array([qua[3], qua[0], qua[1], qua[2]])  # [w, x, y, z]
            pose = np.concatenate([x, qua])
        else:
            R = np.array([[end_frame.M[0, 0], end_frame.M[0, 1], end_frame.M[0, 2]],
                          [end_frame.M[1, 0], end_frame.M[1, 1], end_frame.M[1, 2]],
                          [end_frame.M[2, 0], end_frame.M[2, 1], end_frame.M[2, 2]]])
            T = np.concatenate([R, x.reshape(-1, 1)], axis=1)
            T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)

        if return_jac:
            jac = kdl.Jacobian(7)
            self.jac_calc.JntToJac(q_, jac)
            jac_array = kdl_parser.kdl_matrix_to_mat(jac)

        if return_jac:
            return pose, jac_array
        else:
            return pose

    @property
    def q(self):
        return self._q

    @property
    def dq(self):
        return self._dq

    @property
    def effort(self):
        return self._effort

    @property
    def  x(self):
        return self.forward_kine(self.q, return_jac=False)

    @property
    def x_cmd(self):
        return self._x_cmd

    @property
    def J(self):
        x, jac = self.forward_kine(self.q, return_jac=True)
        return jac

    @property
    def dx(self):
        """
            Cartesian velocities of the end-effector frame
            Compute site end-effector Jacobian
        :return: (6, )
        """
        dx = self.J @ self.dq
        return dx.flatten()


if __name__ == "__main__":
    r = Robot(camera=False, position_control=False)

    while np.linalg.norm(r.q) < 1e-5:
        time.sleep(0.1)
    x_cmd = np.copy(r.x)
    print("ready, start torque control")
    while not rospy.is_shutdown():
        if r.x_cmd is not None:
            x_cmd = r.x_cmd
            # print()
        r.iiwa_impedance(x_cmd)
        # print(r.x_cmd)
        time.sleep(0.001)




