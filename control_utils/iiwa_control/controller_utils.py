"""
control interface for iiwa and allegro hand
"""
import time

import rospy
import numpy as np
from functools import partial

import tools.rotations as rot
# import kinematics.allegro_hand_sym as allegro
# from iiwa_tools.srv import GetIK, GetFK
# from trac_ik_python.trac_ik import IK
# there are some issues for installing trac_ik under conda, with system python it works: apt install ros-noetic-trac_ik_python
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
                 camera_object_name=None, path_prefix='', init_node=True):

        if init_node:
            rospy.init_node('iiwa_allegro_controller', anonymous=True)
        # rospy.tim

        self.optitrack_frame_names = optitrack_frame_names

        # recalibration should be done if the relative pose between the marker and iiwa base changes
        # self.iiwa_base2m = np.loadtxt(path_prefix +
        #                               'description/config/iwwa_link_0_2_iiwabase7_calibration.txt')  # iiwa_link_0 to iiwa_base7
        # self.allegro_base2_ee = np.loadtxt(path_prefix +
        #                                    'description/config/allegro_left_base_2_iiwa_link_ee.txt')  # allegro_base to iiwa_link_ee
        self.base2world = None
        self.base2world_b_ = True
        if self.optitrack_frame_names is not None:
            self._x_obj = {}

            # subscriber all object poses from the Optitrack system
            for marker in self.optitrack_frame_names:
                rospy.Subscriber('/vrpn_client_node/' + marker + '/pose', PoseStamped,
                                 partial(self.object_pose_callback, name=marker,
                                         obj=marker in self.optitrack_frame_names))

        # for hand
        # self.joint_cmd_pub = rospy.Publisher('/allegroHand_0/joint_cmd', JointState, queue_size=10)
        # rospy.Subscriber('/allegroHand_0/joint_states', JointState, self._hand_joint_states_callback)
        # self.hand = allegro.Robot(right_hand=False, path_prefix=path_prefix)  # load the left hand kinematics
        self.fingertip_sites = ['index_site', 'middle_site', 'ring_site',
                                'thumb_site']  # These site points are the fingertip (center of semisphere) positions
        self._qh = np.zeros(16)
        self._q = np.zeros(7)
        self._dq = np.zeros(7)
        self._effort = np.zeros(7)
        self.hand_joint_cmd = JointState()
        self.hand_joint_cmd.name = ['joint_0', 'joint_1', 'joint_2', 'joint_3',
                                    # index finger: ad/abduction, extensions
                                    'joint_4', 'joint_5', 'joint_6', 'joint_7',  # middle finger
                                    'joint_8', 'joint_9', 'joint_10', 'joint_11',  # ring finger
                                    'joint_12', 'joint_13', 'joint_14', 'joint_15']  # thumb
        self.hand_joint_cmd.position = []  # 0-3: index, 4-7: middle, 8-11: ring, 12-15: thumb
        self.hand_joint_cmd.velocity = []
        self.hand_joint_cmd.effort = []
        self.hand_bounds = np.array([])

        # for iiwa
        # self.control_mode =
        self.iiwa_bounds = np.array([[-2.96705972839, -2.09439510239, -2.96705972839, -2.09439510239, -2.96705972839,
                                      -2.09439510239, -3.05432619099],
                                     [2.96705972839, 2.09439510239, 2.96705972839, 2.09439510239, 2.96705972839,
                                      2.09439510239, 3.05432619099]])
        self._iiwa_js_sub = rospy.Subscriber("/iiwa/joint_states", JointState, self._iiwa_joint_state_cb)

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
            self._joint_kp = np.array([800, 800, 800, 800, 300, 50, 10.])
            # self._joint_kp = np.array([400, 400, 400, 400, 200, 50, 10.])
            self._joint_kd = np.array([80, 100, 80, 80, 10, 1, 1.])
            self.q_cmd = None
            self.x_cmd = None

            # for torque control in Cartesian space
            self.iiwa_cmd_pub = rospy.Publisher('/iiwa_impedance_pose', PoseStamped, queue_size=10)

        # self.fk_service = '/iiwa/iiwa_fk_server'
        # self.get_fk = rospy.ServiceProxy(self.fk_service, GetFK)

        self.freq = 200
        self.dt = 1. / self.freq
        self._iiwa_home = np.array(
            [-0.32032434, 0.02706913, -0.22881953, -1.42621454, 1.3862661, 0.55966738, 1.79477984 - np.pi])
        self._iiwa_home_pose = np.array(
            [0.47769025, -0.21113556, 0.78239543, 0.70487697, 0.00342266, 0.70931291, 0.0034541])
        self._hand_home = np.zeros(16)
        self._hand_home[12] = 0.7

        # for iiwa base calibration
        # if calibration:
        #     self.marker_name = ['iiwa_base7', 'iiwa_ee_m'] # for calibration, always in world frame

        # iiwa ik
        self.iiwa_start_link = "iiwa_link_0"
        self.iiwa_end_link = "iiwa_link_ee"

        # self._urdf_str = rospy.get_param('/robot_description')
        # print(self._urdf_str)
        # # relax_ik
        # self._ik_solver = IK(self.iiwa_start_link, self.iiwa_end_link, solve_type="distance", timeout=0.005,
        #                      epsilon=5e-4)
        # lower_bound, upper_bound = self._ik_solver.get_joint_limits()
        # # print(lower_bound, upper_bound)
        # self._ik_solver.set_joint_limits(lower_bound, upper_bound)

        # self._iiwa_urdf = URDF.from_xml_string(self._urdf_str)
        self._iiwa_urdf = URDF.from_xml_file(path_prefix + 'descriptions/iiwa_description/urdf/iiwa7_lasa.urdf')
        # self._iiwa_urdf = URDF.from_xml_file(path_prefix + 'descriptions/iiwa_description/urdf/iiwa14.urdf')

        self._iiwa_urdf_tree = kdl_parser.kdl_tree_from_urdf_model(self._iiwa_urdf)
        self._iiwa_urdf_chain = self._iiwa_urdf_tree.getChain(self.iiwa_start_link, self.iiwa_end_link)
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self._iiwa_urdf_chain)

        # self.fk_solver.
        self.jac_calc = kdl.ChainJntToJacSolver(self._iiwa_urdf_chain)
        # kdl.JntSpaceInertiaMatrix

        # For camera
        if camera:
            time.sleep(0.2)
            self.camera_tracking_obj_name = camera_object_name  # todo, for multiple objects
            # from the eye-in-hand calibration, ~/.ros/easy_handeye/*.yaml
            # *_m means the frame from optitrack
            T_base_link2rs_m = np.array(
                [0.0021646149766359063, 0.03432535243479323, -0.024087630091230904, 0.5850475665981398,
                 -0.5919615614203375, 0.34476563541615446, 0.43090950285189444])

            T_camera_link2base_link = np.array([-0.011, 0.012, -0.033, 0.977, 0.026, -0.010, 0.211])
            T_camera_link2rs_m = rot.pose_mul(T_base_link2rs_m, T_camera_link2base_link)
            T_optical2camera_link = np.array([-0.001, 0.015, 0.000, 0.500, -0.502, 0.499, -0.499])
            self.T_optical2rs_m = rot.pose_mul(T_camera_link2rs_m, T_optical2camera_link)
            rospy.Subscriber("objects_pose", PoseArray, self.pose_callback)
            self.obj_pose_in_optical = {}  # put this as the initialization of object pose in icg_ros/config/banana_detector.yaml

        # grasping config
        self.bottle_grasping_xy_offset = np.array([-0.141546, 0.013219])
        self.bottle_grasping_z = 0.17947467
        self.bottle_grasping_quat = np.array([0.705685, 0.001096, 0.708522, 0.001966])

        time.sleep(1)
        signal.signal(signal.SIGINT, Robot.clean_up)
        # rospy.spin()

        # iiwa with aruco marker 1 for the NMI paper experiments
        self.T_m1_2_base = np.array([0.87253061, 1.34477449, 0.0207895, 0.7558567, -0.01277603, 0.00218702, 0.65460877])
        self.T_m3_2_e = np.array([-0.00198612, 0.02612706, 0.0786672, 0.17395611, 0.26438827, -0.6727699, -0.66874418])

        # pose when connect with the crawling hand.
        self.T_m3_2_m2 = np.array(
            [0.11654254, -0.03420449, -0.00340296, 0.73513868, 0.01457556, 0.02488386, 0.67730309])

        self.q_desired = None

    def pose_callback(self, state: PoseArray):

        # print(state.pose.position.z)
        for i in range(len(self.camera_tracking_obj_name)):
            pose = np.array([state.poses[i].position.x, state.poses[i].position.y, state.poses[i].position.z,
                             state.poses[i].orientation.w, state.poses[i].orientation.x, state.poses[i].orientation.y,
                             state.poses[i].orientation.z])
            self.obj_pose_in_optical[
                self.camera_tracking_obj_name[i]] = pose  # obj pose in the camera_color_optical_frame
            # print(pose)
            obj_pose_rs_m = rot.pose_mul(self.T_optical2rs_m, pose)
            self._x_obj[self.camera_tracking_obj_name[i]] = rot.pose_mul(self.x_obj['realsense_m'],
                                                                         obj_pose_rs_m)  # the final pose in allegro base

    def object_pose_callback(self, data: PoseStamped, name='iiwa_base7', obj=False):
        current_object_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        current_object_quat = np.array(
            [data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])
        tmp = np.concatenate([current_object_pos, current_object_quat])  # in Optitrack frame (world)
        self._x_obj[name] = tmp
        # todo, do the calibration
        if self.base2world is not None:
            # update the pose of objects to be manipulated, in base frame (iiwa_link_0)
            self._x_obj[name] = rot.pose_mul(rot.pose_inv(self.base2world), tmp)
        else:
            self._x_obj[name] = tmp  # in Optitrack frame
            if name == 'iiwa_base7' and self.base2world_b_:  # update it only once, and do not move the hand base anymore.
                self.base2world = rot.pose_mul(self._x_obj['iiwa_base7'], self.iiwa_base2m)
                self.base2world_b_ = False

    def _hand_joint_states_callback(self, data):
        # update current joint angles
        self._qh = np.copy(np.array(data.position))

    def _iiwa_joint_state_cb(self, data):
        self._q = np.copy(np.array(data.position))
        self._dq = np.copy(np.array(data.velocity))
        self._effort = np.copy(np.array(data.effort))
        # if not self._sending_torque and self.control_mode == 'torque':
        #     if self.x_cmd is not None:
        #         x = self.x_cmd
        #     else:
        #         x = np.copy(self.x)
        #     # self._iiwa_joint_space_impedance(q)  # keep at the current pose
        #     self._iiwa_impedance(x)  # keep at the current pose

    def send_hand_position(self, joints: np.ndarray) -> None:
        self.hand_joint_cmd.position = list(joints)
        self.joint_cmd_pub.publish(self.hand_joint_cmd)

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

    def kinesthetic_teaching(self):
        assert self.control_mode == 'torque'
        while True and not rospy.is_shutdown():
            self._send_iiwa_torque(self.q)
            time.sleep(self.dt)

    def _send_iiwa_position(self, joints: np.ndarray) -> None:

        iiwa_position_cmd = Float64MultiArray()

        # Fill out layout information
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension())
        layout.data_offset = 0
        layout.dim[0].size = len(joints)
        layout.dim[0].stride = 1
        layout.dim[0].label = "joints"
        iiwa_position_cmd.layout = layout

        # Fill joints
        iiwa_position_cmd.data = joints

        self._iiwa_position_pub.publish(iiwa_position_cmd)

    def iiwa_go_home(self):
        if self.control_mode == 'position':
            self.move_to_joints(self._iiwa_home, vel=[0.1, 1])
        else:
            # print("Joint space torque control by an impedance controller")
            self._sending_torque = True
            # self._iiwa_joint_control(self._iiwa_home)
            self.iiwa_cartesion_impedance_control(self._iiwa_home_pose)
            # self.q_cmd = copy.deepcopy(self.q)
            self.x_cmd = copy.deepcopy(self._iiwa_home_pose)
            self._sending_torque = False
            print("Finish going home")

        # t0 = time.time()
        # while 1:
        #     self._iiwa_impedance(self._iiwa_home_pose)
        #     time.sleep(self.dt)
        #     print(r.x - r._iiwa_home_pose)
        #
        #     t = time.time() - t0
        #     if t > 10:
        #         break

    def iiwa_cmd(self, x):
        p = PoseStamped()
        p.header.frame_id = 'iiwa_link_0'
        p.header.stamp = rospy.Time.now()

        p.pose.position.x = x[0]
        p.pose.position.y = x[1]
        p.pose.position.z = x[2]
        p.pose.orientation.w = x[3]
        p.pose.orientation.x = x[4]
        p.pose.orientation.y = x[5]
        p.pose.orientation.z = x[6]

        self.iiwa_cmd_pub.publish(p)

    def hand_go_home(self):
        self.move_to_joints(self._hand_home)

    def sin_test(self):
        t0 = time.time()
        x0 = np.copy(r.x)
        self._sending_torque = True
        while 1:
            t = time.time() - t0
            xd = np.copy(x0)
            xd[2] += 0.1 * np.sin(2 * np.pi * 0.2 * t)
            self._iiwa_impedance(xd)
            print(self.x - xd)
            time.sleep(self.dt)
            if t > 10:
                break

        self._sending_torque = False

    def sin_test_joint_space(self, i=2, a=0.1):
        t0 = time.time()
        q0 = copy.deepcopy(r.q)
        self._sending_torque = True
        while 1:
            t = time.time() - t0
            qd = np.copy(q0)
            qd[i] += a * np.sin(2 * np.pi * 0.2 * t)
            self._iiwa_joint_space_impedance(qd)
            print(t, self.q[i] - qd[i])
            time.sleep(self.dt)
            if t > 10:
                break

        self._sending_torque = False
        print("Finish test.")

    def iiwa_cartesion_impedance_control(self, xd, vel=0.1, direct=False, h=None, q=None):
        self._sending_torque = True
        if direct:
            pass
            xd_list = xd.reshape(1, -1)
        else:
            xd_list = self.motion_generation(xd, vel=vel, cartesian=True)

        if h is not None: # the crawling hand control
            q_list = q_list = np.linspace(h.q, q, xd_list.shape[0])
        for i in range(xd_list.shape[0]):
            # self._iiwa_impedance(xd_list[i, :])
            self.iiwa_cmd(xd_list[i, :])  # publish to controller_utils2
            if h is not None:
                h.move_to(q_list[i, :])
            time.sleep(self.dt)

        self._sending_torque = False

    def _iiwa_impedance(self, pose: np.ndarray, d_pose=None):
        if d_pose is None:
            d_pose = np.zeros(6)
        kp = np.array([300, 40.])
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
        null_space_damping = 0.1
        null_space_stiffness = 10
        projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(6), J))
        projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
        null_space_control = -null_space_damping * self.dq
        null_space_control += -null_space_stiffness * (
                self.q - nominal_qpos)
        tau_null = projection_matrix.dot(null_space_control)
        tau_null_c = np.clip(tau_null, -5, 5)  # set the torque limit for null space control  todo, normalize it
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
        # assert np.max(np.abs(error_q)) < 0.1
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

    def move_to_joints(self, joints: np.ndarray, vel=[0.2, 1], fix_fingers=None):
        """
        linear interpolation in joint space
        :param joints:
        :param vel:
        :return:
        """
        if fix_fingers is None:
            fix_fingers = [None, None, None, None]
        n = len(joints)
        assert n in [7, 16, 23]
        if n == 7:
            error = self.q - joints
            vel = vel[0]
        elif n == 16:
            error = self.qh - joints
            vel = vel[1]
        else:
            error = self.q_all - joints
            vel = vel[0]
        # print(vel, error)
        t = np.max(np.abs(error)) / vel
        # print(t, self.dt)
        NTIME = int(t / self.dt)

        print("Linear interpolation by", NTIME, "joints")
        if len(joints) == 7:
            q_list = np.linspace(self.q, joints, NTIME)
            for i in range(NTIME):
                self._send_iiwa_position(q_list[i, :])
                time.sleep(self.dt)
        elif len(joints) == 16:
            q_list = np.linspace(self.qh, joints, NTIME)
            for i in range(NTIME):
                q_d = q_list[i, :]
                for j, q_fix in enumerate(fix_fingers):
                    if q_fix is not None:
                        breakpoint()
                        q_d[j * 4: j * 4 + 4] = q_fix
                self.send_hand_position(q_d)
                time.sleep(self.dt)
        else:
            q_list = np.linspace(self.q_all, joints, NTIME)
            for i in range(NTIME):
                self._send_iiwa_position(q_list[i, :7])
                self.send_hand_position(q_list[i, 7:])
                time.sleep(self.dt)
        print('Trajectory has been executed.')

        # while True:
        #
        #     error = self.q - joints
        #    # c = c / (np.abs(error) + 1e-6)  # for each joint should take different c?
        #
        #     dq_desired = - c * error
        #     q_desired = self.q + dq_desired * self.dt
        #     self._send_iiwa_position(q_desired)
        #     time.sleep(self.dt)
        #     if np.linalg.norm(error) < 1e-4:
        #         break

    def trac_ik_solver(self, target_pose: np.ndarray, seed=None):
        if seed is None:
            seed = self.q
        next_joint_positions = self._ik_solver.get_ik(seed,
                                                      target_pose[0],
                                                      target_pose[1],
                                                      target_pose[2],  # X, Y, Z
                                                      target_pose[4],
                                                      target_pose[5],
                                                      target_pose[6],
                                                      target_pose[3])  # QX, QY, QZ, QW
        if next_joint_positions is None:
            print('IK failed')  # very easy to fail...
            sys.exit(1)
        q = np.array(next_joint_positions)
        for i in range(7):
            assert q[i] > self.iiwa_bounds[0, i] and q[i] < self.iiwa_bounds[1, i]

        return np.array(next_joint_positions)

    def move_to_target_cartesian_pose(self, target_pose, vel=[0.1, 1], joint_space_intelporation=True):
        """

        :param target_pose: [x,y,z,qw,qx,qy,qz]
        :return:
        """
        if joint_space_intelporation:
            desired_joints = []
            if type(target_pose) == list:
                for pose in target_pose:
                    desired_joint = self.trac_ik_solver(pose)
                    self.move_to_joints(desired_joint, vel=vel)
                    desired_joints.append(desired_joint)
            else:
                desired_joints = self.trac_ik_solver(target_pose)
                self.q_desired = desired_joints
                assert len(desired_joints) == 7
                self.move_to_joints(desired_joints, vel=vel)
        else:
            if type(target_pose) == list:
                joint_lists = []
                for pose in target_pose:
                    joint_list = self.motion_generation(pose, vel=vel[0], intepolation='linear',
                                                        cartesian=False)  # (n, 7)
                    joint_lists.append(joint_list)
                joint_list = np.vstack(joint_lists)
            else:
                # do the interpolation in Cartesian space
                joint_list = self.motion_generation(target_pose, vel=vel[0], intepolation='linear',
                                                    cartesian=False)  # (n, 7)
            self.iiwa_move_to_joints(joint_list)

    def motion_generation(self, poses, vel=0.05, intepolation='linear', cartesian=False):
        # poses : (n,7) array, n: num of viapoints. [position, quaternion]
        if len(poses.shape) == 1:
            poses = poses.reshape(1, -1)
        poses = np.concatenate([self.x.reshape(1, -1), poses], axis=0)  # add current points

        keypoints_num = poses.shape[0]

        path_length = 0
        for i in range(keypoints_num - 1):
            path_length += np.linalg.norm(poses[i, :3] - poses[i + 1, :3])
        path_time = path_length / vel

        joint_seed = self.q
        joint_list = []
        x_list = []
        for i in range(keypoints_num - 1):
            path_i = np.linalg.norm(poses[i, :3] - poses[i + 1, :3])
            # print(path_i)
            sample_num = int(path_i / vel / self.dt + 1)
            if sample_num < 300:
                sample_num = 300

            if intepolation == 'linear':
                pos = np.concatenate((np.linspace(poses[i, 0], poses[i + 1, 0], num=sample_num).reshape(-1, 1),
                                      np.linspace(poses[i, 1], poses[i + 1, 1], num=sample_num).reshape(-1, 1),
                                      np.linspace(poses[i, 2], poses[i + 1, 2], num=sample_num).reshape(-1, 1)),
                                     axis=1)  # print
            ori = rot.slerp(poses[i, 3:], poses[i + 1, 3:],
                            np.array(range(sample_num)) / (sample_num - 1))
            x = np.concatenate([pos, ori], axis=1)
            x_list.append(x)
            if not cartesian:
                for j in range(sample_num):
                    target_x = np.concatenate((pos[j, :], ori[j, :]))
                    desired_joints = self.trac_ik_solver(target_x, seed=joint_seed)
                    joint_list.append(desired_joints)
                    joint_seed = desired_joints

        if cartesian:
            return np.vstack(np.vstack(x_list))
        else:
            return np.vstack(joint_list)

    def iiwa_move_to_joints(self, joints):
        # need to check the smoothness todo
        # error = (np.linalg.norm(joints[1:, :] - joints[0:-1, :], axis=1))
        for j in range(joints.shape[0]):
            self._send_iiwa_position(joints[j, :])
            rospy.sleep(self.dt)

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

    def iiwa_joint_space_PD(self, q: np.ndarray):
        """
             \tau = M(q) ( \ddq + kp e + kd \dot{e} ) + Coriolis + gravity
        :param q: the direct goal joint position
        :return:
        """
        error_q = q - self.q
        error_dq = 0 - self.dq
        kp = np.array([800, 800, 800, 800, 330, 160, 130.])
        # kd = np.array([80, 100, 80, 80, 10, 1, 1.])
        # kd = np.zeros(7)
        kd = np.array([800 / 15., 800 / 15., 800 / 15., 800 / 15., 30, 19, 15.])

        tau = self.M @ (kp * error_q + kd * error_dq) + self.C
        # tau = (kp * error_q + kd * error_dq) + self.C
        # breakpoint()
        # self.send_torque(tau)
        return tau

    def full_joint_space_control(self, q, qh=None):

        if qh is not None:
            q = np.concatenate([q, qh])

        tau = self.iiwa_joint_space_PD(q[:7])
        tau_hand = self.hand_move_torque(q[7:23], kh_scale=[0.2, 0.2, 0.2, 0.2])
        self.send_torque(np.concatenate([tau, tau_hand]))

    def run(self):
        while 1:
            self.send_torque(r.C_)
            time.sleep(0.002)

    def iiwa_step_test(self, i=6, dq=0.2):
        """

        :param i: index of iiwa joints, to determine the kp, kd for each joint
        :param dq: offset for the step response
        :return:
        """

        q0 = self.q
        q1 = np.copy(q0)
        qh0 = np.copy(self.qh)
        q1[i] += dq
        q_record = []
        t0 = time.time()
        first = True
        while 1:
            t1 = time.time() - t0

            self.full_joint_space_control(q1, qh0)
            # time.sleep(0.002)
            error = q1[i] - self.q[i]
            if np.abs(error) < dq * 0.01 and first:
                print(t1, error)  # show the rise time
                first = False
            q_record.append(np.array([t1, error]))
            if t1 > 10:
                break

        q_record = np.vstack(q_record)

        # plot the figure of step response, try to avoid overshooting
        plt.plot(q_record[:, 0], q_record[:, 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Error (rad)')
        plt.title('iiwa joint ' + str(i))
        plt.xlim([0, np.max(q_record[:, 0])])
        plt.ylim([None, np.max(q_record[:, 1])])
        plt.show()

    def iiwa_joint_space_reaching(self, q, qh=None, vel=[1., 1], coupling=False):
        """
         reaching motion in joint space, with a linear interpolation
        :param coupling: if coupling the motion for iiwa and hand? if not, the iiwa will reach the goal and then the hand moves
        :param q: goal point for iiwa
        :param qh:  for hand
        :param vel: joint velocity for iiwa and hand
        :return:
        """

        error = np.max(np.abs(q - self.q))
        nums = int(error / vel[0] / 0.002)
        q_list = np.linspace(self.q, q, num=nums)
        if qh is None or not coupling:
            qh0 = np.copy(self.qh)
            qh_list = np.vstack([qh0] * nums)
        else:
            qh_list = np.linspace(self.qh, qh, num=nums)

        for i in range(nums):
            self.full_joint_space_control(q_list[i, :], qh_list[i, :])
            # time.sleep(0.002)

        if not coupling:  # move the arm first, then move the hand
            assert qh is not None
            error_h = np.max(np.abs(qh - self.qh))
            nums_h = int(error_h / vel[1] / 0.002)
            qh_list = np.linspace(self.qh, qh, num=nums_h)
            for i in range(nums_h):
                self.full_joint_space_control(q, qh_list[i, :])  # iiwa keeps static, move the hand only

        # print(q - r.q)
        # print(qh - r.qh)

    def iiwa_joint_space_test(self, i=0, t=10):

        t0 = time.time()
        f = 0.1
        q0 = self.q
        qh0 = self.qh
        error_mean = []
        while time.time() - t0 < t:
            t_now = time.time() - t0
            q1 = np.copy(q0)
            q1[i] += 0.2 * np.sin(2 * np.pi * f * t_now)
            self.full_joint_space_control(q1, qh0)
            time.sleep(0.002)
            print(q1[i] - self.q[i])
            error_mean.append(q1[i] - self.q[i])

        error_mean = np.abs(error_mean)
        print("mean error", np.mean(error_mean), np.std(error_mean))

    @property
    def xh(self) -> list:
        # fingertip (center of semisphere) positions
        return self.hand.forward_kine(self.qh)

    @property
    def x_obj(self) -> dict:
        # get object poses in hand base frame
        return copy.deepcopy(self._x_obj)

    @property
    def q_all(self):
        return np.concatenate([self.q, self.qh])

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
    def qh(self):
        return self._qh

    @property
    def x(self):
        return self.forward_kine(self.q, return_jac=False)

    @property
    def J(self):
        x, jac = self.forward_kine(self.q, return_jac=True)
        return jac

    @property
    def Jh(self):
        return self.hand.get_jac(self.qh)

    @property
    def dx(self):
        """
            Cartesian velocities of the end-effector frame
            Compute site end-effector Jacobian
        :return: (6, )
        """
        dx = self.J @ self.dq
        return dx.flatten()

    # @property
    # def x(self, q=None):
    #     """
    #     Forward kinematics of iiwa robot
    #     :param q:
    #     :return:
    #     """
    #     seed = Float64MultiArray()
    #     seed.layout = MultiArrayLayout()
    #     seed.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
    #     seed.layout.dim[0].size = 1
    #     seed.layout.dim[1].size = 7
    #     if q is None:
    #         seed.data = self.q
    #     else:
    #         seed.data = q
    #     resp1 = self.get_fk(joints=seed)
    #     sol_pose = resp1.poses[0]
    #     x = np.array([sol_pose.position.x, sol_pose.position.y, sol_pose.position.z])
    #     q = np.array([sol_pose.orientation.w, sol_pose.orientation.x, sol_pose.orientation.y, sol_pose.orientation.z])
    #     return np.concatenate([x, q])


if __name__ == "__main__":
    # r = Robot()
    # time.sleep(0.3)
    # print(r.q)
    # q0 = np.copy(r.q)
    # q0[-2] = -1
    # r.move_to_joints(q0)

    # r = Robot(optitrack_frame_names=['iiwa_base7', 'iiwa_ee_m'], calibration=True)
    # time.sleep(0.3)
    # target_pose = np.copy(r.x)
    #
    # target_pose[2] -= 0.05
    # q = r.trac_ik_solver(target_pose)
    # # error = r - r.q
    # r.move_to_target_cartesian_pose(target_pose)
    # print(r.q - q)
    # r.iiwa_go_home()

    # r = Robot(camera=True, optitrack_frame_names=['iiwa_base7', 'realsense_m'], camera_object_name='bottle')
    # x0 = np.copy(r.x)
    # x0[0] = 0.694
    # r.move_to_target_cartesian_pose(x0)
    #
    # time.sleep(0.2)
    # while  not rospy.is_shutdown():
    #
    #     print(r.x_obj)
    #     time.sleep(1)

    # impedance control test
    r = Robot(camera=True, optitrack_frame_names=['iiwa_base7', 'realsense_m'],
              camera_object_name=['cross_part', 'bottle'], position_control=False)

    # r.iiwa_go_home()
    # time.sleep(0.2)
    # r.sin_test()
    r.sin_test_joint_space(i=6, a=0.1)
    # r.iiwa_go_home()
    # while np.linalg.norm(r.q) < 1e-5:
    #     time.sleep(0.1)
    # print("ready")
    #
    # r.iiwa_go_home()
    # r.iiwa
