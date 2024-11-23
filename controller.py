import time

import numpy as np
import mujoco
from mujoco import viewer
import os
# import cvxpy as cp
import trimesh

import tools.rotations as rot
import kinematics.hand_sym as hand_sym
import copy
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Robot(object):
    def __init__(self, xml_path, auto_sync=True,
                 q0=None, finger_num=5, dof=4, obj_names=[], path_suffix='', v='v1'):
        self.m = mujoco.MjModel.from_xml_path(path_suffix + xml_path)
        self.d = mujoco.MjData(self.m)
        self.view = viewer.launch_passive(self.m, self.d)
        self.auto_sync = auto_sync
        self.obj_names = obj_names

        self.finger_num = finger_num
        self.dof = dof
        self.n = finger_num * dof
        self.q_ref = np.zeros(self.n)
        if q0 is None:
            self.q0 = np.zeros(self.n)
        else:
            self.q0 = q0
            if len(q0)==7+ self.n:
                self.q0 = q0[7:]


        # hand fk
        self.hand_kine = hand_sym.Robot(path_suffix=path_suffix, finger_num=finger_num,
                                        version=v)  # all frames are wrt the `hand_base`
        q_limit = [[-1, 1]] + [[-np.pi / 2, np.pi / 2]] * (dof - 1)
        self.q_limit = np.array(q_limit)
        self.q_limit = self.q_limit.T

        self.path = 'data_records/finger_' + str(finger_num) + '_dof_' + str(dof)
        file_name = self.path + '_standup_pos.npy'
        if os.path.isfile(file_name):
            self.standup_pos = np.array(np.load(file_name))
            # self.standup_pos[8+7:12+7] = np.zeros(4)
            self.standup_pos[8 + 7:12 + 7] = np.array([0, -np.pi / 2 + 0.1, -0.8, - 0.8])
            self.standup_pos[4 * 4 + 7:4 * 4 + 1 + 7] = -1
            # self.standup_pos[3:7] = np.array([0, 1, 0, 0])
        else:
            print('Need to generate the standup position')
            self.standup_pos = np.zeros(self.n + 7)
            self.standup_pos[3:7] = np.array([0, 1, 0, 0])

        if q0 is not None:
            self.standup_pos = q0
        # for contact
        self.finger_name = ['finger_' + str(i + 1) for i in range(self.finger_num)]
        self.finger_tips_name = ['distal_' + str(i + 1) for i in range(self.finger_num)]
        self.finger_tips_site_name = ['finger_' + str(i + 1) + '_tip' for i in range(self.finger_num)]
        self.floor_name = 'floor'
        self._tip_force = {}
        for i in self.finger_tips_name:
            self._tip_force[i] = np.zeros(3)

        self.modify_joint(self.standup_pos)  # set the initial joint positions
        self.step()
        self.sync()
        self.dt = 0.002

        # self.viewer_setup()

    def warmup(self, t=None, q=None):
        t0 = time.time()
        if t is None:
            t = 2
        if q is None:
            q = self.standup_pos[7:]
        while 1:
            self.joint_computed_torque_control(q)
            time.sleep(0.002)
            if time.time() - t0 > t:
                break

    def step(self):
        mujoco.mj_step(self.m, self.d)  # run one-step dynamics simulation

    def modify_joint(self, joints: np.ndarray) -> None:
        """
        :param joints: (7,) or (20,) or (27,), modify joints for iiwa or/and allegro hand
        :return:
        """
        if len(joints) == 7:
            self.d.qpos[:7] = joints
        if len(joints) == self.n:
            self.d.qpos[7: 7 + self.n] = joints
        elif len(joints) == self.n + 7:
            self.d.qpos[: 7 + self.n] = joints
        else:
            pass

        self.step()
        self.sync()

    def joint_limits(self, q):
        q_tmp = q.reshape(self.finger_num, self.dof)
        q_tmp = np.clip(q_tmp, self.q_limit[0, :], self.q_limit[1, :])

        return q_tmp.flatten()

    def sync(self):
        if self.view is not None:
            self.view.sync()
            self._update_contact_force()

    def modify_obj_pose(self, obj_name: str, pose: np.ndarray) -> None:
        """

        :param obj_name: the name of the object, from self.obj_names
        :param pose:   (7,) or (3,), the pose/position command
        :return: None
        """
        assert obj_name in self.obj_names
        start_index = self.obj_names.index(obj_name) * 7 + 7 + self.n
        len_pose = len(pose)
        assert len_pose in [3, 7]

        self.d.qpos[start_index: start_index + len_pose] = pose
        self.step()
        self.sync()

    def reset(self, xq=None, t=None):
        """
        reset all states of the robot
        # """
        if xq is None:
            xq = self.standup_pos
        self.modify_joint(xq)
        self.warmup(t=t, q=xq[7:])

    def send_torque(self, torque, torque_limit=1.):
        """
        control joints by torque and send to mujoco, then run a step.
        input the joint control torque
        Here I imply that the joints with id from 0 to n are the actuators.
        so, please put robot xml before the object xml.
        todo, use joint_name2id to avoid this problem
        :param torque:  (n, ) numpy array
        :return:
        """
        torque = np.clip(torque, -torque_limit, torque_limit)
        self.d.ctrl[:len(torque)] = torque
        mujoco.mj_step(self.m, self.d)

        if self.auto_sync:
            self.sync()

    def finger_ik(self):
        pass

    def Cartesian_space_impedance_control(self, positions: list, d_positions=None):
        """
        ############ the precision is not good
        Given the reference traj in cartesian space, control the finger/leg to track it.
        The orientation is ignored
        1st task, track the position
        2nd task, close the standup joints
        :param d_positions:
        :param positions, (15, ) position of fingertips
        :return:
        """
        if d_positions is None:
            d_positions = [np.zeros(3)] * self.finger_num

        kp = 20
        kd = np.sqrt(kp) * 0.1

        nominal_qpos = self.standup_pos[7:]
        null_space_damping = 0.1 * 1
        null_space_damping = 0.1 * 0
        null_space_stiffness = 1
        null_space_stiffness = 1 * 0
        tau = []
        for i in range(self.finger_num):  # for each finger
            xh = self.xh_local[i][:3]  # positions of fingertips, in the hand base frame
            dq = self.dq[i * self.dof: i * self.dof + self.dof]
            q = self.q[i * self.dof: i * self.dof + self.dof]
            dxh = self.dxh[i][:3]
            J = self.jac_local
            J = J[i][:3, :]  # only translational jacobian (3, 4)
            Fx = kp * (positions[i] - xh) + kd * (d_positions[i] - dxh)  # (3,)
            impedance_acc_des1 = J.T @ Fx  # (4, )

            # Add stiffness and damping in the null space of the Jacobian, to make the joints close to initial one
            projection_matrix = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-10 * np.eye(3), J))  # (4, 4)
            projection_matrix = np.eye(projection_matrix.shape[0]) - projection_matrix
            null_space_control = -null_space_damping * dq - null_space_stiffness * (
                    q - nominal_qpos[i * self.dof: i * self.dof + self.dof])
            tau_null = projection_matrix.dot(null_space_control)  # (4, )
            tau.append(impedance_acc_des1 + tau_null)
        tau = np.hstack(tau) + self.C
        # tau = np.hstack(tau)

        self.send_torque(tau)

    def Cartesian_space_cmd_each(self, i: int, position: np.ndarray, d_positions=None):
        """

        :param i: i-th leg
        :param position:
        :param d_positions:
        :return:tau,  (4, )     control torque for the i-th leg
        """
        if d_positions is None:
            dq = np.zeros(4)  # (4, )
        else:
            dq = np.linalg.pinv(self.jac_local[i]) @ d_positions
        positions = [[] for i in range(5)]
        positions[i] = position
        q_desired = self.hand_kine.ik(positions, q_init=self.q, k=i)  # (4,)
        if q_desired is None:
            print('IK failed', i)
            q_desired = np.copy(self.q[i * 4:i * 4 + 4])

        kp = 20000 * 2
        kd = 200

        # computed torque control:  M(kp e + kd \dot{e}) + C + G
        tau = (self.M[i * 4:i * 4 + 4, i * 4:i * 4 + 4] @ (
                kp * (q_desired - self.q[i * 4:i * 4 + 4]) + kd * (dq - self.dq[i * 4:i * 4 + 4]))
               + self.C[i * 4:i * 4 + 4])

        return tau

    def Cartesian_space_cmd(self, positions: list, d_positions=None, return_q=False, return_tau=False):
        """


        :param d_positions: vel at fingertips
        :param positions: a list of positions
        :param return_tau:
        :return:
        """
        if d_positions is None:
            dq = None
        else:
            dq = [np.linalg.pinv(self.jac_local[i]) @ d_positions[i] for i in range(self.finger_num)]
            dq = np.hstack(dq)
        q_desired = self.hand_kine.ik(positions, q_init=self.q)  # (20,)
        if q_desired is None:
            print('IK failed')
            q_desired = np.copy(self.q)
        tau = self.joint_computed_torque_control(q_desired, dq=dq, return_tau=return_tau)
        if return_q and return_tau:
            return q_desired, tau
        if return_q:
            return q_desired
        if return_tau:
            return tau

    def sin_test_cartesian_space(self, i=0):
        """

        :param i: i-th finger
        :return:
        """
        x_record = []
        t0 = time.time()
        f = 0.2
        A = 0.005 * 1
        j = 2
        x0 = self.xh_local.copy()
        while 1:
            t = time.time() - t0
            x1 = copy.deepcopy(x0)
            dx1 = [np.zeros(3) for i in range(self.finger_num)]
            x1[i][j] += A * np.sin(2 * np.pi * f * t)
            dx1[i][j] = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)

            self.Cartesian_space_cmd(x1, d_positions=dx1)
            time.sleep(0.002)
            error = x1[i][j] - self.xh_local[i][j]
            # print(t, error)
            x_record.append(np.array([t, x1[i][j], self.xh_local[i][j], error]))
            if t > 10:
                break

        x_record = np.vstack(x_record)
        error = np.abs(x_record[:, 3])
        print('Error:', np.mean(error), np.std(error))

        plt.plot(x_record[:, 0], x_record[:, 1])
        plt.plot(x_record[:, 0], x_record[:, 2])
        plt.xlabel('Time (s)')
        plt.ylabel('Ref (rad)')
        plt.title(str(j) + '-axis tracking for finger ' + str(i))
        plt.xlim([0, np.max(x_record[:, 0])])
        # plt.ylim([None, np.max(q_record[:, 1])])
        plt.show()

    def sin_test_joint_space(self, i=0):

        q_record = []
        t0 = time.time()
        f = 0.2
        A = np.pi / 2 * 0.5
        q0 = np.copy(self.q)
        while 1:
            t = time.time() - t0
            q1 = np.copy(q0)
            dq1 = np.zeros(self.n)
            q1[i] += A * np.sin(2 * np.pi * f * t)
            dq1[i] = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
            self.joint_computed_torque_control(q1, dq=dq1)
            time.sleep(0.002)
            error = q1[i] - self.q[i]
            # print(t, error)
            q_record.append(np.array([t, q1[i], self.q[i], error]))
            if t > 10:
                break

        q_record = np.vstack(q_record)
        error = np.abs(q_record[:, 3])
        print('Error:', np.mean(error), np.std(error))

        plt.plot(q_record[:, 0], q_record[:, 1])
        plt.plot(q_record[:, 0], q_record[:, 2])
        plt.xlabel('Time (s)')
        plt.ylabel('Ref (rad)')
        plt.title('Joint tracking ' + str(i))
        plt.xlim([0, np.max(q_record[:, 0])])
        # plt.ylim([None, np.max(q_record[:, 1])])
        plt.show()

    def joint_motion_generator(self, q, vel=0.5) -> None:
        """

        :param q:
        :param vel:
        :return:
        """
        error = q - self.q
        t = np.max(np.abs(error)) / vel
        NTIME = int(t / self.dt)
        q_list = np.linspace(self.q, q, NTIME)
        for i in range(NTIME):
            self.joint_computed_torque_control(q_list[i, :])
            time.sleep(self.dt)

    def move_to_q0(self):
        """

        :return:
        """
        self.joint_motion_generator(self.q0)


    def cartesian_motion_generator(self, x, vel=0.1) -> None:
        """

        :param x: list
        :param vel:
        :return:
        """
        x_array = np.hstack(x)
        x_a = np.hstack([self.xh_local[i][:3] for i in range(self.finger_num)])
        error = x_array - x_a
        t = np.max(np.abs(error)) / vel
        NTIME = int(t / self.dt)
        x_all = np.linspace(x_a, x_array, NTIME)
        for i in range(NTIME):
            x_ref = [x_all[i, 4 * j: 4 + 4 * j] for j in range(self.finger_num)]
            q_desired = self.hand_kine.ik(x_ref, q_init=self.q)
            self.joint_computed_torque_control(q_desired)

    def joint_computed_torque_control(self, q: np.ndarray, dq=None, ddq=None, return_tau=False):
        """
        \tau = M(ddq + kp * q_error  + kd * dq_error) + C + g
        :param return_tau:
        :param q:  (20, ) desired joint positions
        :param dq:  (20, ) desired joint velocities
        :param ddq:
        :return:
        """
        # kp = 20000 *2
        kp = 10000
        kd = 200 / 2
        if dq is None:
            dq = np.zeros(self.n)  # (20, )
        if ddq is None:
            ddq = np.zeros(self.n)  # (20, )
        tau = self.M @ (ddq + kp * (q - self.q) + kd * (dq - self.dq)) + self.C
        if np.any(tau > 2):
            print("Torque out of limit")
            pass
        else:
            pass
            # print(np.max(np.abs(tau)))
        if return_tau:
            return tau
        else:
            self.send_torque(tau)

    def joint_impedance_control(self, q, dq=None, k=0.5):
        q = self.joint_limits(q)
        kp = np.ones(self.n) * 0.4 * k
        kd = 2 * np.sqrt(kp) * 0.005 * k
        if dq is None:
            dq = np.zeros(self.n)

        error_q = q - self.q
        error_dq = dq - self.dq

        qacc_des = kp * error_q + kd * error_dq + self.C
        # qacc_des = self.M @ (kp * error_q + kd * error_dq) + self.C

        self.send_torque(qacc_des)
        self.q_ref = np.copy(q)

    def _update_contact_force(self):
        # reset
        self._tip_force = {}
        for i in self.finger_tips_name:
            self._tip_force[i] = np.zeros(3)

        if self.d.ncon:  # number of contacts
            for i in range(self.d.ncon):
                c_array = np.zeros(6)
                c = self.d.contact[i]
                mujoco.mj_contactForce(self.m, self.d, i, c_array)

                # type 5 means geom,
                # https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html?highlight=mjtObj%20type#mjtobj
                geom_name = [mujoco.mj_id2name(self.m, 5, c.geom1), mujoco.mj_id2name(self.m, 5, c.geom2)]

                F = trans_force2world(c.pos, c.frame.reshape(3, 3).T, c_array)  # transfer it to world frame
                # F = trans_force2world(c.pos, c.frame.reshape(3, 3), c_array)  # transfer it to world frame
                # F is the force that geom1 applies on geom2
                if geom_name[0] == self.floor_name and geom_name[1] in self.finger_tips_name:
                    self._tip_force[geom_name[1]] = F[:3]
                elif geom_name[1] == self.floor_name and geom_name[0] in self.finger_tips_name:
                    self._tip_force[geom_name[0]] = -F[:3]
                else:
                    # print('Unexpected contact', geom_name, F)
                    pass

    @property
    def q(self):
        """
        hand joint angles
        :return: (10, ), numpy array
        """
        return self.d.qpos[7: 7 + self.n]  # noting that the order of joints is based on the order in *.xml file

    @property
    def dq(self):
        """
        hand joint velocities
        :return: (7, )
        """
        return self.d.qvel[6:6 + self.n]

    @property
    def C(self):
        """
        for iiwa, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (20, )
        """
        return self.d.qfrc_bias[6:6 + self.n]

    @property
    def M(self):
        """
        mass matrix for joints
        :return: (self.n, self.n)
        """
        M = np.zeros([self.m.nv, self.m.nv])
        mujoco.mj_fullM(self.m, M, self.d.qM)
        return M[6:6 + self.n, 6:6 + self.n]

    @property
    def x(self):
        """
        pose of the palm
        :return: (7,)
        """
        return self.d.qpos[:7]

    @property
    def dx(self):
        """
        velocity of the palm
        :return: (6,)
        """
        return self.d.qvel[:6]

    @property
    def ddx(self):
        """
        acceleration of the palm
        :return: (6,)
        """
        # mujoco.mj_rnePostConstraint(self.m, self.d)
        return self.d.qacc[:6]

    @property
    def x_obj(self) -> list:
        """
        :return: [(7,),...] objects poses by list, 
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = []
        for i in self.obj_names:
            poses.append(np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses

    @property
    def x_obj_dict(self) -> dict:
        """
        :return: [(7,),...] objects poses by list, 
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = {}
        for i in self.obj_names:
            poses[i] = (np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses

    @property
    def xh(self) -> list:
        """
        get the fingertip positions, wrt the world
        :return:  [(7,), (7,),...] a list of poses(position and quaternion)
        """
        x_list = []
        for i in self.finger_tips_site_name:
            x = self.d.site(i).xpos  # get position of the site
            R = self.d.site(i).xmat.reshape(3, 3)
            q = rot.mat2quat(R)
            x_list.append(np.concatenate([x, q]))

        return x_list

    @property
    def xh_local(self) -> list:
        """
        get the fingertip positions, wrt the hand_base
        :return:  [(7,), (7,),...] a list of poses(position and quaternion)
        """
        return self.hand_kine.forward_kine(self.q)

    def validate_fk(self):
        # to validate the correctness of fk
        for i in range(5):
            error = rot.pose_mul(self.x, self.xh_local[i]) - self.xh[i]
            print(error)  # should be all zeros

    @property
    def jac_local(self) -> list:
        """
        get the jacobian of fingertips, wrt the hand_base
        :return: jacobian,  [(3,4), (3,4)...]
        """
        return self.hand_kine.get_jac(self.q)

    @property
    def dxh(self):
        """
        get fingertip velocities
        :return:
        """
        vel_list = []
        for i in self.finger_tips_site_name:
            vel = np.zeros(6)
            # get the velocity of site
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_SITE,
                                     self.d.site(i).id, vel, 0)  # 1 for local, 0 for world, rot:linear
            vel_list.append(np.concatenate([vel[3:], vel[:3]]))  # linear, rot

        # J_tips = self.J_tips(full=True)
        # dq = [np.concatenate(self.dx, self.dq[i*4:i*4+4])  for i in range(self.finger_num)]
        # v_list = [J_tips[i]@ dq[i] for i in range(self.finger_num)]
        return vel_list

    @property
    def tip_force_dict(self):
        """
        contact forces between the floor and fingertips
        :return: a dict {'distal_1': np.zeros(3), 'distal_2': np.zeros(3),...}
        """
        return self._tip_force

    @property
    def tip_force(self):
        """
        contact forces between the floor and fingertips
        :return: a list of contact forces {np.zeros(3), np.zeros(3),...}
        """
        return list(self._tip_force.values())

    # @property
    # def J_tips(self, full=False):
    #     """
    #     fingertip jacobian wrt world
    #     \dot{x} = J(q) @ \dot{q}
    #     :param: full, if full jacobian or just to the end of finger
    #     :return: [(6, 4), (6, 4)...], a list of jacobian
    #     """
    #     J_list = []
    #     for i, site_name in enumerate(self.finger_tips_site_name):
    #         jacp = np.zeros([3, self.m.nv])  # translational jacobian wrt world
    #         jacr = np.zeros([3, self.m.nv])  # rotational jacobian wrt world
    #         mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.m.site(site_name).id)
    #         if full:
    #             pass
    #             jacp = np.concatenate([jacp[:, :6], jacp[:, 6 + i * 4: 6 + (i + 1) * 4]], axis=1)  # (3, 10)
    #             jacr = np.concatenate([jacr[:, :6], jacr[:, 6 + i * 4: 6 + (i + 1) * 4]], axis=1)  # (3, 10)
    #         else:
    #             jacp = jacp[:, 6 + i * 4: 6 + (i + 1) * 4]  # (3,4)
    #             jacr = jacr[:, 6 + i * 4: 6 + (i + 1) * 4]  # (3,4)
    #         J = np.concatenate([jacp, jacr], axis=0)
    #         J_list.append(J)
    #     return J_list


def trans_force2world(pos, R, F):
    """
    Transformation of force and torque from a frame to world frame
    :param pos: position of the frame wrt world
    :param R:  rotation matrix
    :param F:  force and torque (6, )
    :return: force and torque in world. (6, )
    """
    S = np.array([[0, -pos[2], pos[1]],
                  [pos[2], 0, -pos[0]],
                  [-pos[1], pos[0], 0]])
    T1 = np.concatenate([R, np.zeros((3, 3))], axis=1)
    T2 = np.concatenate([S, R], axis=1)
    T = np.concatenate([T1, T2])
    return T @ F


#  for locomotion,
# the goal is that,  given a command in Cartesian space, control the robot to move
class locomotion(Robot):
    def __init__(self, xml_path,
                 q0=None, finger_num=5, dof=4, obj_names=[], var_num=12, path_suffix='', v='v1'):
        super().__init__(xml_path, q0=q0, finger_num=finger_num, dof=dof, obj_names=obj_names, path_suffix=path_suffix,
                         v=v)
        # self.q0 = np.copy(self.q)
        # self.x0 = self.xh.copy()
        self.leg_states = [1] * self.finger_num  # 1 means supporting leg, 0 for swing
        self.f = 1.  # frequency
        self.mass = 3.11 * 0.2
        # self.mass = 3.11
        mesh = trimesh.load(path_suffix + 'descriptions/single_finger/shortest/meshes/palm_01.stl')
        mass_scale = self.mass / mesh.mass
        inertia = mesh.mass_properties['inertia'] * mass_scale  # the mesh has different xyz axis wrt mujoco
        self.I = np.array([[inertia[2, 2], 0, 0],
                           [0, inertia[1, 1], 0],
                           [0, 0, inertia[0, 0]]])  # [xx, yy, zz] inertia

        # self.g = np.array([])9.81
        self.var_num = var_num
        self.used_fingers = [0, 1, 3, 4]
        self.used_fingers_num = len(self.used_fingers)
        self.contact_states = np.ones(self.used_fingers_num)
        self.vel_cmd = np.zeros(6)

        # for swing leg
        self.swing_t = 0
        self.swing_t_total = 0.5
        self.stance_t_total = 0.5
        self.four_legs_t_total = 0.2
        self.lifting_height = 0.01

        # COM control
        self.kp_COM = np.diag([100, 100, 200, 100, 100, 200])
        self.kd_COM = np.diag([5, 5, 25, 5, 5, 5])

        self.var_num = 12

        # DS for locomotion

        self.linear_DS = linear_system(np.zeros(2))
        self.xh_local_init = copy.deepcopy(
            self.xh_local)  # the middle position of fingertips wrt the center of the robot.
        self.dis_error = 0
        self.ori_error = 0

    def update_leg_switch_state(self):
        """
        based on the contact states of feet, update contact type
        Based on the contact model coming out from the contact detection,
        the normal forces of swing legs are constrained to be zeros
        :return:
        """
        AA = np.zeros([self.var_num, self.var_num])
        F_contact_norm = np.linalg.norm(np.vstack(self.tip_force)[self.used_fingers, :], axis=1)
        F_contact_norm_b = F_contact_norm < 1e-10
        self.contact_states = ~ F_contact_norm_b
        for i in range(len(F_contact_norm)):
            if F_contact_norm_b[i]:  # for each swing leg, add equality constraint A @ F = 0, so that F_i = 0
                AA[i * 3:i * 3 + 3, i * 3:i * 3 + 3] = np.eye(3)
                # self.contact_states[]

        return AA

    def swing_controller(self, i):
        """
        tracking controller for swing legs.
        using a parabolic traj
        :param i: the index of leg
        :return: torque command
        """

    def stance_controller(self, i):
        """

        :param i: the index of leg
        :return:
        """

    def standby(self, f=0.5):
        """
        the crawling robot is standby mode
        :param f: frequency of swing legs
        :return:
        """
        pass

    def sin_test(self):
        """
        tuning the parameters of Cartesian impedance control
        :return:
        """

        pass

    def standby_test(self, v=None, lifting_height=0.01, return_q_list=False, T=10):
        """

        :param lifting_height: height of lifting legs
        :param v: (3,) moving velocity in world frame
        :return:
        """
        swing_leg = 0
        if v is None:
            v = np.zeros(3)

        t0 = time.time()
        x0 = self.xh_local.copy()
        self.leg_states = [0, 1, 0, 1, 1]
        self.f = 1

        q_list = []
        first = 0
        while 1:
            v_inhand = np.linalg.inv(rot.pose2T(self.x)[:3, :3]) @ v  # vel in hand_base frame
            t = time.time() - t0
            state_switch = int(t * self.f) % 2
            if state_switch == 0:
                self.leg_states = [0, 1, 0, 1, 1]

            elif state_switch == 1:
                self.leg_states = [1, 0, 1, 0, 1]
                first = 1
            else:
                # self.leg_states = [1, 1, 0, 1, 0]
                pass
            t1 = t - int(t * self.f) / self.f
            if int(t * self.f) > T:
                break

            x1 = copy.deepcopy(x0)
            x_cmd = []
            dx_cmd = [np.zeros(3)] * self.finger_num
            for i, state in enumerate(self.leg_states):
                if not state:  # if swing leg
                    x1[i][2] += lifting_height * (1 - np.cos(2 * np.pi * self.f * t + 0))
                    x1[i][:3] += v_inhand * t1

                else:  # for supporting leg
                    x1[i][:3] = x0[i][:3] + (v_inhand * (1 / self.f - t1)) * first

                if i == 4:
                    x1[i] = x0[i]
                x_cmd.append(x1[i][:3])
            if return_q_list:
                q_desired = self.Cartesian_space_cmd(x_cmd, return_q=True)
                q_desired = q_desired[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]]
                q_list.append(q_desired)
            else:
                self.Cartesian_space_cmd(x_cmd)
            error = [np.linalg.norm(x_cmd[i] - self.xh_local[i][:3]) for i in range(5)]
            # print(error)
            time.sleep(self.dt)
            print(self.ddx)

        if return_q_list:
            return q_list

    def Grf_ref_pre(self, base_acc, base_r, base_r_acc):
        """
        pre-computed reference of ground reaction force
        :return:
        """
        # base_acc[2, 0] += self.g
        # F_total = self.mass * base_acc
        #
        # Body_Rmatrix = self.RotMatrixfromEuler(base_r)
        # Global_I = np.dot(np.dot(Body_Rmatrix, self.body_I), Body_Rmatrix.T)
        # Torque_total = np.dot(Global_I, base_r_acc)
        #
        # self.F_Tor_tot[0:3, 0] = F_total[0:3, 0]
        # self.F_Tor_tot[3:6, 0] = Torque_total[0:3, 0]
        #
        # return self.F_Tor_tot
        pass

    def Grf_ref_opt(self):
        """
        optimal force by QP solver
        :return:
        """
        base_acc = self.kp_COM @ self.vel_cmd * self.dt + self.kd_COM @ (self.vel_cmd - self.dx)  # (6, )

        ## qp formulation
        var_num = self.var_num
        con_num = 24

        leg_force_opt_old = np.zeros([var_num, 1])
        F_balance = np.zeros([var_num, 1])
        F_balance[[2, 5, 8, 11], 0] = self.mass * np.abs(self.m.opt.gravity)[2] / 4  # the gravity to 4 legs

        A = np.zeros([6, 12])
        A[0:3, 0:3] = np.eye(3)
        A[0:3, 3:6] = np.eye(3)
        A[0:3, 6:9] = np.eye(3)
        A[0:3, 9:12] = np.eye(3)

        # qp parameter
        qp_S = np.diag([100000, 100000, 10000000, 100000, 100000, 10000000])
        qp_alpha = 10
        qp_beta = 1
        qp_gama = 1
        for i, leg in enumerate([0, 1, 3, 4]):
            com_fr = -self.x[:3] + self.xh[leg][:3]
            w_hat = self.skew_hat(com_fr)
            A[3:6, i * 3: i * 3 + 3] = w_hat

        H = 2 * A.T @ qp_S @ A + (qp_alpha + qp_beta) * np.eye(var_num)
        H = 0.5 * (H.T + H)

        # Eq.(9) in "Optimized Jumping on the MIT Cheetah 3 Robot"
        b = np.zeros([6, 1])
        b[:3, 0] = self.mass * (base_acc[:3] - self.m.opt.gravity)  # + 9.81
        I_global = rot.quat2mat(self.x[3:]) @ self.I @ rot.quat2mat(self.x[3:]).T
        b[3:6, 0] = I_global @ base_acc[3:]

        C = -2 * A.T @ qp_S @ b + qp_beta * F_balance + qp_gama * leg_force_opt_old  # (12, 1)

        ################ build constraints
        # AA @ x = 0
        AA = self.update_leg_switch_state()  # (12, 12)

        # inequality constraints for friction cone
        qp_H = np.zeros([con_num, var_num])
        qp_L = np.zeros([con_num, 1])
        ####  0<=fz<=fz_max
        mu = 0.5
        fz_max = - self.mass * self.m.opt.gravity[2] * 1.5
        for i in range(0, 4):
            qp_H[2 * i, 3 * i + 2] = -1
            qp_H[2 * i + 1, 3 * i + 2] = 1
            qp_L[2 * i + 1, 0] = fz_max
            # # pring(qp_L)
        ####  -u f_z =< f_x <= u f_z
        for i in range(0, 4):
            qp_H[8 + 2 * i, 3 * i] = -1
            qp_H[8 + 2 * i, 3 * i + 2] = -mu
            qp_H[8 + 2 * i + 1, 3 * i] = 1
            qp_H[8 + 2 * i + 1, 3 * i + 2] = -mu

        for i in range(0, 4):
            qp_H[16 + 2 * i, 3 * i + 1] = -1
            qp_H[16 + 2 * i, 3 * i + 2] = -mu
            qp_H[16 + 2 * i + 1, 3 * i + 1] = 1
            qp_H[16 + 2 * i + 1, 3 * i + 2] = -mu

        # use cvxpy to solve the qp problem
        t_opt = time.time()
        x = cp.Variable([var_num, 1])

        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, H) + C.T @ x), [qp_H @ x <= qp_L,
                                                                                AA @ x == np.zeros([var_num, 1])])
        prob.solve()
        leg_force_opt = x.value  # (12, )

        return leg_force_opt

    def skew_hat(self, vec_w):
        """
        skew matrix
        :param vec_w:
        :return:
        """
        w_hat = np.array([[0, -vec_w[2], vec_w[1]],
                          [vec_w[2], 0, -vec_w[0]],
                          [-vec_w[1], vec_w[0], 0]])
        return w_hat

    def bezier(self, t, x0, x1):
        """
        bezier curve for gait planning
        :param t:  [0, 1]
        :param x0:   start point
        :param x1:    end point
        :return:
        """
        assert 0 <= t <= 1
        return x0 + (t ** 3 + 3 * (t ** 2 * (1 - t))) * (x1 - x0)

    def collision_free_DS(self, attractor, x=None, x_objs=None, dx=None, alpha=30, mu=1, rho=1,
                          radius=[0.2, 0.2, 0.3, 0.3]):
        """

        :param attractor: (7,), position and orientation reference for the center of the robot
        :return: (2,), linear vel along xy
        """
        dim = 2
        self.linear_DS.attractor = attractor[:dim]  # update the attrator into the linear DS
        if x is None:
            x = self.x  # get the current position

        if x_objs is None:
            x_objs = self.x_obj[:len(radius)]
        if x_objs == []:
            vel = self.linear_DS.eval(x[:2], )
            # print(vel)
        else:
            len_obstacles = len(x_objs)
            phi = [alpha * (np.linalg.norm(x[:2] - x_objs[i][:2]) / mu - radius[i]) + 1 for i in
                   range(len_obstacles)]  # distance function

            vector = [alpha * (x[:2] - x_obj[:2]) / np.linalg.norm(x[:2] - x_obj[:2]) for x_obj in
                      x_objs]  # gradient wrt x

            M_all = np.eye(dim)  # calculate the modulation matrix
            for i in range(len_obstacles):
                w = 1
                for j in range(len_obstacles):
                    if j != i:
                        w = w * (phi[j] - 1) / (phi[i] - 1 + phi[j] - 1)
                # w = np.abs(w)  # w in [0, 1]
                w = np.clip(w, 0, 1)
                E = np.zeros((dim, dim))
                D = np.zeros((dim, dim))
                a0 = vector[i][np.nonzero(vector[i])[0][0]]  # the first nonzero component in vector (gradient)

                E[:, 0] = vector[i]

                tmp = np.power(np.abs(phi[i]), 1 / rho)
                # solve the tail-effect
                if dx is None:
                    lambda_0 = 1 - w / tmp
                else:
                    if np.dot(vector[i], dx) >= 0:  # remove the tailor effect. After passing the obstacle, recover
                        lambda_0 = 1
                    else:
                        lambda_0 = 1 - w / tmp
                lambda_1 = 1 + w / tmp
                # if phi[i] < 1:   # if in collision
                #     # lambda_0 =  lambda_0 * 40
                #     lambda_0 =  20
                #     lambda_1 = 0   # remove the vel towards the attractor
                # lambda_0 = np.abs(lambda_0)

                D[0, 0] = lambda_0
                for j in range(1, dim):
                    if dim in [2, 4]:
                        E[0, j] = - vector[i][j]
                        E[j, j] = a0
                    else:
                        E[0, j] = - vector[i][0, j]
                        E[j, j] = a0[j]

                    D[j, j] = lambda_1

                M = E @ D @ np.linalg.inv(E)
                # if np.any(np.array(phi) <1):
                #     M = np.zeros([dim, dim]) # if goes into the obstacle, then return zero velocity
                M_all = M_all @ M

            vel = M_all @ self.linear_DS.eval(x[:2], )  # desired vel in world frame
            if np.any(np.array(phi) < 1):
                k = np.nonzero(np.array(phi) < 1)[0][0]
                vel_linear = self.linear_DS.eval(x[:2], )
                vel = 0.7 * vector[k] / np.linalg.norm(vector[k]) * np.linalg.norm(
                    vel_linear) + 0.7 * vel_linear  # along the gradient direction
        #     vel =
        vel_local = rot.quat2mat(x[3:7]).T @ np.array([vel[0], vel[1], 0])
        return vel_local[:2]

    def moveto_attractor(self, attractor, x=None, f=2., swing_legs=[0, 2, 4], stance_legs=[1, 3, 5], threshold=0.01,
                         max_vel=0.005,
                         radius=[0.2, 0.2, 0.3, 0.3], angle_threshold=5):
        """

        :param attractor: (7,), attractor for the center of the palm
        :param f:
        :param swing_legs: index list of swing legs for the first half period of locomotion
        :param stance_legs:
        :param threshold:  the distance threshold to determine that if it reaches the goal or not
        :return:
        """
        dx_local = np.zeros(2)
        t_period_last = 1e10
        t0 = time.time()
        dx = np.zeros(2)
        beta = 0.01
        w = 0
        while 1:
            t_now = time.time() - t0
            t_period = t_now % (1 / f)  # get the mod
            if t_period < t_period_last:  # at the beginning of each period, update the velocity command
                vel_local = self.collision_free_DS(attractor, dx=dx_local, radius=radius)
                w = 1 * rot.log(rot.quat_mul(self.x[3:7], rot.quat_conjugate(attractor[3:]))) / (
                        1 + rot.ori_dis(self.x[3:7], attractor[3:]))
                w = w[2] * 0.5
            q_desired = self.locomotion_traj(t_now, swing_legs, stance_legs, v=vel_local[:2] * beta, w=w, f=f,
                                             max_vel=max_vel)
            self.joint_computed_torque_control(q_desired)

            t_period_last = t_period
            self.dis_error = np.linalg.norm(self.x[:2] - attractor[:2])
            self.ori_error = rot.ori_dis(self.x[3:], attractor[3:]) * 180 / np.pi

            if self.dis_error < threshold and self.ori_error < angle_threshold:
                print("reach the goal:", attractor, '   with xy error:', self.dis_error, 'ori_error', self.ori_error)
                break

    def locomotion_traj(self, t_now, swing_legs, stance_legs, v=np.array([0.0, -0.02, 0]), w=None, f=2., h=0.02,
                        max_vel=0.01, xh_local_init=None, q_init=None):
        """

        :param swing_legs:
        :param stance_legs:
        :param t_now: current time
        :param v: (2,), desired vel
        :param w: float,  angular vel along z axis
        :param f:
        :param h: 0.04   z axis lifting height for swing legs
        :param max_vel,   the max vel of linear steps
        :return:
        """
        if xh_local_init is None:
            xh_local_init = self.xh_local_init
        if q_init is None:
            q_init = copy.deepcopy(self.q)
        T = 1 / f
        xy_step_linear = [v[:2] * T / 2 * 0.5] * 6  # based the MIT paper
        # adjust the linear vel for better convergence to the goal position
        if self.dis_error > 0.015 and np.linalg.norm(xy_step_linear[0] < max_vel):
            xy_step_linear = [xy_step_linear[0] / (
                    np.linalg.norm(xy_step_linear[0]) + 1e-5) * max_vel] * self.finger_num

        if w is not None:
            theta = w * T / 2 * 0.5
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            # a combination of motion for angular velocity and linear velocity
            xy_step = [xh_local_init[i][:2] @ R - xh_local_init[i][:2] + xy_step_linear[i] for i in
                       range(self.finger_num)]

        x_cmd = copy.deepcopy(xh_local_init)
        x_cmd = [a[:3] for a in x_cmd]  # position command for legs

        dt = 0.002
        t_period = t_now % (1 / f)  # [0, T]

        if t_period < T / 2:  # FL and BR, front left and back right legs move during [0, T/2]
            swing_legs_, stance_legs_ = swing_legs, stance_legs
        else:
            swing_legs_, stance_legs_ = stance_legs, swing_legs

        for i in swing_legs_:
            t_tmp = t_period % (T / 2)
            x_cmd[i][:2] = self.bezier(t_tmp / (T / 2), xh_local_init[i][:2] - xy_step[i],
                                       xh_local_init[i][:2] + xy_step[i])
            if t_tmp <= T / 4:
                x_cmd[i][2] = self.bezier(t_tmp / (T / 4), xh_local_init[i][2], xh_local_init[i][2] + h)  # z axis
            else:
                x_cmd[i][2] = self.bezier((t_tmp - T / 4) / (T / 4), xh_local_init[i][2] + h, xh_local_init[i][2])

        for i in stance_legs_:
            t_tmp = t_period % (T / 2)
            x_cmd[i][:2] = self.bezier(t_tmp / (T / 2), xh_local_init[i][:2] + xy_step[i],
                                       xh_local_init[i][:2] - xy_step[i])
            x_cmd[i][2] = xh_local_init[i][2]

        q_desired = self.hand_kine.ik(x_cmd, q_init=q_init, fingers=swing_legs + stance_legs)

        return q_desired

    def locomotion_traj_3_phase(self, t_now, fingers=[0,1,4],  v=np.array([0.0, -0.02, 0]), w=None, f=2., h=0.02,
                        max_vel=0.01, xh_local_init=None, q_init=None):
        """


        :param t_now: current time
        :param v: (2,), desired vel
        :param w: float,  angular vel along z axis
        :param f:
        :param h: 0.04   z axis lifting height for swing legs
        :param max_vel,   the max vel of linear steps
        :return:
        """
        if xh_local_init is None:
            xh_local_init = self.xh_local_init
        if q_init is None:
            q_init = copy.deepcopy(self.q)
        T = 1 / f
        xy_step_linear = [v[:2] * T / 2 * 0.5] * 6  # based the MIT paper
        # adjust the linear vel for better convergence to the goal position
        if self.dis_error > 0.015 and np.linalg.norm(xy_step_linear[0] < max_vel):
            xy_step_linear = [xy_step_linear[0] / (
                    np.linalg.norm(xy_step_linear[0]) + 1e-5) * max_vel] * self.finger_num

        if w is not None:
            theta = w * T / 2 * 0.5
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            # a combination of motion for angular velocity and linear velocity
            xy_step = [xh_local_init[i][:2] @ R - xh_local_init[i][:2] + xy_step_linear[i] for i in
                       range(self.finger_num)]

        x_cmd = copy.deepcopy(xh_local_init)
        x_cmd = [a[:3] for a in x_cmd]  # position command for legs

        dt = 0.002
        t_period = t_now % (1 / f)  # [0, T]

        if t_period < T / 3:  # FL and BR, front left and back right legs move during [0, T/2]
            swing_legs = [fingers[0]]
            stance_legs = [i for i in fingers if i not in swing_legs]
        elif T / 3 <= t_period < T / 3 * 2:
            swing_legs = [fingers[1]]
            stance_legs = [i for i in fingers if i not in swing_legs]
        else:
            swing_legs = [fingers[2]]
            stance_legs = [i for i in fingers if i not in swing_legs]

        for i in swing_legs:
            t_tmp = t_period % (T / 3)
            x_cmd[i][:2] = self.bezier(t_tmp / (T / 3), xh_local_init[i][:2] - xy_step[i], xh_local_init[i][:2] + xy_step[i])
            if t_tmp <= T / 6:
                x_cmd[i][2] = self.bezier(t_tmp / (T / 6), xh_local_init[i][2], xh_local_init[i][2] + h)  # z axis
            else:
                x_cmd[i][2] = self.bezier((t_tmp - T / 6) / (T / 6), xh_local_init[i][2] + h, xh_local_init[i][2])

        for flag, i in enumerate(stance_legs):
            t_tmp = t_period % (T / 3)
            x_cmd[i][:2] = self.bezier(t_tmp / (T / 3), xh_local_init[i][:2] + xy_step[i] * flag,
                                    xh_local_init[i][:2] - xy_step[i] * (1 - flag))
            x_cmd[i][2] = xh_local_init[i][2]

        q_desired = self.hand_kine.ik(x_cmd, q_init=q_init,fingers=fingers)


        return q_desired

    def locomotion_traj_4_phase(self, t_now, fingers=[0,3,5,4],  v=np.array([0.0, -0.02, 0]), w=None, f=2., h=0.02,
                        max_vel=0.01, xh_local_init=None, q_init=None):
        """


        :param t_now: current time
        :param v: (2,), desired vel
        :param w: float,  angular vel along z axis
        :param f:
        :param h: 0.04   z axis lifting height for swing legs
        :param max_vel,   the max vel of linear steps
        :return:
        """
        if xh_local_init is None:
            xh_local_init = self.xh_local_init
        if q_init is None:
            q_init = copy.deepcopy(self.q)
        T = 1 / f
        xy_step_linear = [v[:2] * T / 2 * 0.5] * 6  # based the MIT paper
        # adjust the linear vel for better convergence to the goal position
        if self.dis_error > 0.015 and np.linalg.norm(xy_step_linear[0] < max_vel):
            xy_step_linear = [xy_step_linear[0] / (
                    np.linalg.norm(xy_step_linear[0]) + 1e-5) * max_vel] * self.finger_num

        if w is not None:
            theta = w * T / 2 * 0.5
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
            # a combination of motion for angular velocity and linear velocity
            xy_step = [xh_local_init[i][:2] @ R - xh_local_init[i][:2] + xy_step_linear[i] for i in
                       range(self.finger_num)]

        x_cmd = copy.deepcopy(xh_local_init)
        x_cmd = [a[:3] for a in x_cmd]  # position command for legs

        dt = 0.002
        t_period = t_now % (1 / f)  # [0, T]

        if t_period < T / 4:
            swing_legs = [fingers[0]]
            stance_legs = [fingers[1], fingers[2], fingers[3]]
        elif T / 4 <= t_period < T / 4 * 2:
            swing_legs = [fingers[1]]
            stance_legs = [ fingers[2], fingers[3],  fingers[0]]
        elif T / 4 * 2 <= t_period < T / 4 * 3:
            swing_legs = [fingers[2]]
            stance_legs = [ fingers[3], fingers[0],  fingers[1]]
        else:
            swing_legs = [fingers[3]]
            stance_legs = [fingers[0],  fingers[1], fingers[2]]

        for i in swing_legs:
            t_tmp = t_period % (T / 4)
            x_cmd[i][:2] = self.bezier(t_tmp / (T / 4), xh_local_init[i][:2] - xy_step[i], xh_local_init[i][:2] + xy_step[i])
            if t_tmp <= T / 8:
                x_cmd[i][2] = self.bezier(t_tmp / (T / 8), xh_local_init[i][2], xh_local_init[i][2] + h)  # z axis
            else:
                x_cmd[i][2] = self.bezier((t_tmp - T / 8) / (T / 8), xh_local_init[i][2] + h, xh_local_init[i][2])

        for flag, i in enumerate(stance_legs):
            t_tmp = t_period % (T / 4)
            x_cmd[i][:2] = self.bezier(t_tmp / (T / 4), xh_local_init[i][:2] + xy_step[i] * (-1/6 + 2/6*flag),
                                    xh_local_init[i][:2] + xy_step[i] * (-3/6 + 2/6*flag))
            x_cmd[i][2] = xh_local_init[i][2]

        q_desired = self.hand_kine.ik(x_cmd, q_init=q_init,fingers=fingers)


        return q_desired

    def move_along_z(self, finger_list=[0,1,2,3,4], z=-0.02, vel=0.5, xh_local_init=None, q_keep=None, real=True):
        """

        :param finger_list:
        :param z:
        :param t:
        :param xh_local_init:
        :return:
        """
        if xh_local_init is None:
            xh_local_init = self.xh_local_init
        if q_keep is None:
            q_keep = self.q
        x_fingers = [x[:3] - np.array([0, 0, z]) for x in xh_local_init]
        q_desired = self.hand_kine.ik(x_fingers, q_init=self.q, fingers=finger_list)
        for i in range(self.finger_num):
            if i not in finger_list:
                q_desired[i*4:i*4+4] = q_keep[i*4:i*4+4]
        if real:
            return q_desired
        else:
            self.joint_motion_generator(q_desired, vel=vel)


    def move_finger_tips(self, t_now, finger_list, x_fingers, t_all=0.5, q_init=None, xh_local_init=None, h=0.02):
        """


        :param finger_list:
        :param x_cmd:
        :param xh_local_init:
        :return:
        """
        if xh_local_init is None:
            xh_local_init = self.xh_local_init

        if q_init is None:
            q_init = copy.deepcopy(self.q)
        # xh_local_init = self.hand_kine.forward_kine(q_init)

        x_cmd = copy.deepcopy(xh_local_init)
        t_period = t_now % t_all
        for i in finger_list:
            t_tmp = t_period % t_all
            x_cmd[i][:2] = self.bezier(t_tmp / t_all, xh_local_init[i][:2],
                                       x_fingers[i][:2])
            if t_tmp <= t_all / 2:
                x_cmd[i][2] = self.bezier(t_tmp / (t_all / 2), xh_local_init[i][2], xh_local_init[i][2] + h)  # z axis
            else:
                x_cmd[i][2] = self.bezier((t_tmp - t_all / 2) / (t_all / 2), xh_local_init[i][2] + h,
                                          xh_local_init[i][2])

        q_desired = self.hand_kine.ik(x_cmd, q_init=q_init)

        return q_desired


class linear_system:
    def __init__(self, attractor, A=None, max_vel=2, scale_vel=2, eps=1e-8, dt=0.001):

        self.dim = len(attractor)
        if A is None:
            A = np.eye(self.dim) * (-1)

        self.A = A * scale_vel
        self.attractor = attractor
        self.max_vel = max_vel

        # self.scale_vel = scale_vel
        self.eps = eps
        self.dt = 0.001

    def eval(self, q: np.ndarray, k=1) -> np.ndarray:
        A = self.A / (np.linalg.norm(q - self.attractor) + self.eps)
        dq = self.A.dot((q - self.attractor) ** k)
        if self.max_vel is not None:
            if np.linalg.norm(dq) > self.max_vel:
                dq = dq / np.linalg.norm(dq) * self.max_vel

        return dq


if __name__ == '__main__':
    # xml_path = 'descriptions/five_finger_hand_ssss.xml'
    # # chain = pk.build_chain_from_mjcf('descriptions/single_finger_ssss.xml', 'finger_1')
    # # print(chain)
    # r = locomotion(xml_path)
    # r.warmup()
    # # r.standby_test()
    # #### ik test
    # print(r.hand_kine.ik(r.xh_local, r.q) - r.q)

    # r.sin_test_joint_space(i=3)
    # r.sin_test_cartesian_space(i=4)

    # test the locomotion to an attractor
    path_suffix = ''
    xml_path = 'descriptions/six_finger_hand_llll.xml'
    q = np.array([0, 0, 0.13, 9.99999738e-01,
                  -4.46340938e-04, 5.67392530e-04, -4.89152197e-05,
                  0, 7.91144283e-01, 0.3, 0.3,
                  0, 7.91144283e-01, 0.3, 0.3,
                  0, 7.91144283e-01, 0.3, 0.3,
                  0, 7.91144283e-01, 0.3, 0.3,
                  0, 7.91144283e-01, 0.3, 0.3,
                  0, 7.91144283e-01, 0.3, 0.3, ])
    obj_names = ['sphere_1', 'cylinder_1', 'box_1', 'box_2']
    r = locomotion(xml_path, q0=q, finger_num=6, path_suffix=path_suffix, obj_names=obj_names)

    print('locomotion freq', r.f)
    # take a rest to let the acceleration to zero
    t0 = time.time()
    q0 = copy.deepcopy(q[7:])
    while 1:
        # r.step()
        # r.send_torque(r.C)
        r.joint_computed_torque_control(q0)
        if time.time() - t0 > 4:
            break
    r.xh_local_init = copy.deepcopy(r.xh_local)

    r.moveto_attractor(np.array([1, 1, 0, 1, 1, 0, 0]))
