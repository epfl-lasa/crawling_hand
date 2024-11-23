import numpy as np
import mujoco
import tools.rotations as rot
import os


class Robot:
    def __init__(self, m: mujoco._structs.MjModel, d: mujoco._structs.MjModel, view, fingers, dof, obj_names=[],
                 auto_sync=True,xyz=None,
                 q0=None):
        self.m = m
        self.d = d
        self.view = view
        self.auto_sync = auto_sync
        self.obj_names = obj_names
        self.fingers = fingers
        self.dof = dof

        self.finger_num = sum(self.fingers)
        self.n = int(sum(np.array(self.fingers) * np.array(self.dof)))
        self.q_ref = np.zeros(self.n)
        if q0 is None:
            self.q0 = np.zeros(self.n)

        q_limit = [-np.pi / 2, np.pi / 2]
        self.q_limit = np.array(q_limit)

        # self.path = 'data_records/finger_' + str(finger_num) + '_dof_' + str(dof) + '/'
        # file_name = self.path + 'standup_pos.npy'
        # if os.path.isfile(file_name):
        #     self.standup_pos = np.array(np.load(file_name))
        # else:
        #     print('Need to generate the standup position')
        #     self.standup_pos = np.empty(0)
        # self.modify_joint(self.standup_pos)  # set the initial joint positions
        self.step()
        self.sync()
        self.finger_tips_site_name = ['finger_site_0', 'finger_site_1']


        if view is not None:
            self.viewer_setup(xyz=xyz)
        #
        # self.render = mujoco.Renderer(self.m, 1080, 1920)
        # camera_info = mujoco._structs.MjvCamera()
        # camera_info.distance = 0.9 * 10
        # camera_info.lookat[0] = 0
        # camera_info.lookat[1] = 0
        # camera_info.lookat[2] = 0.21066664
        # camera_info.elevation = -90
        # camera_info.azimuth = 90
        # camera_info.trackbodyid = 0
        # self.render.update_scene(self.d, camera='closeup')

    def viewer_setup(self, xyz=None):
        # self.view.cam.type= mujoco.mjtCamera.mjCAMERA_TRACKING
        # self.view.cam.trackbodyid = 2  # id of the body to track ()
        # self.viewer.cam.distance = self.sim.model.stat.extent * 0.05  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.view.cam.distance = 0.9 # how much you "zoom in", model.stat.extent is the max limits of the arena
        if xyz is None:
            xyz = [0,0.3,0.22]

        self.view.cam.lookat[0] = xyz[0]  # x,y,z offset from the object (works if trackbodyid=-1)
        self.view.cam.lookat[1] = xyz[1]
        self.view.cam.lookat[2] = xyz[2]
        self.view.cam.elevation = -45  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.view.cam.azimuth = -90   # camera rotation around the camera's vertical axis



    def step(self):
        mujoco.mj_step(self.m, self.d)  # run one-step dynamics simulation
        # mujoco.mj_forward(self.m, self.d)

    def modify_joint(self, joints: np.ndarray) -> None:
    #     """
    #     :param joints: (7,) or (16,) or (23,), modify joints for iiwa or/and allegro hand
    #     :return:
    #     """
    #     if len(joints) == 0 and self.finger_num == 6 and self.dof == 4:
    #         joints = np.array([-7.74409002e-02, -1.06154745e-02, 1.48529641e-01, 9.99950094e-01,
    #                            -4.66807779e-03, 5.83562850e-03, -6.63054066e-03, 4.95841229e-01,
    #                            7.98255111e-01, 4.47088967e-01, -5.12779831e-01, -2.87583006e-04,
    #                            9.03542841e-01, 5.07803362e-01, -5.36107242e-01, -4.69454802e-04,
    #                            8.88426676e-01, 5.00475426e-01, -5.33795484e-01, -7.98168577e-01,
    #                            9.32265733e-01, 5.22083704e-01, -5.40848648e-01, -4.96924787e-01,
    #                            8.96803783e-01, 5.06104424e-01, -5.36594938e-01, -4.96329204e-01,
    #                            8.64216828e-01, 4.88780153e-01, -5.31951625e-01])  # stand up position
    #         self.d.qpos[7: 7 + self.n] = q_init
    #     elif len(joints) == 0 and self.finger_num == 6 and self.dof == 2:
    #         q_init = np.array([0.5, -1.3,
    #                            0, 1.3,
    #                            0, 1.3,
    #                            0, 1.3,
    #                            -0.5, 1.3,
    #                            -0.5, 1.3])
    #         self.d.qpos[7: 7 + self.n] = q_init
    #     if len(joints) == self.n:
    #         self.d.qpos[7: 7 + self.n] = joints
    #     elif len(joints) == self.n + 7:
    #         self.d.qpos[: 7 + self.n] = joints
    #     else:
    #         pass
    #
    #     self.step()
    #     self.sync()
        self.d.qpos[:len(joints)] = joints
        self.step()

        self.sync()


    def joint_limits(self, q):

        q_tmp = np.clip(q, self.q_limit[0], self.q_limit[1])

        return q_tmp.flatten()

    def sync(self):
        if self.view is not None:
            self.view.sync()

    def modify_first_finger(self, q, offset=0):

        self.d.qpos[7+offset: 7+offset + len(q)] = q



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
        if self.auto_sync:
            self.sync()

    def reset(self, x=None, q=None):
        """
        reset all states of the robot
        """
        if x is None:
            x = np.array([0, 0, 0.01, 1, 0, 0, 0.])
        if q is None:
            q = np.zeros(self.n)
        self.d.qpos[7: 7 + self.n] = q
        self.d.qpos[:7] = x
        for i in range(100):
            self.joint_impedance_control(q)
            self.step()
            self.sync()

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

    def joint_impedance_control(self, q, dq=None, k=0.5):
        q = self.joint_limits(q)
        kp = np.ones(len(q)) * 0.4 * k
        kd = 2 * np.sqrt(kp) * 0.005 * k
        if dq is None:
            dq = np.zeros(len(q))

        error_q = q - self.q
        error_dq = dq - self.dq

        qacc_des = kp * error_q + kd * error_dq + self.C

        self.send_torque(qacc_des)
        self.q_ref = np.copy(q)

    @property
    def q(self):
        """
        iiwa joint angles
        :return: (10, ), numpy array
        """
        return self.d.qpos[7: 7 + self.n]  # noting that the order of joints is based on the order in *.xml file

    @property
    def dq(self):
        """
        iiwa joint velocities
        :return: (7, )
        """
        return self.d.qvel[6:6 + self.n]

    @property
    def C(self):
        """
        for iiwa, C(qpos,qvel), Coriolis, computed by mj_fwdVelocity/mj_rne (without acceleration)
        :return: (7, )
        """
        return self.d.qfrc_bias[6:6 + self.n]

    @property
    def x(self):
        """
        pose of the palm
        :return:
        """
        return self.d.qpos[:7]

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
    def x_obj(self):
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
    def x_obj_dict(self):
        """
        :return: [(7,),...] objects poses by list, 
         // computed by mj_fwdPosition/mj_kinematics
        https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=xipos#mjdata
        """  # print(joint_id)
        poses = {}
        for i in self.obj_names:
            poses[i] = (np.concatenate([self.d.body(i).xpos, self.d.body(i).xquat]))
        return poses






