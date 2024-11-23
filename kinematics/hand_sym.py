"""
 An symbolic experssion of kinematic model for the new real robotic hand

"""
import sympy as sy
import xml.etree.ElementTree as ET
import numpy as np

import sys

sys.path.append("..")
import tools.rotations as rot
import os.path
import pickle
from sympy.abc import x, y, z, a


class Robot:
    def __init__(self, path=None, path_suffix='', finger_num=5, version='v1'):
        if version == 'v1':
            if path is None:
                if finger_num == 5:
                    path = path_suffix + 'descriptions/five_finger_hand_bodies.xml'
                else:
                    path = path_suffix + 'descriptions/six_finger_hand_bodies.xml'
            alpha = 1
        else:
            if finger_num == 5:
                path = path_suffix + 'descriptions/v2/hand_v2_bodies.xml'
            else:
                path = path_suffix + 'descriptions/v2/hand_6_fingers_v2_bodies.xml'
            alpha = 1

        file_name = path_suffix + 'kinematics/q2pose_' + str(finger_num) + '_' + version + '.txt'
        self.finger_num = finger_num
        self.dof = self.finger_num * 4
        tree = ET.parse(path)
        self.root = tree.getroot()

        # Jacobian calculation
        if finger_num == 5:
            self.q_list = ['q_t', 'q_i', 'q_m', 'q_r', 'q_l']  # thumb, index, middle, ring, little
        else:
            self.q_list = ['q_t', 'q_i', 'q_m', 'q_r', 'q_l', 'q_n']  # thumb, index, middle, ring, little
        self.n = len(self.q_list)
        self.T_list = None  # a list of symbolic expression for the fingertip poses based on joints
        self.T_fingers = None  # "finger_i" wrt hand_base
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                self.T_list = pickle.load(f)
            print('Kinematic model has been loaded from ' + file_name)
            T = np.load(path_suffix + 'kinematics/' + 'T_fingers.npy')
            self.T_fingers = [T[i, :, :] for i in range(self.n)]
        else:
            print(
                'Start to load xml file to build the kinematics. This might take about 20s, but only for the first time.')
            self.read_xml()  # read the kinematic chains from .xml file
            with open(file_name, 'wb') as f:
                pickle.dump(self.T_list, f)
            print('Kinematic model has been saved to ' + file_name)
            np.save(path_suffix + 'kinematics/T_fingers', self.T_fingers)

        self.qt = sy.symbols('q_t:4')  # 4 dofs for each finger
        self.qi = sy.symbols('q_i:4')
        self.qm = sy.symbols('q_m:4')
        self.qr = sy.symbols('q_r:4')
        self.ql = sy.symbols('q_l:4')
        self.qt2 = sy.symbols('q_n:4')
        if finger_num == 5:
            self.q = [self.qt, self.qi, self.qm, self.qr, self.ql]
        else:
            self.q = [self.qt, self.qi, self.qm, self.qr, self.ql, self.qt2]

        # a list of lambda function for the fingertip poses, which will receive joints for numeric computation
        self.fk = [sy.lambdify([self.q[i]], self.T_list[i]) for i in range(self.n)]

        # for position jacobian
        self.jac_syms = [sy.Matrix(self.T_list[i][:3, 3]).jacobian(list(self.q[i])) for i in
                         range(self.n)]  # symbolic value for position jacobian
        # a list of lambda function for the fingertip jacobians, which will receive joints for numeric computation
        self.jac = [sy.lambdify([self.q[i]], self.jac_syms[i]) for i in range(self.n)]
        self.joint_limits = [-np.pi / 2, np.pi / 2]
        # symbolic ik

        for i in range(1):
            lengths = sy.symbols('l_:3')
            Y = sy.symbols('Y')
            Z = sy.symbols('Z')
            Eqs = self.T_list[i][:3, 3] - sy.Matrix([x, y, z])
            Eqs = Eqs.subs({self.q[i][-1]: self.q[i][-2]})
            Eq2 = sy.simplify(Eqs[0] * sy.cos(self.q[i][0]) - Eqs[1] * sy.sin(self.q[i][0]) * alpha, rational=True)
            q0 = sy.solve(Eq2, self.q[i][0], dict=False)  # the first joint, two solutions
            self.q0 = sy.lambdify([[x, y]], q0)  # given x,y, calculate q0

            Eq2_ = [0, 0, 0]
            Eq2_[1] = lengths[0] * sy.cos(self.q[0][1]) + lengths[1] * sy.cos(self.q[0][1] + self.q[0][2]) + \
                      lengths[2] * sy.cos(self.q[0][1] + 2 * self.q[0][2])
            Eq2_[2] = lengths[0] * sy.sin(self.q[0][1]) + lengths[1] * sy.sin(self.q[0][1] + self.q[0][2]) + \
                      lengths[2] * sy.sin(self.q[0][1] + 2 * self.q[0][2])

            l_sol = sy.solve(-Eqs[2] - Eq2_[2] + Z, list(lengths) + [Z], dict=True)[0]
            Z = l_sol[Z]
            y_sol = sy.solve(Eqs[1] - Eq2_[1] + Y, list(lengths) + [Y], dict=True)[0]
            Y = y_sol[Y] / sy.cos(self.q[0][0])
            self.YZ = sy.lambdify([[y, z, self.q[0][0]]], [Y, Z])

            # OB = l_sol[lengths[0]] + l_sol[lengths[1]] / (2 * sy.cos(finger.q[i][2]))
            OB = l_sol[lengths[0]] + l_sol[lengths[1]] / (2 * a)
            OB_ = sy.symbols('OB_')
            MB_ = sy.symbols('MB_')
            Y_ = sy.symbols('Y_')
            Z_ = sy.symbols('Z_')
            # MB = l_sol[lengths[2]] + l_sol[lengths[1]] / (2 * sy.cos(finger.q[i][2]))
            MB = l_sol[lengths[2]] + l_sol[lengths[1]] / (2 * a)

            Eq_theta_2 = OB ** 2 + MB ** 2 - (Y_ ** 2 + Z_ ** 2) - 2 * OB * MB * (1 - 2 * a ** 2)  # Law of cosines
            a_ = sy.solve(Eq_theta_2, a)  # two solutions, the third joint, a = cos(q2)
            self.a_ = sy.lambdify([[Y_, Z_]], a_)
            self.OB_MB = sy.lambdify([a], [OB, MB])  # calculate the four intermediate

            tmp = sy.acos((OB_ ** 2 + Y_ ** 2 + Z_ ** 2 - MB_ ** 2) / (2 * OB_ * sy.sqrt(Y_ ** 2 + Z_ ** 2)))
            q1 = [sy.atan(Z_ / Y_) - tmp, sy.atan(Z_ / Y_) + tmp]
            self.q1 = sy.lambdify([[Y_, Z_, OB_, MB_]], q1)

    def ik(self, positions: list, q_init=None, k=-1, fingers=None):
        """
        Given a list of fingertip positions, calculate the joint positions
        772 µs
        :param positions:
        :return: [(self.dof,)]
        """

        if q_init is None:
            q_all = [np.zeros(self.dof), np.zeros(self.dof)]
        else:
            q_all = np.zeros(self.dof)

        # fingers = list(range(self.finger_num))
        if fingers is None:
            fingers =  list(range(self.finger_num))
        for i in fingers:
            # q_float = np.zeros(4)
            if 0 <= k != i:
                continue
            else:
                pass
            if i == 0:
                xyz = positions[i][:3]
            else:
                T = np.eye(4)
                T[:3, 3] = positions[i][:3]
                T_relative = np.linalg.inv(self.T_fingers[i]) @ T  # xyz in finger i root frame
                xyz = self.T_fingers[0] @ T_relative  # xyz in hand_base frame
                xyz = xyz[:3, 3]  # x, y, z (3,)

            q0 = self.q0(xyz[:2])
            q0 = [qi for qi in q0 if self.joint_limits[0] < qi < self.joint_limits[1]]
            q0 = q0[0]

            YZ = self.YZ([xyz[1], xyz[2], q0])
            a_ = self.a_(YZ)
            a_ = [ai for ai in a_ if ai > 0]
            a_ = a_[0]

            if not -1 - 1e-10 < a_ <= 1 + 1e-10:
                print('IK no solution error cos(q0)=', a_,  "i=", i)
                print('Check if x y z is in the reachable region')
                # return None
                a_ = np.clip(a_, 1e-10 - 1, 1 - 1e-10)

            OB_MB = self.OB_MB(a_)


            q2 = np.arccos(a_)
            q2 = [q2, -q2]
            q1 = self.q1([YZ[0], YZ[1], OB_MB[0], OB_MB[1]])
            q_float = [np.array([q0, q1[0], q2[0], q2[0]]), np.array([q0, q1[1], q2[1], q2[1]])]
            if q_init is not None:
                dis_1 = np.linalg.norm(q_float[0] - q_init[4 * i: 4 + 4 * i])
                dis_2 = np.linalg.norm(q_float[1] - q_init[4 * i: 4 + 4 * i])
                if dis_1 < dis_2:
                    q_float = q_float[0]
                else:
                    q_float = q_float[1]
                q_all[4 * i: 4 + 4 * i] = q_float
                if k == i:
                    return q_float
            else:
                q_all[0][4 * i: 4 + 4 * i] = q_float[0]
                q_all[1][4 * i: 4 + 4 * i] = q_float[1]

        return q_all

    def read_xml(self):
        T_list = []  # the list of fingertip poses in (4, 4)
        site = []
        T0 = []
        for body in self.root.iter('site'):
            if body.attrib['name'][-3:] == 'tip':
                site.append(body.attrib)

        for a in range(self.n):
            b = []  # list of bodies for one finger
            j = []  # list of joints for one finger
            for body in self.root[a].iter('body'):
                b.append(body.attrib)
            for body in self.root[a].iter('joint'):
                j.append(body.attrib)

            T = sy.eye(4)
            pos = np.fromstring(b[0]['pos'], sep=' ')
            quat = np.fromstring(b[0]['quat'], sep=' ') if 'quat' in b[0] else np.array([1, 0, 0, 0.])
            T = T * rot.pose2T(np.concatenate([pos, quat]))
            T0.append(np.array(T).astype(np.float64))
            num = len(j)
            q = sy.symbols(self.q_list[a] + ':' + str(num))

            # for i in range(len(b)):
            for i in range(num):  # number of dof for each finger
                pos = np.fromstring(b[i + 1]['pos'], sep=' ') if 'pos' in b[i + 1] else np.array([0, 0, 0.])
                quat = np.fromstring(b[i + 1]['quat'], sep=' ') if 'quat' in b[i + 1] else np.array([1, 0, 0, 0.])
                # print(pos, quat)
                T = T * rot.pose2T(np.concatenate([pos, quat]))
                # print(T)
                T = T * rotation(q[i], j[i]['axis'])
                # print(T)

            s = np.fromstring(site[a]['pos'], sep=' ')
            Ts = sy.eye(4)
            Ts[:3, 3] = s
            T = T * Ts
            T = sy.simplify(T)
            T_list.append(T)
        self.T_list = T_list
        self.T_fingers = T0

    def forward_kine(self, q: np.ndarray, quat=True):
        """
        forward kinematics for all fingers
        :param quat: return quaternion or rotation matrix
        :param q: numpy array  (20,)
        :return: x:  poses for all fingertips (site)
        """
        assert len(q) == self.dof

        poses = [self.fk[i](q[i * 4: 4 + i * 4]) for i in range(self.n)]
        if quat:
            poses = [rot.T2pose(pose) for pose in poses]

        # for i in range(self.n):
        #     pose = self.fk[i](q[i * 4: 4 + i * 4])  # (4,4)
        #     if quat:
        #         pose = rot.T2pose(pose)  # (7, )
        #     poses.append(pose)
        return poses

    def get_jac_bad(self, q):
        """
         get the position jacobian for all fingertips

         !!!!!!!!!!!!!warning, this would be too slow if do the subs online
         please use the lambdify function version

        """
        subs_dic = subs_value([self.qt, self.qi, self.qm, self.qr, self.ql], [q[:4], q[4:8], q[8:12], q[12:16], q[16:]])
        jac_list = []
        for i in range(self.n):
            jac_tmp = self.jac_syms[i].subs(subs_dic)  # numeric value
            jac_list.append(np.array(jac_tmp))
        return jac_list

    def get_jac(self, q: np.ndarray):
        """
         get the position jacobian for all fingertips

        """
        jac_list = []
        for i in range(self.n):
            jac_tmp = self.jac[i](q[i * 4: 4 + i * 4])  # numeric value
            jac_list.append(np.array(jac_tmp))
        return jac_list


def rotation(theta, axis):
    """
    Given the rotation axis and angle, calculate the transformation matrix.
    :param theta:
    :param axis: (3, )
    :return:
    """
    if type(axis) == str:
        axis = np.fromstring(axis, dtype=np.int8, sep=' ')
    T = sy.eye(4)
    tmp = np.sum(axis)
    c1 = sy.cos(theta * tmp)
    s1 = sy.sin(theta * tmp)
    if np.abs(axis[0]):
        T[:3, :3] = sy.Matrix([[1, 0, 0],
                               [0, c1, -s1],
                               [0, s1, c1]])
        return T
    if np.abs(axis[1]):
        T[:3, :3] = sy.Matrix([[c1, 0, s1],
                               [0, 1, 0],
                               [-s1, 0, c1]])
        return T
    if np.abs(axis[2]):
        T[:3, :3] = sy.Matrix([[c1, -s1, 0],
                               [s1, c1, 0],
                               [0, 0, 1]])
        return T


def subs_value(vars, vars_value):
    subs_dic = {}
    for i in range(len(vars)):
        for j in range(len(vars[i])):
            subs_dic.update({vars[i][j]: vars_value[i][j]})
    return subs_dic


if __name__ == '__main__':
    ### fk and jac tests
    finger = Robot(path_suffix='../', finger_num=6)
    poses = finger.forward_kine(np.zeros(finger.dof))
    print(poses)
    jacs = finger.get_jac(np.zeros(finger.dof))
    print(jacs)
