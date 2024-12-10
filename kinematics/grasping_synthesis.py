"""

"""
import copy

import numpy as np
from kinematics.hand_sym import rotation, subs_value
import tools.rotations as rot
import sympy as sy
import os
import pickle
import mujoco
from mujoco import viewer
from gait_test.hand_generator_from_genes import crawling_hand
from controller_full import Robot
import time

from scipy.optimize import minimize
import xml.etree.ElementTree as ET
import trimesh


class grasping(object):
    def __init__(self, path_prefix=''):
        pass

        # define fk parameters

        self.q_list = ['q_0', 'q_1', 'q_2', 'q_3', 'q_4']  # finger placement, joints of robot
        self.q = sy.symbols('q_:5')  # finger placement angle, joints of robot
        self.axes = [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
        self.dof = len(self.q)
        self.l = sy.symbols('l_:5')  # link lengths
        self.r = sy.symbols('r_:2')  # radius for capsule and sphere
        self.T_all = self.fk()  # finger tip 4x4 matrix

        # the sphere as an object to be grasped
        self.x = sy.symbols('x_:3')

        # for second finger:
        self.p = sy.symbols('p_:5')  # joint symbols for second finger
        self.m = sy.symbols('m_:5')  # link lengths for second finger
        self.p2q = dict((self.q[tmp], self.p[tmp]) for tmp in range(self.dof))
        self.l2m = dict((self.l[tmp], self.m[tmp]) for tmp in range(self.dof))
        # self.ql2qm = self.p2q | self.l2m
        self.ql2qm = self.p2q.copy()
        self.ql2qm.update(self.l2m)

        self.T_all_j = [T.subs(self.ql2qm) for T in self.T_all]  # for the second finger
        self.M = sy.Matrix(self.x)

        self.N1_N2 = {}
        for grasping_link_index in [[1, 1], [2, 2], [3, 3], [4, 4], [3, 4], [2, 4], [4, 2]]:
            i = grasping_link_index[0]
            j = grasping_link_index[1]
            N1_N2_name = path_prefix + 'N1_N2_' + str(i) + str(j) + '.txt'
            if os.path.isfile(N1_N2_name):
                with open(N1_N2_name, 'rb') as f:
                    N1_N2 = pickle.load(f)
            else:
                ### get A B C D points
                # A = self.T_all_j[i - 1][:3, 3]
                # B = self.T_all_j[i][:3, 3]
                if grasping_link_index == [2, 4] or grasping_link_index == [4, 2]:
                    A = self.T_all_j[i - 1][:3, 3]
                    B = self.T_all_j[i][:3, 3]
                else:
                    A = self.T_all[i - 1][:3, 3]
                    B = self.T_all[i][:3, 3]
                C = self.T_all_j[j - 1][:3, 3]
                D = self.T_all_j[j][:3, 3]

                ### get N1, N2 points, the perpendicular intersections from M to AB, and M to CD
                # AB and CD are the links to grasp the sphere, N1 and N2 are contact points
                n1 = sy.symbols('n_1')
                Eq_n1 = (B - A).dot(self.M - (A + (B - A) * n1))
                n1_val = sy.solve(Eq_n1, n1, dict=False)[0]
                N1 = A + (B - A) * n1_val

                n2 = sy.symbols('n_2')
                Eq_n2 = (D - C).dot(self.M - (C + (D - C) * n2))
                n2_val = sy.solve(Eq_n2, n2, dict=False)[0]
                N2 = C + (D - C) * n2_val
                N1_N2 = [N1, N2]
                with open(N1_N2_name, 'wb') as f:
                    pickle.dump(N1_N2, f)
                    print(N1_N2_name + ' has been saved to file.')

            self.N1_N2[tuple(grasping_link_index)] = N1_N2

        # fk positions
        self.T_all_ = [sy.lambdify([self.q + self.l], tmp) for tmp in self.T_all]
        self.T_all_j_ = [sy.lambdify([self.p + self.m], tmp) for tmp in self.T_all_j]

        self.ABCD = None

    def fk(self):
        T = sy.eye(4)
        T_all = []
        for i in range(self.dof):
            T = T * rotation(self.q[i], self.axes[i])
            T_translation = sy.eye(4)
            T_translation[:3, 3] = [self.l[i], 0, 0]
            T = T * T_translation  # fingertip pose
            # T_translation_c = sy.eye(4)
            # T_translation_c[:3, 3] = [-self.l[i] / 2, 0, 0]
            # T_c = T * T_translation_c  # link center
            T_all.append(T)
        return T_all

    # def subs_value(self):
    #     pass

    def generate_triple_grasp(self, l_, m_, r_, q0, p0, q_init=None, x_init=None, grasping_link_index=(4, 4)):

        self.x = sy.symbols('x_:9')
        self.M = [sy.Matrix(self.x[:3]), sy.Matrix(self.x[3:6]), sy.Matrix(self.x[6:])]

        # for first grasp

        c = []
        dis_sum = 0
        for k in range(2):
            if k:  # k = 1
                A = self.T_all_j[1][:3, 3]
                B = self.T_all_j[2][:3, 3]
                C = self.T_all_j[3][:3, 3]
                D = self.T_all_j[4][:3, 3]
                N1 = self.N1_N2[(2, 2)][1]
                N2 = self.N1_N2[(4, 4)][1]
                M2BC = self.N1_N2[(3, 3)][1]
            else:
                A = self.T_all[1][:3, 3]
                B = self.T_all[2][:3, 3]
                C = self.T_all[3][:3, 3]
                D = self.T_all[4][:3, 3]
                N1 = self.N1_N2[(2, 2)][0]
                N2 = self.N1_N2[(4, 4)][0]
                M2BC = self.N1_N2[(3, 3)][0]
            # constraints of acute angles
            c1 = (A - B).dot(A - self.M[k])  # to be positive
            c2 = (B - A).dot(B - self.M[k])
            c3 = (C - D).dot(C - self.M[k])
            c4 = (D - C).dot(D - self.M[k])

            # equality constraints. Keep contacts
            # eq_1 = sy.sqrt((self.M[k] - N1).dot(self.M[k] - N1)) - self.r[0] - r_[k + 1]
            eq_1 = sy.sqrt((self.M[k] - N1).dot(self.M[k] - N1)) - self.r[0] - r_[k]
            # eq_2 = sy.sqrt((self.M[k] - N2).dot(self.M[k] - N2)) - self.r[0] - r_[k + 1]
            eq_2 = sy.sqrt((self.M[k] - N2).dot(self.M[k] - N2)) - self.r[0] - r_[k]

            # contact angle, N1_M_N2 > 90 degrees
            c5 = - (self.M[k] - N1).dot(self.M[k] - N2)  # to be positive

            # friction cone
            mu = 0.5
            cos_theta = np.cos(np.arctan(mu))
            c6 = (self.M[k] - N1).dot(N2 - N1) / sy.sqrt((self.M[k] - N1).dot(self.M[k] - N1)) / sy.sqrt(
                (N2 - N1).dot(N2 - N1)) - cos_theta
            c7 = (self.M[k] - N2).dot(N1 - N2) / sy.sqrt((self.M[k] - N2).dot(self.M[k] - N2)) / sy.sqrt(
                (N2 - N1).dot(N2 - N1)) - cos_theta

            c_collision = sy.sqrt((self.M[k] - M2BC).dot(self.M[k] - M2BC)) - self.r[0] - self.r[
                1]  # to be positive, BC has no collision with the object

            c += [c1, c2, c3, c4, c6, c7, c_collision]
            # dis_sum += (eq_1 ** 2 + eq_2 ** 2) * 2000 - c5
            dis_sum += (eq_1 ** 2 + eq_2 ** 2) * 5000
        i = grasping_link_index[0]
        j = grasping_link_index[1]

        A = self.T_all[i - 1][:3, 3]
        B = self.T_all[i][:3, 3]
        C = self.T_all_j[j - 1][:3, 3]
        D = self.T_all_j[j][:3, 3]
        N1 = self.N1_N2[(2, 2)][0]
        N2 = self.N1_N2[(2, 2)][1]
        # constraints of acute angles
        c1 = (A - B).dot(A - self.M[2])  # to be positive
        c2 = (B - A).dot(B - self.M[2])
        c3 = (C - D).dot(C - self.M[2])
        c4 = (D - C).dot(D - self.M[2])

        # equality constraints. Keep contacts
        eq_1 = sy.sqrt((self.M[2] - N1).dot(self.M[2] - N1)) - self.r[0] - r_[3]
        eq_2 = sy.sqrt((self.M[2] - N2).dot(self.M[2] - N2)) - self.r[0] - r_[3]

        # contact angle, N1_M_N2 > 90 degrees
        c5 = - (self.M[2] - N1).dot(self.M[2] - N2)  # to be positive
        #
        # ## cost function
        # g = - c5 / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt((self.M - N2).dot(self.M - N2))  # to be minimized, should be negative

        # friction cone
        mu = 0.5
        cos_theta = np.cos(np.arctan(mu))
        c6 = (self.M[2] - N1).dot(N2 - N1) / sy.sqrt((self.M[2] - N1).dot(self.M[2] - N1)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta
        c7 = (self.M[2] - N2).dot(N1 - N2) / sy.sqrt((self.M[2] - N2).dot(self.M[2] - N2)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta

        c += [c1, c2, c3, c4, c6, c7]
        c += [sy.sqrt((self.M[2] - self.M[1]).dot(self.M[2] - self.M[1])) - r_[2] - 2 * r_[
            3]]  # ball 2 and ball 1 collision
        c += [sy.sqrt((self.M[2] - self.M[0]).dot(self.M[2] - self.M[0])) - 2 * r_[3] - r_[
            1]]  # ball 2 and ball 0 collision
        c += [
            sy.sqrt((self.M[1] - self.M[0]).dot(self.M[1] - self.M[0])) - r_[2] - r_[1]]  # ball 1 and ball 0 collision
        c_collision = []
        for i in range(1, self.dof):
            A1 = self.T_all[i][:3, 3]
            A2 = self.T_all_j[i][:3, 3]
            c_i = sy.sqrt((A1 - A2).dot(A1 - A2)) - 2 * self.r[0]
            c_collision.append(c_i)
        c += c_collision

        # collision between 1st link and 4th link
        # c_collision = []
        # A1 = self.T_all[0][:3, 3]
        # A2 = self.T_all[4][:3, 3]
        # c_i = sy.sqrt((A1 - A2).dot(A1 - A2)) - 4 * self.r[0]
        # c_collision.append(c_i)
        # A1 = self.T_all_j[0][:3, 3]
        # A2 = self.T_all_j[4][:3, 3]
        # c_i = sy.sqrt((A1 - A2).dot(A1 - A2)) - 4 * self.r[0]
        # c_collision.append(c_i)

        # c += c_collision

        qpx = self.q[1:] + self.p[1:] + self.x
        subs_dict = subs_value([self.q, self.p, self.x, self.l, self.m, self.r],
                               [(q0,) + self.q[1:], (p0,) + self.p[1:], self.x, l_, m_, [r_[0], r_[3]]])
        center = ((A + C) / 2 + (B + D) / 2) / 2
        center_ = sy.lambdify([qpx[:8]], center.subs(subs_dict))  # for initial value of x

        c_ = [sy.lambdify([qpx], tmp.subs(subs_dict)) for tmp in c]
        dis_sum += (eq_1 ** 2 + eq_2 ** 2) * 1000 - c5
        dis_sum_ = sy.lambdify([qpx], dis_sum.subs(subs_dict))

        dis_sum_jac = sy.Matrix([dis_sum.subs(subs_dict)]).jacobian(qpx)
        dis_sum_jac_ = sy.lambdify([qpx], dis_sum_jac)

        # optimization
        cons = ()
        for k in c_:
            cons += ({'type': 'ineq', 'fun': k},)
        # for k in eqs_:
        #     cons += ({'type': 'eq', 'fun': k},)

        # bounds
        joint_bnds = (
            (-np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2), (-np.pi / 2 * 1.25, np.pi / 2 * 1.25),
            (-np.pi / 2 * 1.25, np.pi / 2 * 1.25))
        xyz_bounds = ((0, l_[0] * 5), (0, l_[0] * 5), (-l_[0] * 5, l_[0] * 5))
        bnds = joint_bnds + joint_bnds + xyz_bounds + xyz_bounds + xyz_bounds

        if q_init is None:
            q_init = np.ones(8) * 0.1
            x_init = np.ones(6) * 0.1
        else:
            assert len(q_init) == 8
        x0 = center_(q_init).flatten()
        x0 = np.concatenate([q_init, x_init, x0])
        res = minimize(dis_sum_, x0, jac=dis_sum_jac_, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': 1000, 'disp': 0})  # 0.45 s with a good initial guess
        return res

    def generate_single_grasp(self, m_, r_, p0, check_con=False):
        A = self.T_all_j[1][:3, 3]
        B = self.T_all_j[2][:3, 3]
        C = self.T_all_j[3][:3, 3]
        D = self.T_all_j[4][:3, 3]
        N1 = self.N1_N2[(4, 2)][1]
        N2 = self.N1_N2[(2, 4)][1]
        M2BC = self.N1_N2[(3, 3)][1]
        # constraints of acute angles
        c1 = (A - B).dot(A - self.M)  # to be positive
        c2 = (B - A).dot(B - self.M)
        c3 = (C - D).dot(C - self.M)
        c4 = (D - C).dot(D - self.M)

        # equality constraints. Keep contacts
        eq_1 = sy.sqrt((self.M - N1).dot(self.M - N1)) - self.r[0] - self.r[1]
        eq_2 = sy.sqrt((self.M - N2).dot(self.M - N2)) - self.r[0] - self.r[1]

        # contact angle, N1_M_N2 > 90 degrees
        c5 = - (self.M - N1).dot(self.M - N2)  # to be positive
        #
        # ## cost function
        # g = - c5 / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt((self.M - N2).dot(self.M - N2))  # to be minimized, should be negative

        # friction cone
        mu = 0.5
        cos_theta = np.cos(np.arctan(mu))
        c6 = (self.M - N1).dot(N2 - N1) / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta
        c7 = (self.M - N2).dot(N1 - N2) / sy.sqrt((self.M - N2).dot(self.M - N2)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta

        ## self-collision avoidance. Todo, need to add more,
        c_collision = []
        for i in range(self.dof):
            A1 = self.T_all[i][:3, 3]
            A2 = self.T_all_j[i][:3, 3]
            c_i = sy.sqrt((A1 - A2).dot(A1 - A2)) - 2 * self.r[0]
            c_collision.append(c_i)

        c = [c1, c2, c3, c4, c6, c7] + c_collision
        subs_dict = subs_value([self.p, self.x, self.m, self.r],
                               [(p0,) + self.p[1:], self.x, m_, r_])

        # eqs = [eq_1, eq_2]
        # g_ = sy.lambdify([qpx], g.subs(subs_dict))
        # eqs_ = [sy.lambdify([qpx], tmp.subs(subs_dict)) for tmp in eqs]

        # cost function, these two take 2 s
        c_collision = sy.sqrt((self.M - M2BC).dot(self.M - M2BC)) - self.r[0] * 2 - self.r[
            1]  # to be positive, BC has no collision with the object

        c = [c1, c2, c3, c4, c6, c7, c_collision]
        # c = [c1, c2, c3, c4, c6, c7]
        # subs_dict = subs_value([self.q, self.x, self.l, self.m, self.r],
        #                        [(q0,) + self.q[1:], self.x, l_, l_, r_])

        px = self.p[1:] + self.x
        c_ = [sy.lambdify([px], tmp.subs(subs_dict)) for tmp in c]
        dis_sum = (eq_1 ** 2 + eq_2 ** 2) * 1000
        # dis_sum = (eq_1 ** 2 + eq_2 ** 2) * 100000
        dis_sum_ = sy.lambdify([px], dis_sum.subs(subs_dict))

        dis_sum_jac = sy.Matrix([dis_sum.subs(subs_dict)]).jacobian(px)
        dis_sum_jac_ = sy.lambdify([px], dis_sum_jac)

        center = ((A + C) / 2 + (B + D) / 2) / 2
        center_ = sy.lambdify([px[:4]], center.subs(subs_dict))  # for initial value of x

        # optimization
        cons = ()
        for k in c_:
            cons += ({'type': 'ineq', 'fun': k},)
        # for k in eqs_:
        #     cons += ({'type': 'eq', 'fun': k},)

        # bounds
        joint_bnds = (
            (-np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2), (-np.pi / 2 * 1.25, np.pi / 2 * 1.25),
            (-np.pi / 2 * 1.25, np.pi / 2 * 1.25))
        xyz_bounds = ((0, m_[0] * 5), (0, m_[0] * 5), (-m_[0] * 5, m_[0] * 5))
        bnds = joint_bnds + xyz_bounds

        q_init = np.array([0, 0, np.pi / 2, np.pi / 2])
        x0 = center_(q_init).flatten()
        x0 = np.concatenate([q_init, x0])
        res = minimize(dis_sum_, x0, jac=dis_sum_jac_, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': 1000, 'disp': 0})  # 0.45 s with a good initial guess
        if check_con:
            cons = []
            for i in c_ + [dis_sum_]:
                cons.append(i(res.x))

            return res, cons
        else:
            return res

    def generate_2nd_grasp(self, l_, m_, r_, q0, p0, q_fixed, x_list, grasping_link_index=(4, 4), single=False,
                           check_cons=False):
        """

        :param l_: (5, ), [palm radius, link lengths * 4] for the 1st finger
        :param m_: (5, ), [palm radius, link lengths * 4] for the 2nd finger
        :param r_: (2, ), [radius of capsule, radius of sphere]
        :param q0: float, angle for the 1st finger placement
        :param p0: float, angle for the 2nd finger placement
        :param q_fixed: fixed joints
        :param x_list: collision-free with other objects
        :param grasping_link_index: the index of links that used for grasping
        :param single: bool, grasp a single object using only one finger
        :return:
        """
        ## input:  i-th link for grasping,
        #          link-length, radius of the capsule and sphere, q0=0, p0,
        ## output: joint configuration

        i = grasping_link_index[0]
        j = grasping_link_index[1]
        A = self.T_all[i - 1][:3, 3]
        B = self.T_all[i][:3, 3]
        C = self.T_all_j[j - 1][:3, 3]
        D = self.T_all_j[j][:3, 3]
        N1 = self.N1_N2[grasping_link_index][0]
        N2 = self.N1_N2[grasping_link_index][1]

        # constraints of acute angles
        c1 = (A - B).dot(A - self.M)  # to be positive
        c2 = (B - A).dot(B - self.M)
        c3 = (C - D).dot(C - self.M)
        c4 = (D - C).dot(D - self.M)

        # equality constraints. Keep contacts
        eq_1 = sy.sqrt((self.M - N1).dot(self.M - N1)) - self.r[0] - self.r[1]
        eq_2 = sy.sqrt((self.M - N2).dot(self.M - N2)) - self.r[0] - self.r[1]

        # contact angle, N1_M_N2 > 90 degrees
        c5 = - (self.M - N1).dot(self.M - N2)  # to be positive
        #
        # ## cost function
        # g = - c5 / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt((self.M - N2).dot(self.M - N2))  # to be minimized, should be negative

        # friction cone
        mu = 0.5
        cos_theta = np.cos(np.arctan(mu))
        c6 = (self.M - N1).dot(N2 - N1) / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta
        c7 = (self.M - N2).dot(N1 - N2) / sy.sqrt((self.M - N2).dot(self.M - N2)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta

        ## self-collision avoidance. Todo, need to add more,
        c_collision = []
        for i in range(1, self.dof):
            A1 = self.T_all[i][:3, 3]
            A2 = self.T_all_j[i][:3, 3]
            c_i = sy.sqrt((A1 - A2).dot(A1 - A2)) - 2 * self.r[0]
            c_collision.append(c_i)

        for x_obj in x_list:
            pos = x_obj[0].reshape(3, 1)

            dis_obj = sy.sqrt((self.M - pos).dot(self.M - pos)) - r_[1] - x_obj[1] - 0.02  # radius of the fixed object
            c_collision.append(dis_obj)

        c = [c1, c2, c3, c4, c6, c7] + c_collision
        # qpx = self.q[1:] + self.p[1:] + self.x
        px = self.p[1:] + self.x
        subs_dict = subs_value([self.q, self.p, self.x, self.l, self.m, self.r],
                               [(q0,) + tuple(q_fixed), (p0,) + self.p[1:], self.x, l_, m_, r_])

        # eqs = [eq_1, eq_2]
        # g_ = sy.lambdify([qpx], g.subs(subs_dict))
        # eqs_ = [sy.lambdify([qpx], tmp.subs(subs_dict)) for tmp in eqs]

        # cost function, these two take 2 s
        c_ = [sy.lambdify([px], tmp.subs(subs_dict)) for tmp in c]
        dis_sum = (eq_1 ** 2 + eq_2 ** 2) * 100
        dis_sum_ = sy.lambdify([px], dis_sum.subs(subs_dict))

        dis_sum_jac = sy.Matrix([dis_sum.subs(subs_dict)]).jacobian(px)
        dis_sum_jac_ = sy.lambdify([px], dis_sum_jac)

        center = ((A + C) / 2 + (B + D) / 2) / 2
        center_ = sy.lambdify([self.p[1:]], center.subs(subs_dict))  # for initial value of x

        # optimization
        cons = ()
        for k in c_:
            cons += ({'type': 'ineq', 'fun': k},)
        # for k in eqs_:
        #     cons += ({'type': 'eq', 'fun': k},)

        # bounds
        joint_bnds = (
            (-np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2), (-np.pi / 2 * 1.25, np.pi / 2 * 1.25),
            (-np.pi / 2 * 1.25, np.pi / 2 * 1.25))
        xyz_bounds = ((0, l_[0] * 5), (0, l_[0] * 5), (-l_[0] * 5, l_[0] * 5))
        bnds = joint_bnds + xyz_bounds

        q_init = np.ones(4) * 0.6
        # q_init[4] = -0.5
        # q_init[5] = -0.5
        x0 = center_(q_init).flatten()
        print('x0', x0)
        x0 = np.concatenate([q_init, x0])
        res = minimize(dis_sum_, x0, jac=dis_sum_jac_, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': 1000, 'disp': 0})  # 0.45 s with a good initial guess

        if check_cons:
            self.ABCD = [sy.lambdify([px], i.subs(subs_dict)) for i in [A, B, C, D, N1, N2]]
            self.ABCD = [i(res.x) for i in self.ABCD]
            cons = []
            for i in c_ + [dis_sum_]:
                cons.append(i(res.x))

            return res, cons
        else:
            return res

    def generate_grasp(self, l_, m_, r_, q0, p0, grasping_link_index=(4, 4), single=False, check_cons=False):
        """

        :param l_: (5, ), [palm radius, link lengths * 4] for the 1st finger
        :param m_: (5, ), [palm radius, link lengths * 4] for the 2nd finger
        :param r_: (2, ), [radius of capsule, radius of sphere]
        :param q0: float, angle for the 1st finger placement
        :param p0: float, angle for the 2nd finger placement
        :param grasping_link_index: the index of links that used for grasping
        :param single: bool, grasp a single object using only one finger
        :return:
        """
        ## input:  i-th link for grasping,
        #          link-length, radius of the capsule and sphere, q0=0, p0,
        ## output: joint configuration

        i = grasping_link_index[0]
        j = grasping_link_index[1]
        if single:
            A = self.T_all_j[1][:3, 3]
            B = self.T_all_j[2][:3, 3]
            C = self.T_all_j[3][:3, 3]
            D = self.T_all_j[4][:3, 3]
            N1 = self.N1_N2[(2, 2)][1]
            N2 = self.N1_N2[(4, 4)][1]

            M2BC = self.N1_N2[(3, 3)][1]
        else:
            A = self.T_all[i - 1][:3, 3]
            B = self.T_all[i][:3, 3]
            C = self.T_all_j[j - 1][:3, 3]
            D = self.T_all_j[j][:3, 3]
            N1 = self.N1_N2[grasping_link_index][0]
            N2 = self.N1_N2[grasping_link_index][1]

        # constraints of acute angles
        c1 = (A - B).dot(A - self.M)  # to be positive
        c2 = (B - A).dot(B - self.M)
        c3 = (C - D).dot(C - self.M)
        c4 = (D - C).dot(D - self.M)

        # equality constraints. Keep contacts
        eq_1 = sy.sqrt((self.M - N1).dot(self.M - N1)) - self.r[0] - self.r[1]
        eq_2 = sy.sqrt((self.M - N2).dot(self.M - N2)) - self.r[0] - self.r[1]

        # contact angle, N1_M_N2 > 90 degrees
        c5 = - (self.M - N1).dot(self.M - N2)  # to be positive
        #
        # ## cost function
        # g = - c5 / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt((self.M - N2).dot(self.M - N2))  # to be minimized, should be negative

        # friction cone
        mu = 0.5
        cos_theta = np.cos(np.arctan(mu))
        c6 = (self.M - N1).dot(N2 - N1) / sy.sqrt((self.M - N1).dot(self.M - N1)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta
        c7 = (self.M - N2).dot(N1 - N2) / sy.sqrt((self.M - N2).dot(self.M - N2)) / sy.sqrt(
            (N2 - N1).dot(N2 - N1)) - cos_theta

        ## self-collision avoidance. Todo, need to add more,
        c_collision = []
        for i in range(1, self.dof):
            A1 = self.T_all[i][:3, 3]
            A2 = self.T_all_j[i][:3, 3]
            c_i = sy.sqrt((A1 - A2).dot(A1 - A2)) - 2 * self.r[0]
            c_collision.append(c_i)

        c = [c1, c2, c3, c4, c6, c7] + c_collision
        qpx = self.q[1:] + self.p[1:] + self.x
        subs_dict = subs_value([self.q, self.p, self.x, self.l, self.m, self.r],
                               [(q0,) + self.q[1:], (p0,) + self.p[1:], self.x, l_, m_, r_])

        # eqs = [eq_1, eq_2]
        # g_ = sy.lambdify([qpx], g.subs(subs_dict))
        # eqs_ = [sy.lambdify([qpx], tmp.subs(subs_dict)) for tmp in eqs]

        # cost function, these two take 2 s
        if single:
            c_collision = sy.sqrt((self.M - M2BC).dot(self.M - M2BC)) - self.r[0] * 2 - self.r[
                1]  # to be positive, BC has no collision with the object

            c = [c1, c2, c3, c4, c6, c7, c_collision]
            subs_dict = subs_value([self.p, self.x, self.l, self.m, self.r],
                                   [(p0,) + self.p[1:], self.x, l_, m_, r_])

            px = self.p[1:] + self.x
            c_ = [sy.lambdify([px], tmp.subs(subs_dict)) for tmp in c]
            dis_sum = (eq_1 ** 2 + eq_2 ** 2) * 1000 - c5
            dis_sum_ = sy.lambdify([px], dis_sum.subs(subs_dict))

            dis_sum_jac = sy.Matrix([dis_sum.subs(subs_dict)]).jacobian(px)
            dis_sum_jac_ = sy.lambdify([px], dis_sum_jac)

            center = ((A + C) / 2 + (B + D) / 2) / 2
            center_ = sy.lambdify([px[:4]], center.subs(subs_dict))  # for initial value of x

            # optimization
            cons = ()
            for k in c_:
                cons += ({'type': 'ineq', 'fun': k},)
            # for k in eqs_:
            #     cons += ({'type': 'eq', 'fun': k},)

            # bounds
            joint_bnds = (
                (-np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2), (-np.pi / 2 * 1.25, np.pi / 2 * 1.25),
                (-np.pi / 2 * 1.25, np.pi / 2 * 1.25))
            xyz_bounds = ((0, l_[0] * 5), (0, l_[0] * 5), (-l_[0] * 5, l_[0] * 5))
            bnds = joint_bnds + xyz_bounds

            q_init = np.array([0, 0, np.pi / 2, np.pi / 2])
            x0 = center_(q_init).flatten()
            x0 = np.concatenate([q_init, x0])
            res = minimize(dis_sum_, x0, jac=dis_sum_jac_, method='SLSQP', bounds=bnds, constraints=cons,
                           options={'maxiter': 1000, 'disp': 0})  # 0.45 s with a good initial guess

        else:
            c_ = [sy.lambdify([qpx], tmp.subs(subs_dict)) for tmp in c]
            dis_sum = (eq_1 ** 2 + eq_2 ** 2) * 100 - c5
            dis_sum_ = sy.lambdify([qpx], dis_sum.subs(subs_dict))

            dis_sum_jac = sy.Matrix([dis_sum.subs(subs_dict)]).jacobian(qpx)
            dis_sum_jac_ = sy.lambdify([qpx], dis_sum_jac)

            center = ((A + C) / 2 + (B + D) / 2) / 2
            center_ = sy.lambdify([qpx[:8]], center.subs(subs_dict))  # for initial value of x

            # optimization
            cons = ()
            for k in c_:
                cons += ({'type': 'ineq', 'fun': k},)
            # for k in eqs_:
            #     cons += ({'type': 'eq', 'fun': k},)

            # bounds
            joint_bnds = (
                (-np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2), (-np.pi / 2 * 1.25, np.pi / 2 * 1.25),
                (-np.pi / 2 * 1.25, np.pi / 2 * 1.25))
            xyz_bounds = ((0, l_[0] * 5), (0, l_[0] * 5), (-l_[0] * 5, l_[0] * 5))
            bnds = joint_bnds + joint_bnds + xyz_bounds

            q_init = np.ones(8) * 0.6
            q_init[4] = -0.5
            # q_init[5] = -0.5
            x0 = center_(q_init).flatten()
            # print('x0', x0)
            x0 = np.concatenate([q_init, x0])
            res = minimize(dis_sum_, x0, jac=dis_sum_jac_, method='SLSQP', bounds=bnds, constraints=cons,
                           options={'maxiter': 1000, 'disp': 0})  # 0.45 s with a good initial guess

        if check_cons:
            self.ABCD = [sy.lambdify([qpx], i.subs(subs_dict)) for i in [A, B, C, D, N1, N2]]
            self.ABCD = [i(res.x) for i in self.ABCD]
            cons = []
            for i in c_ + [dis_sum_]:
                cons.append(i(res.x))

            return res, cons
        else:
            return res

    def generate_robot_pc(self, q: np.ndarray, link_length, theta: float, palm_r: float, link_sampling_interval=0.002,
                          finger_index=[0, 1]) -> list:
        """
        # Given q and link lengths, generate point cloud of the robot
        :param q: (8,), joint positions of two fingers
        :param link_length: (2,), length of links for the two fingers respectively
        :param theta:  the angle of the second finger,should be np.pi/4 or np.pi/2
        :param palm_r: the radius of palm
        :param link_sampling_interval: sampling interval
        :return: (n, 3)
        """
        ql = [0] + list(q[:4]) + [palm_r] + [link_length[0]] * 4
        pm = [theta] + list(q[4:8]) + [palm_r] + [link_length[1]] * 4
        finger_1 = [tmp(ql)[:3, 3] for tmp in self.T_all_]  # (5, )  positions of 5 point
        finger_2 = [tmp(pm)[:3, 3] for tmp in self.T_all_j_]
        fingers = [finger_1, finger_2]
        # fingers = [fingers[i] for i in finger_index]
        fingers_pc = [[], []]
        # finger_1_pc = []
        # finger_2_pc = []
        for k, finger in enumerate(fingers):
            for i in range(len(finger) - 1):
                x1 = finger[i]
                x2 = finger[i + 1]
                num = int(np.linalg.norm(x1 - x2) / link_sampling_interval)
                robot_pc_i = np.linspace(x1, x2, num)  # (num, 3), sample points for each link
                fingers_pc[k].append(robot_pc_i)

        return [np.vstack(fingers_pc[0]), np.vstack(fingers_pc[1])]


class data_driven_grasp(object):
    def __init__(self, link_length=[0.02, 0.04], theta=np.pi / 4, palm_r=0.05, mu=0.5, single_obj_test=0,
                 capsule_r=0.01):
        self.link_length = link_length
        self.theta = theta
        self.palm_r = palm_r
        self.capsule_r = capsule_r
        self.g1 = grasping(path_prefix='../kinematics/')

        # load object mesh and simplify
        self.obj_names = ['apple', 'bottle', 'fork', 'lemon', 'strawberry', 'can']
        obj_path = '../descriptions/objs/'
        obj_meshes = []
        num_faces = 300
        for obj_name in self.obj_names:
            if not os.path.isfile(obj_path + 'simplified_mesh/' + obj_name + '.obj'):
                print('generate simplified mesh for', obj_name)
                mesh = trimesh.load_mesh(obj_path + obj_name + '/textured.obj')
                mesh_simplified = mesh.simplify_quadric_decimation(300)
                mesh_simplified = mesh_simplified.convex_hull
                print(mesh_simplified.is_watertight)
                mesh_simplified.export(obj_path + 'simplified_mesh/' + obj_name + '.obj')

        self.meshes = {obj_name: trimesh.load_mesh(obj_path + 'simplified_mesh/' + obj_name + '.obj') for obj_name in
                       self.obj_names}
        self.single_obj_test = single_obj_test

        self.contact_distance = []
        self.mu = mu
        self.cos_theta = np.cos(np.arctan(mu))
        self.mesh = trimesh.Trimesh.copy(self.meshes[self.obj_names[self.single_obj_test]])
        self.fingers_pc = []

        self.contact_normals = []
        self.contact_points = []

    def cost_fun(self, x):
        """
        using two fingers for making contact
        :param x: [q, x], 8+7 dim
        :return:
        """
        q = x[:8]
        pose = x[8:15]
        pose[3:] = pose[3:] / np.linalg.norm(pose[3:])
        self.fingers_pc = self.g1.generate_robot_pc(q, self.link_length, self.theta, self.palm_r, finger_index=[0, 1])
        self.mesh = trimesh.Trimesh.copy(self.meshes[self.obj_names[self.single_obj_test]])
        #  apply transformation
        # pose = np.array([0.06, 0.05, -0.01, 1, 0, 0, 0])
        T = rot.pose2T(pose)
        self.mesh.apply_transform(T)
        tri_obj = trimesh.proximity.ProximityQuery(self.mesh)

        # contact distances
        self.contact_distance = []
        for i in range(2):
            dis_tmp = tri_obj.signed_distance(self.fingers_pc[i])
            dis_0 = np.min(-dis_tmp)  # positive means collision-free, get the minimum distance
            self.contact_distance.append(dis_0)

        # cost = self.contact_distance[0] - self.capsule_r+ self.contact_distance[1] - self.capsule_r
        cost = self.contact_distance[0] ** 2 + self.contact_distance[1] ** 2

        return cost

    def constraints_contact(self, x):
        return self.contact_distance[0] - self.capsule_r * 0.5, self.contact_distance[1] - self.capsule_r * 0.5

    def constraints_friction(self, x):
        """

        :param x:
        :return:
        """
        self.contact_points = []
        self.contact_normals = []
        for i in range(2):
            closest, distance, triangle_id = trimesh.proximity.closest_point(self.mesh, self.fingers_pc[i])
            # closest ((m, 3) float) – Closest point on triangles for each point
            # distance ((m,) float) – Distance to mesh. always positive !!
            # triangle_id ((m,) int) – Index of triangle containing closest point
            k = np.where(distance - np.abs(self.contact_distance[i]) == 0)  # the index where is closest to the robot
            # print(k[0][0])
            # contact_point = np.array(self.mesh.vertices[k[0][0]])  # get the contact point on the object mesh
            contact_point = np.array(closest[k[0][0], :])  # get the closest point
            self.contact_points.append(contact_point)

            triangle_id_contact = triangle_id[k][0]
            # triangle = self.mesh.triangles[triangle_id_contact, :, :]
            # normal = np.cross(triangle[0, :] - triangle[1, :], triangle[2, :] - triangle[1, :])
            # normal = normal / np.linalg.norm(normal)
            normal = self.mesh.face_normals[triangle_id_contact, :]
            self.contact_normals.append(normal)

        n1 = self.contact_points[1] - self.contact_points[0]
        n1 = n1 / (np.linalg.norm(n1) + 1e-12)
        c1 = n1.dot(self.contact_normals[1]) - self.cos_theta
        c2 = (-n1).dot(self.contact_normals[0]) - self.cos_theta
        return c1, c2

    def constraints_collision_free(self, x):
        pass


class mujoco_sim(object):
    def __init__(self, fingers, dof, link_lengths, n, static=[0, 1], view=None, obj_names=[], q_grasp=None, q_used=None,
                 N_TIME=2000, dt=0.002, q0=None, GA_solve=True, ycb_poses=False, used_objs=None, real=False,
                 tracking_camera=False, version=1, prefix='', fixed_finger=1):
        self.fingers = fingers
        self.dof = dof
        self.obj_names = obj_names
        hand = crawling_hand(self.fingers, self.dof, link_lengths, objects=obj_names,
                             real=real)  # build the robot xml file
        # hand = crawling_hand(self.fingers, self.dof, link_lengths, objects=len(obj_names),
        #                      real=real)  # build the robot xml file
        self.real = real
        self.view = view
        self.version = version

        if type(ycb_poses) is bool:
            ycb_poses = []

        if len(ycb_poses):
            if self.real:
                pass
                if self.version == 1:
                    xml_data = hand.return_xml_ycb_real(q_grasp=q_grasp, q_used=q_used, ycb_poses=ycb_poses,
                                                        used_objs=used_objs)
                elif self.version == 2:
                    xml_data = hand.return_xml_ycb_real_v2(q_grasp=q_grasp, q_used=q_used, ycb_poses=ycb_poses,
                                                           used_objs=used_objs, )
                elif self.version == 3:
                    xml_data = hand.return_xml_ycb_real_v3(q_grasp=q_grasp, q_used=q_used, ycb_poses=ycb_poses,
                                                           used_objs=used_objs, )
                else:
                    raise NotImplementedError

            else:
                xml_data = hand.return_xml_ycb(q_grasp=q_grasp, q_used=q_used, ycb_poses=ycb_poses, used_objs=used_objs)
        else:
            if len(self.obj_names) >= 2:
                xml_data = hand.return_xml(q_grasp=q_grasp, q_used=q_used, obj_names=obj_names)
            else:
                xml_data = hand.return_xml_1_obj(q_grasp=q_grasp, q_used=q_used,obj_names=obj_names, fixed_finger=fixed_finger)

        # print(xml_data)
        root = ET.fromstring(xml_data)

        try:
            with open("original_hand.xml", "w") as file:
                file.write(ET.tostring(root, encoding='unicode'))
        except:
            print(root)

        # if q_grasp is not None:
        #     pass
        self.xml_data = prefix + xml_data

        self.N_TIME = N_TIME
        self.dt = dt
        self.n = n

        self.static = static
        self.q0 = q0
        self.GA_solve = GA_solve
        self.obj_names = obj_names

        self.q_grasp = q_grasp
        self.q_used = q_used
        self.r = None
        self.tracking_camera = tracking_camera
        if self.tracking_camera:
            self.filter_len = 500
            self.x_filter = np.zeros([self.filter_len, 3])
            self.x_count = 0
            self.x_all = []

    def run(self, x, init_only=False, tracking_camera=False, disable_links=False, record_q=False):
        #### for CPGs, run the simulation in MuJoCo ####

        model = mujoco.MjModel.from_xml_string(self.xml_data)
        data = mujoco.MjData(model)
        if self.view is not None:
            self.view = viewer.launch_passive(model, data)

        self.r = Robot(model, data, self.view, self.fingers, self.dof, auto_sync=True, obj_names=self.obj_names)

        if not init_only:

            if disable_links:
                colors = {}
                visible_links = ['floor', 'hand_base_1', 'hand_base_2', 'MCP_spread_motor_1', 'metacarpal_1',
                                 'MCP_motor_1', 'proximal_1', 'PIP_DIP_motor_1', 'middle_1', 'distal_1',
                                 'MCP_spread_motor_0',
                                 'metacarpal_0', 'MCP_motor_0', 'proximal_0', 'PIP_DIP_motor_0', 'middle_0', 'distal_0',
                                 'sphere_1', 'cylinder_1', 'box_1', 'box_2', 'box_3' 'base_link_0', 'base_link_1',
                                 'distal_tip_1', 'distal_tip_0']
                visible_links = ['floor', 'hand_base_1', 'hand_base_2',     'MCP_spread_motor_0',
                                 'metacarpal_0', 'MCP_motor_0', 'proximal_0', 'PIP_DIP_motor_0', 'middle_0', 'distal_0',
                                 'sphere_1', 'cylinder_1', 'box_1', 'box_2', 'box_3' 'base_link_0',
                                  'distal_tip_0']
                for i in range(self.r.m.ngeom):
                    name = self.r.m.geom(i).name
                    if name not in visible_links:
                        # print(name)
                        colors[name] = copy.deepcopy(self.r.m.geom(i).rgba)
                        self.r.m.geom(i).rgba = np.array([0, 0.5, 0, 0])  # make it transparent
                self.r.sync()
                time.sleep(1)
                for i in range(self.r.m.ngeom):
                    if self.r.m.geom(i).name not in visible_links:
                        self.r.m.geom(i).rgba = colors[self.r.m.geom(i).name]
                self.r.sync()
                time.sleep(1)
            # self.r.modify_joint(np.array([0,0,0,1,0,0,0]))

            # self.r.sync()
            if record_q:
                q_record = []
            x_record = []
            q0 = x[:self.n]
            # f = x[self.n]  # frequency of each CPG
            f = 1
            a = x[self.n: 2 * self.n]

            phi = x[2 * self.n: 3 * self.n]
            error_detect = False

            for i in range(1000):
                q_ctrl = q0 + a * np.sin( phi)
                self.r.joint_impedance_control(q_ctrl, k=1)
                self.r.step()  # let the robot drop to the ground
            # time.sleep(100)
            for t in range(self.N_TIME):
                q_ctrl = q0 + a * np.sin(2 * np.pi * f * t * self.dt + phi)

                # self.r.joint_impedance_control(q_active, k=1)
                self.r.joint_impedance_control(q_ctrl, k=1)
                if self.view is not None and tracking_camera:

                    self.x_count += 1
                    self.x_all.append(self.r.x[:3])
                    if self.x_count > self.filter_len:
                        self.x_all.pop(0)
                    x_mean = np.mean(np.vstack(self.x_all), axis=0)
                    self.r.viewer_setup(xyz=x_mean.flatten())

                # if t > 100:
                #
                # else:
                #     self.r.sync() # let the robot drop to the ground

                if not self.GA_solve and self.view is not None:
                    time.sleep(self.dt)  # sleep for sim
                    self.r.sync()

                x_record.append(copy.deepcopy(self.r.x))

                if record_q:
                    q_record.append(copy.deepcopy(self.r.q))

                # quat_dis = rot.ori_dis(self.r.x[3:], np.array([1, 0, 0, 0]))
                # if (self.r.x[2] > 0.4 or quat_dis> np.pi / 2 * 1.5) and self.GA_solve:  # check if the robot is flying too high in the air
                #     # error_detect = True
                #     # print('Robot flies too high', self.r.x[2], quat_dis)
                #     break  # stop the for loop


            for i in range(100):
                # q_ctrl = q0 + a * np.sin( phi)
                self.r.joint_impedance_control(q_ctrl, k=1)
                self.r.step() # let the robot drop to the ground
                x_record.append(copy.deepcopy(self.r.x))

            x_record = np.vstack(x_record)
            # copy.deepcopy(self.r.x)

            if self.GA_solve:
                if error_detect:
                    return None
                else:
                    return x_record

            if record_q:
                return np.vstack(q_record)

    def fitness_func(self, ga_instance, solution, solution_idx):
        # solution_0 = np.copy(solution)
        # solution_0[self.r.n +1 :self.r.n +9] =0
        results = self.run(solution)  # the final position of the robot
        if self.view is not None:
            self.r.view.close()

        if sum(self.fingers) == 0:
            return -1000
        if results is None:
            fitness = -1000
        else:
            # to keep the palm always horizontal
            # quat_error = np.arccos(np.abs(results[:, 3])) / np.pi*180
            quat_error = np.arccos(np.abs(results[:, 3]))  # quat error with [1,0,0,0]
            quat_mean = np.mean(quat_error)
            height = np.mean(results[:, 2])
            # height_std = np.std(results[:, 2])
            # height_lowest = np.min(results[:, 2])
            distance = np.linalg.norm(results[-1, :2])  # crawling distance

            # fitness = distance * 10 + (height - height_std) * 200  - quat_mean * 10
            # fitness = distance * 100 + height * 200 - quat_mean * 100
            # fitness = (distance * 100 - quat_mean * 100)   # 6000
            fitness = distance    # 7000
            # print(distance)

        return fitness
