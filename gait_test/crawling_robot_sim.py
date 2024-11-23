import time

import numpy as np
import mujoco
from mujoco import viewer

import tools.rotations as rot
from hand_generator_from_genes import crawling_hand
from controller_full import Robot
import pygad

import tools.rotations as rot

# from kinematics.grasping_synthesis import grasping
from kinematics.grasping_synthesis import grasping, mujoco_sim

class crawling_robot_sim:
    def __init__(self, x, view=None, N_TIME=2000, dt=0.002, GA_solve=True, objects=False, auto_sync=True,
                 first_static=False, q1=None, grasping_link_index=None, q0=None):
        # x: [fingers, dof, lengths, q0, f, a, phi],

        self.fingers = np.concatenate([np.array([1]), np.int8(x[:7])])  # [1, 1,1,1,1,1,1,0,1,1,1,1]

        self.dof = np.int8(np.ones(8)*4)  # [2,3,4,2,3,4,2,3,4,2,3,3]

        link_lengths = x[7:15]  # [0.05]*12
        if first_static:
            self.fingers[0] = 1
            self.dof[0] = 4
            link_lengths[0] = 0.1
        hand = crawling_hand(self.fingers, self.dof, link_lengths, objects=objects)  # build the robot xml file
        xml_data = hand.return_xml()
        model = mujoco.MjModel.from_xml_string(xml_data)
        data = mujoco.MjData(model)
        if view is not None:
            self.view = viewer.launch_passive(model, data)
        else:
            self.view = None
        if objects:
            obj_names = ['sphere_1', ]
        else:
            obj_names = []
        self.r = Robot(model, data, self.view, self.fingers, self.dof, auto_sync=auto_sync, obj_names=obj_names)
        self.N_TIME = N_TIME
        self.dt = dt
        self.GA_solve = GA_solve

        self.locomotion_parameters = None
        self.first_static = first_static
        self.grasping_link_index = grasping_link_index
        self.q0 = q0


        # if self.first_static:
        #     x_box = np.array([0, 0, 0.03, 1, 0, 0, 0])
        #     self.r.modify_obj_pose('sphere_1', x_box)
        #     q0 = np.copy(self.r.q)
        #     q0[:4] = q1
        #     self.q1 = q0
        #     self.r.reset(x=None, q=q0)

    def run(self, x):
        #### for CPGs, run the simulation in MuJoCo ####
        x_record = np.zeros([self.N_TIME, 7])
        q0 = x[:self.r.n]
        f = x[self.r.n]  # frequency of each CPG
        f = 1
        a = x[self.r.n + 1: 2 * self.r.n + 1]
        a[:8] = 0
        if self.q0 is not None: # the first and second fingers
            a[:self.grasping_link_index[0]] = 0
            q0[:self.grasping_link_index[0]] = self.q0[:self.grasping_link_index[0]]
            # if self.fingers[1]:
            a[4:4+self.grasping_link_index[1]] = 0
            q0[4:4+self.grasping_link_index[1]] = self.q0[4:4+self.grasping_link_index[1]]
                   #
        # if self.first_static:  # the first finger is used for grasping
        #     a[:4] = 0
        #     q0[:4] = self.q1[:4]
        #     assert self.dof[0] == 4
        #     assert self.fingers[0] == 1
        phi = x[2 * self.r.n + 1: 3 * self.r.n + 1]
        error_detect = False
        for t in range(self.N_TIME):
            q_ctrl = q0 + a * np.sin(2 * np.pi * f * t * self.dt + phi)
            # q_active = []
            # for i in range(12):
            #     if self.fingers[i]:
            #         dof_i = self.dof[i]
            #         q_active.append(q_ctrl[i * 4: i * 4 + dof_i])
            # if len(q_active) > 1:
            #     q_active = np.hstack(q_active)
            # else:
            #     q_active = q_active[0]
            # assert len(q_active) == len(self.r.q)

            # self.r.joint_impedance_control(q_active, k=1)
            self.r.joint_impedance_control(q_ctrl, k=1)
            if self.first_static:
                x_box = np.copy(self.r.x)
                x_box[:3] += rot.quat2mat(self.r.x[3:]) @ np.array([0, 0, 0.02])
                self.r.modify_obj_pose('box_1', x_box)
                self.r.modify_first_finger(self.q1[:4])

                self.r.sync()
            x_record[t, :] = self.r.x
            if not self.GA_solve and self.view is not None:
                time.sleep(self.dt)  # sleep for sim
            if (self.r.x[2] > 0.4 or rot.ori_dis(self.r.x[3:], np.array(
                    [1, 0, 0, 0])) > np.pi / 2) and self.GA_solve:  # check if the robot is flying too high in the air
                error_detect = True
                break  # stop the for loop
        if self.GA_solve:
            if error_detect:
                return None
            else:
                return x_record

    def fitness_func(self, ga_instance, solution, solution_idx):
        # solution_0 = np.copy(solution)
        # solution_0[self.r.n +1 :self.r.n +9] =0
        results = self.run(solution)  # the final position of the robot

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
            fitness = (distance * 100 - quat_mean * 200)
            # fitness = distance * 10

        return fitness


def GA_locomotion(x, print_result=False, para=None, first_static=False, objects=False,q1=None, obj_nums=3, pinch_one_only=False):
    """
    GA for locomotion for the specific robot
    x : the structure of the hand
    :return:
    """
    # print('test GA ')
    fingers= x[:8]   # if there is a finger or not
    fingers[0] = 1  # always a finger at the x-axis
    # fingers[1] = 1
    x[0] = 1        # always a finger at the x-axis
    # fingers = np.concatenate([np.array([1]), np.int8(x[:7])])   # [1, 1,1,1,1,1,1,0,1,1,1,1]
    dof = [4] * 8  # each finger has 4 DoF
    if fingers[1] == 0 and obj_nums >= 2:
        fingers[2] = 1   # the 3rd place has to be a finger
        x[2] = 1
        p0 = np.pi/2
        m_ = np.array([0.05, x[10],x[10],x[10],x[10]]) # length of 3rd place
    else:
        p0 = np.pi/4
        m_ = np.array([0.05, x[9], x[9], x[9], x[9]])

    l_ = np.array([0.05, x[8],x[8],x[8],x[8]])
    r_ = [0.01, 0.02,0.02, 0.02]
    q0 = 0
    g1 = grasping(path_prefix='../kinematics/')

    tmp = [(2, 2), (3, 3), (4, 4), (3, 4)]   # possible way to grasp one object by two fingers
    tmp += [(2, 4)]   # grasp one object by one finger
    grasping_link_index = tmp[np.random.choice([0, 1, 2, 3, 4])]
    # generate grasping configuration
    # res = g1.generate_grasp(l_,m_,r_,q0,p0, grasping_link_index=grasping_link_index)


    feasible = True

    q_used = [1 for i in range(8)]  # todo, it needs to adjust based on grasping_link_index

    fixed_finger = 2

    if obj_nums == 1:
        obj_names = ['sphere_1']

        if pinch_one_only:  # pinch grasp only one object by two fingers
            pass
            res = g1.generate_grasp(l_, m_, r_, q0, p0, grasping_link_index=(4, 4), single=False)

            n = int(sum(np.array(fingers[:]) * np.array(dof[:]))) - 8  # DoFs
            q_grasp = np.copy(res.x[:11])  # 8+3, joints + the position of one sphere in the hand base frame
            feasible = res.success and n > 0
            fixed_finger = 2
            print(res.x, res.fun)

        else:
            res2 = g1.generate_single_grasp(l_, r_, 0)  # 1st finger, grasp it by wrapping
            q_grasp = np.copy(res2.x[:7])
            q0 = np.copy(res2.x[:7])
            n = int(sum(np.array(fingers[:]) * np.array(dof[:]))) - 4
            feasible = res2.success and n > 0 and res2.fun < 1e-3
            fixed_finger = 1
    else:
        res1 = g1.generate_single_grasp(m_, r_, p0)  # 2nd finger, wrapping
        res2 = g1.generate_single_grasp(l_, r_, 0)  # 1st finger, wrapping
        q_double = np.concatenate([res2.x[:4], res1.x[:4]])
        x_init = np.concatenate([res2.x[4:7], res1.x[4:7]])

        n = int(sum(np.array(fingers[:]) * np.array(dof[:]))) - 8

        if obj_nums == 3:
            res3 = g1.generate_triple_grasp(l_, m_, r_, q0, p0, q_init=q_double, x_init=x_init)  # pinch grasp
            q0 = np.copy(res3.x[:8])
            q_grasp = np.copy(res3.x[:17])  #  8+3+3+3, joints + the positions of 3 spheres in the hand base frame
            feasible = res3.success and n > 0
            obj_names = ['sphere_1', 'sphere_2', 'sphere_3']
        elif obj_nums == 2:
            q0 = np.copy(q_double)
            q_grasp = np.concatenate([q_double, x_init])  # 8+3+3, joints + the positions of 2 spheres in the hand base frame
            feasible = res1.success and  res2.success and n > 0
            obj_names = ['sphere_1', 'sphere_2']
        else:
            raise NotImplementedError



    link_lengths = x[8:]

    if feasible:
        # n = int(sum(np.array(fingers) * np.array(dof)))
        print('find feasible grasp, n=', n, fingers)

        # num_genes = 1 + n * 3
        num_genes =  n * 3
        mutation_type = "random"
        mutation_percent_genes = 10
        gene_space = [{'low': -1, 'high': 1}] * n  # q0
        # gene_space += [{'low': 0.1, 'high': 10}]  # frequency
        gene_space += [{'low': -np.pi / 2, 'high': np.pi / 2}] * n  # a
        gene_space += [{'low': -np.pi / 2, 'high': np.pi / 2}] * n  # alpha

        if para is None:  # for structure GA training
            num_generations = 10
            num_parents_mating = 8
            sol_per_pop = 4 * 4
            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   # fitness_func=crawling_robot_sim(x).fitness_func,
                                   fitness_func=mujoco_sim(fingers, dof, link_lengths, n, q_grasp=q_grasp,
                                                           q_used=q_used,obj_names=obj_names, fixed_finger=fixed_finger).fitness_func,
                                   # fitness_func=crawling_robot_sim(x, first_static=first_static, objects=True,
                                   #                                 q1=q1,N_TIME=4000, grasping_link_index=grasping_link_index,
                                   #                                 q0=res.x).fitness_func,
                                   # on_generation=on_generation,
                                   mutation_type=mutation_type,
                                   mutation_percent_genes=mutation_percent_genes,
                                   parallel_processing=('process', 30),
                                   gene_space=gene_space
                                   )
        else:  # for replaying the result
            num_generations = para[0]
            num_parents_mating = para[1]
            sol_per_pop = para[2]
            ga_instance = pygad.GA(num_generations=num_generations,
                                   num_parents_mating=num_parents_mating,
                                   sol_per_pop=sol_per_pop,
                                   num_genes=num_genes,
                                   fitness_func=crawling_robot_sim(x,grasping_link_index=grasping_link_index,q0=res.x).fitness_func,
                                   # on_generation=on_generation,
                                   mutation_type=mutation_type,
                                   mutation_percent_genes=mutation_percent_genes,
                                   parallel_processing=('process', 32),
                                   gene_space=gene_space
                                   )

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        return solution_fitness, solution, q_grasp
    else:
        print('no feasible grasp, n=', n, fingers)
        return 0, 0, 0


def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
