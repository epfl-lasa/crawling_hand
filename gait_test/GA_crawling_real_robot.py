import time
from kinematics.grasping_synthesis import grasping, mujoco_sim
import numpy as np
import os
from os import walk
import pygad
import time

# load the static grasping results, then generate fingers on it for locomotion
# grasp only one object by one finger, iterate all the combination of fingers

from itertools import combinations


# path = '../data_records/grasp_synthesis/'
path = '../data_records/grasp_synthesis/real_hand_v2/'
data = []

for (dirpath, dirnames, filenames) in walk(path):
    data.extend(filenames)
    break

nums = 8
for type in [1]:
    prefix = int(type * 100000)
    if type == 1:
        # j = 100000  # only consider crawling distance, for 1 finger press grasp 1 obj
        grasping = ['data_1_2_real_0_3', 'data_1_2_real_0_2']
        #        with the palm,             wrapping

    elif type == 2:
        grasping = ['data_2_2_real_2_2', 'data_2_2_real_3_2']
        j = 200000  # only consider crawling distance, for 2 finger press grasp 2 obj

        # palm and a wrapping,   both palm
    else:
        j = 300000  # 3 objects grasping
        grasping = ['data_3_2_real_2_3']

    fixed = [0, 1]  # the place of 0 has a fixed finger for grasping
    if type == 1:
        fixed = fixed[:1]

    fingers_fixed = [1 for i in range(len(fixed))]
    symmetry_check = [[0, 7, 6, 5, 4, 3, 2, 1], [1, 0, 7, 6, 5, 4, 3, 2]]

    adding_fingers_num = list(range(1, nums - len(fixed) + 1))

    configs = {}

    for i in adding_fingers_num:
        config_i = []
        for p in combinations(range(len(fixed), 8), i):
            config = [0 for _ in range(8)]
            for f in fixed:
                config[f] = 1
            for j in p:
                config[j] = 1
                # symmetry check
            config_sym = [config[s] for s in symmetry_check[len(fixed) - 1]]
            if config_sym in config_i:
                pass
            else:
                config_i.append(config)

        print(i, len(config_i))
        configs[str(i)] = config_i


    trials_num = 10
    obj_names = ['sphere_1', 'cylinder_1', 'box_1', 'box_2', 'box_3']
    link_length = [0.0725, 0.0235, 0.039, 0.0305, 0.026]  # the length of the real hand
    dofs = [4] * 8

    fingers_list = []
    t_strs = []
    data_rands = []
    d_list_cs = []

    real_place = [[[0, 1, 1, 0, 0, 1]]]


    j=0
    for d_list_c in grasping:
        for i in adding_fingers_num:
            for f_i in configs[str(i)]:
                for trial in range(trials_num):
                    # d_list_c = grasping
                    fingers_list.append(f_i)
                    t_str = str(prefix + j)
                    print(t_str)
                    t_strs.append(t_str)

                    d_list_cs.append(d_list_c)
                    data_rand = np.load(path + d_list_c + '.npz')
                    data_rands.append(data_rand)
                    j += 1

    print('finger_list', len(fingers_list))
    def on_generation(ga_instance: pygad.GA):
        print(f"Generation = {ga_instance.generations_completed}")
        print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")


    def GA_loco(i):
        fingers = fingers_list[i]
        # print(fingers)
        t_str = t_strs[i]
        data_rand = data_rands[i]
        q_grasp = data_rand['q_grasp'][:len(fixed)*4]
        q_used = len(q_grasp) * [1]

        n = int(sum(np.array(fingers[:]) * np.array(dofs[:]))) - len(q_grasp)
        ycb_poses = data_rand['obj_poses']
        used_obj = data_rand['objs']
        # print('\n#########################################################')
        # print(fingers, d_list[q_choice], used_obj, '           trial', trial)

        num_genes = n * 3
        mutation_type = "random"
        mutation_percent_genes = 10
        gene_space = [{'low': -1, 'high': 1}] * n  # q0
        # gene_space += [{'low': 0.1, 'high': 10}]  # frequency
        gene_space += [{'low': -np.pi / 2, 'high': np.pi / 2}] * n  # a
        gene_space += [{'low': -np.pi / 2, 'high': np.pi / 2}] * n  # alpha

        num_generations = 25
        num_parents_mating = 8 * 4
        sol_per_pop = 4 * 4 * 4

        # num_generations = 3
        # num_parents_mating = 8
        # sol_per_pop = 4 * 4

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               # fitness_func=crawling_robot_sim(x).fitness_func,
                               fitness_func=mujoco_sim(fingers, dofs, link_length, n, obj_names=obj_names,
                                                       q_grasp=q_grasp, q_used=q_used, ycb_poses=ycb_poses,
                                                       used_objs=used_obj, N_TIME=2500, real=True, view=None,
                                                       version=4 - len(fixed)).fitness_func,
                               on_generation=on_generation,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               parallel_processing=('process', 90),
                               gene_space=gene_space
                               )

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        # time.sleep(2)

        file_save_name = path + 'locomotion_data/' + t_str
        np.savez(file_save_name, fingers=fingers, d_list=d_list_cs[i], solution=solution,
                 solution_fitness=solution_fitness)
        print('Save loco para to ', file_save_name)
        return 1


    from multiprocessing import Pool
    from contextlib import closing
    import tqdm

    # Memory usage keep growing with Python's multiprocessing.pool
    # use this to close them
    num = len(fingers_list)
    for i in range(num):
        GA_loco(i)

# with closing(Pool(30)) as a_pool:
#     result = list(tqdm.tqdm(a_pool.imap(GA_loco, range(num)), total=num))
