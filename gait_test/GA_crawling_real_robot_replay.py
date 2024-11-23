import mujoco
from mujoco import viewer
import sys
######################
# replay the CPGs with the real robot hand
# v2 is the version 2

from controller_full import Robot
import numpy as np
# import tqdm
import time
from os import listdir
from os.path import isfile, join
import os.path
# import pygad
from hand_generator_from_genes import crawling_hand
from crawling_robot_sim import crawling_robot_sim, GA_locomotion
from kinematics.grasping_synthesis import grasping, mujoco_sim
from itertools import combinations

N_TIME = 20000
dt = 0.002
view = None
path = '../data_records/grasp_synthesis/real_hand_v2/locomotion_data/'

type = 1

file_names = [f for f in listdir(path) if isfile(join(path, f)) and len(f)==9 and f[0]==str(type)]


if len(file_names):
    print('File numbers', len(file_names))
else:
    print( 'No file')


nums = 8
fixed = [0, 1]  # the place of 0 has a fixed finger for grasping
if type == 1:
    fixed = fixed[:1]
# file_names = [str(int(10000 * type) + i) + '.npz' for i in range(int(len(file_names)/2), len(file_names))]
file_names = [str(int(10000 * type) + i) + '.npz' for i in range(int(len(file_names)/2))]

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

    # print(i, len(config_i))
    configs[str(i)] = config_i


# 2 finger grasp ###########
fingers_list = []
fingers_2_record = {}
for i in configs:
    for j in configs[i]:
        fingers_list.append(j)
        fingers_2_record[tuple(j)] = []
#
dofs = [4] * 8
#
# # sort data into above dict
#
# real_robot_loco = '1.npz'
#
for f in file_names:
    d = np.load(
        path + f)  # fingers=fingers, d_list=d_list[q_choice], solution=solution, solution_fitness=solution_fitness
    fingers = d['fingers']

    # print(fingers)
    # if list(fingers) != [1, 1, 1, 1, 1, 1, 1, 1]:
    #     continue
    solution = d['solution']
    solution_fitness = d['solution_fitness']

    d_list = str(d['d_list'])
    fingers_2_record[tuple(fingers)].append([d_list, solution, solution_fitness, f])

best_solution = [0, 0]
dis = 0
for i in fingers_2_record:
    tmp = [j[2] for j in fingers_2_record[i]]
    print(i, np.argmax(tmp), fingers_2_record[i][np.argmax(tmp)][0])
    fitness = fingers_2_record[i][np.argmax(tmp)][2]
    if fitness > dis:
        best_solution[0] = i
        best_solution[1] = fingers_2_record[i][np.argmax(tmp)][3]
        dis = fitness

print(best_solution)

# for f in file_names:

for i in [best_solution]:  # only sim the best
    f = best_solution[1]


# for f in file_names:
    d = np.load(path + f)  # fingers=fingers, d_list=d_list[q_choice], solution=solution, solution_fitness=solution_fitness
    fingers = d['fingers']
    fingers_str = [str(i) for i in fingers]
    # if sum(fingers) !=3:
    #     continue

    # if sum(fingers) !=6 or f[:4] in ['4134', '4140', '4145', '4134', '4131', '4124']:
    #     continue
    # if list(fingers) != [1,1,1, 0, 1, 0, 0, 1]:  # the real hand
    #     continue
    # else:
    #     print(fingers)
    solution = d['solution']
    solution_fitness = d['solution_fitness']

    d_list = str(d['d_list'])

    b=''
    for j in fingers:
        b+=str(j)
    # if d_list[5] not in ['2', '3'] or '00' not in b:
    #     continue

    grasp_syn = np.load('../data_records/grasp_synthesis/real_hand_v2/' + d_list + '.npz') # q_grasp=q_grasp, objs=[1, 0], obj_poses=np.array(obj_pos_base)

    q_grasp = grasp_syn['q_grasp'][:len(fixed)*4]

    print(d_list, q_grasp)
    used_obj = grasp_syn['objs']
    ycb_poses = grasp_syn['obj_poses']
    q_used = len(q_grasp) * [1]
    n = int(sum(np.array(fingers[:]) * np.array(dofs[:]))) - len(q_grasp)


    link_length = [0.0725, 0.0235, 0.039,0.0305, 0.026]  # the length of the real hand

    obj_names = ['sphere_1', 'cylinder_1', 'box_1', 'box_2', 'box_3']
    view = 1  # disable GUI view
    tracking_camera = 0
    hand = mujoco_sim(fingers, dofs, link_length, n, view=view, GA_solve=False, N_TIME=1000*10, obj_names=obj_names,
                      q_grasp=q_grasp, q_used=q_used,ycb_poses=ycb_poses,used_objs=used_obj, real=True, tracking_camera=tracking_camera, version=4 - len(fixed))



    print(fingers, f, 'Start to run the simulation. Fitness=', solution_fitness, 'obj_list', used_obj)

    q_record = hand.run(solution, tracking_camera=tracking_camera, disable_links=True, record_q=True)
    np.save(path + f[:4] +'_q', q_record)
    print('Crawling distance in simulation:', np.linalg.norm(hand.r.x[:2]), hand.r.x[2])
    if view is not None:
        hand.r.view.close()
