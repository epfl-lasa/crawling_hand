import time
from kinematics.grasping_synthesis import grasping, mujoco_sim
import numpy as np
import os
from os import listdir
from os.path import isfile, join

path = '../data_records/full_test/'
name = '1711537795'  # 4 fingers, nice
# name = '1711539614'  # 6
# name = '1711541490'  # 6
# name = '1711536188'  # 5 # nice

name = '1711881734'

file_names = [f[3:-4] for f in listdir(path) if isfile(join(path, f)) and len(f) ==30 and f[:3] =='GA_']

file_best = 0
solution_fitness_best = 0
for i in file_names:
    data_i = np.load(path + 'GA_' + i + '.npz')
    solution_fitness = data_i['solution_fitness']
    if solution_fitness > solution_fitness_best:
        file_best = i
        solution_fitness_best = solution_fitness

print(file_best, solution_fitness_best)
file_best = '1000_loco_sol.npy'

data_i = np.load(path + 'GA_' + file_best + '.npz')

solution = data_i['solution']
solution_fitness = data_i['solution_fitness']
fingers = data_i['fingers']
link_lengths = data_i['link_lengths']
n = data_i['n']
obj_names = data_i['obj_names']
obj_names = ['sphere_1']

q_grasp = data_i['q_grasp']
q_used = data_i['q_used']

dofs = [4] * 8
hand = mujoco_sim(fingers, dofs, link_lengths, n, view=1, GA_solve=False, N_TIME=2000 * 10, obj_names=obj_names,
                      q_grasp=q_grasp, q_used=q_used)

    # hand.r.d.qpos[2] = 0.08
    # hand.r.step()

    # hand.r.sync()
    # time.sleep(4)
hand.run(solution)
print(np.linalg.norm(hand.r.x[:2]))


