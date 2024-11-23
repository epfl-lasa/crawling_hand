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

# file_names = [f[3:-4] for f in listdir(path) if isfile(join(path, f)) and len(f) ==17 and f[:3] =='GA_']
# file_names = sorted(file_names)
# latest = -2
#
# num_str = str(file_names[latest])
a = 0
file_names = ['1000', '2000', '3000', '4000']  # load the different grasp, check the best structure for crawling.
for name in file_names:
    # name = '1711881734'
    print(name, a, len(file_names))
    a += 1
    data = np.load(path + 'GA_' + name + '.npz')

    saved_fitness = data['saved_fitness']
    solution = data['solution']
    locomotion_para = data['locomotion_para']
    q_grasp = data['q_grasp']  # 8+3+3+3
    obj_nums = data['obj_nums']
    if obj_nums == 4:  # pinch grasp one obj by two finers
        obj_nums = 1
        finger_fixed = 2
    else:
        finger_fixed = int((q_grasp.shape[0] - obj_nums * 3) / 4)

    fingers = solution[:8]
    fingers[0] = 1
    # fingers[1] = 1
    # if fingers[1] == 0:
    #     fingers[2] = 1  # the 3rd place has to be a finger

    dofs = [4] * 8
    link_lengths = solution[8:]
    n = int(sum(np.array(fingers[:]) * np.array(dofs[:]))) - finger_fixed * 4
    obj_names = ['sphere_' + str(i) for i in range(obj_nums)]

    q_used = [1 for i in range(8)]

    hand = mujoco_sim(fingers, dofs, link_lengths, n, view=1, GA_solve=False, N_TIME=2000 * 2, obj_names=obj_names,
                      q_grasp=q_grasp, q_used=q_used, fixed_finger=finger_fixed, tracking_camera=True)

    # hand.r.d.qpos[2] = 0.08
    # hand.r.step()

    # hand.r.sync()
    # time.sleep(4)
    hand.run(locomotion_para, tracking_camera=True)

    print(np.linalg.norm(hand.r.x[:2]))
    hand.r.view.close()
