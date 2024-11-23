# requires installation of pytorch_kinematics: https://github.com/UM-ARM-Lab/pytorch_kinematics
import math
import pytorch_kinematics as pk
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os


# visualize reachable space of entire hand
def plot_finger_reachable_map(link_1, link_2, tip, ax=None):
    if ax is None:
        fig = plt.figure()
        if_single_finger = True  # if plot single finger
        ax = fig.add_subplot(111, projection='3d')
    else:
        if_single_finger = False

    # if link_1 is torch.Tensor (homogeneous coordinates or not)
    if isinstance(link_1, torch.Tensor):
        link_1 = link_1.detach().numpy()  # shape (N,4)
        link_2 = link_2.detach().numpy()
        tip = tip.detach().numpy()

    if link_1.ndim == 2:
        link_1 = np.expand_dims(link_1, axis=0)
        link_2 = np.expand_dims(link_2, axis=0)
        tip = np.expand_dims(tip, axis=0)

    ax.scatter(link_1[:, :, 0], link_1[:, :, 1], link_1[:, :, 2], c='r', s=1, marker='o')
    ax.scatter(link_2[:, :, 0], link_2[:, :, 1], link_2[:, :, 2], c='g', marker='o')
    ax.scatter(tip[:, :, 0], tip[:, :, 1], tip[:, :, 2], c='b', marker='o')

    if if_single_finger:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.axis('equal')
        plt.show()

    return ax


# joint angle coupling
if_coupled = True  # if coupled, joint angles: DIP = PIP/3
if_symmetry = False  # exploit the symmetry property of the finger reachable space to reduce sampling

# load model
# chain = pk.build_serial_chain_from_urdf(open("URDF_finger_ssss.urdf").read(), "distal_1") # specify end effector link
# print(chain.get_joint_parameter_names()) # list of joint names: 'MCP_spread', 'MCP', 'PIP', 'DIP'

# xml_path = '/home/kunpeng/Workspace/crawling_hand/descriptions/five_finger_hand_ssss.xml'
xml_path = os.getcwd() + '/../../five_finger_hand_ssss.xml'
chain = pk.build_chain_from_mjcf(xml_path, 'hand')
# Number of dof: 20. only sample one single finger instead of the whole hand

# joints of each finger: 'MCP_spread_i', 'MCP_i', 'PIP_i', 'DIP_i', where i = 1,2,3,4,5
if if_symmetry:
    finger_joint_limit = np.array([[0, 0, 0, 0],
                                   [math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2]])
else:
    finger_joint_limit = np.array([[-math.pi / 2, -math.pi / 2, -math.pi / 2, -math.pi / 2],
                                   [math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2]])

## Sample reachability of joints
# exploit symmetry of finger to reduce sampling, only sample 
N = 4000
nF = 5  # number of fingers
iF = 1  # index of finger to sample

# link_names = ['base_link', 'MCP_spread_motor_1', 'metacarpal_1', 'MCP_motor_1', 'proximal_1', 'PIP_DIP_motor_1', 'middle_1', 'distal_1']
# joint_names = ['MCP_spread_'+str(iF), 'MCP_'+str(iF), 'PIP_'+str(iF), 'DIP_'+str(iF)]
joint_names = ['MCP_spread_' + str(iF), 'MCP_' + str(iF), 'PIP_' + str(iF), 'DIP_' + str(iF)]
link_names = ['finger_' + str(iF), 'MCP_spread_motor_' + str(iF), 'MCP_motor_' + str(iF), 'PIP_DIP_motor_' + str(iF),
              'distal_' + str(iF),
              'finger_' + str(iF) + '_tip', 'finger_' + str(iF) + '_link_2',
              'finger_' + str(iF) + '_link_1']  # these link names are obtained from ret.keys() below

# useful links (last three, used as the reachable maps): 'finger_'+str(iF)+'_link_1', 'finger_'+str(iF)+'_link_2', 'finger_'+str(iF)+'_tip'

nLinks = len(link_names)

link_positions = np.zeros(
    (nLinks, N, 3))  # 8 joints, N samples, 3 dimensions # number of joints is obtaineded from the len(ret) below

## Sample reachability of a finger
if os.path.exists('link_positions.npy'):
    link_positions = np.load('link_positions.npy')  # numpy array, (8,4000,3)
    print("File loaded successfully.")
else:
    print("File does not exist. Sampling.")
    for i in range(N):
        # tic = time.perf_counter()

        random.seed(time.time())
        p = [random.uniform(low, high) for low, high in zip(finger_joint_limit[0, :], finger_joint_limit[1, :])]
        if if_coupled:
            p[-1] = p[-2]

        p_hand = np.tile(p, 5)  # type: numpy.ndarray, size: (20,)
        ret = chain.forward_kinematics(p_hand,
                                       end_only=False)  # do forward kinematics and get transform objects; ret is a dictionary, use ret.keys()
        # dict_keys(['hand',...
        # 'finger_1', 'MCP_spread_motor_1', 'MCP_motor_1', 'PIP_DIP_motor_1', 'distal_1', 'finger_1_tip', 'finger_1_link_2', 'finger_1_link_1',...
        # 'finger_2', 'MCP_spread_motor_2', 'MCP_motor_2', 'PIP_DIP_motor_2', 'distal_2', 'finger_2_tip', 'finger_2_link_2', 'finger_2_link_1',...
        # 'finger_3', 'MCP_spread_motor_3', 'MCP_motor_3', 'PIP_DIP_motor_3', 'distal_3', 'finger_3_tip', 'finger_3_link_2', 'finger_3_link_1',...
        # 'finger_4', 'MCP_spread_motor_4', 'MCP_motor_4', 'PIP_DIP_motor_4', 'distal_4', 'finger_4_tip', 'finger_4_link_2', 'finger_4_link_1',...
        # 'finger_5', 'MCP_spread_motor_5', 'MCP_motor_5', 'PIP_DIP_motor_5', 'distal_5', 'finger_5_tip', 'finger_5_link_2', 'finger_5_link_1'])

        for j in range(nLinks):  # itrate over links
            # print(link_names[j])
            tg = ret[link_names[j]]
            m = tg.get_matrix()  # tensor, size [1,4,4]
            pos = m[:, :3, 3]
            link_positions[j, i, :] = pos

        # toc = time.perf_counter()
        # elapsed = toc - tic
        # print(f"Iteration {i}: {elapsed:0.4f} seconds")
    print('finished sampling.')

    # visualize
    '''
    for j in range(len(link_names)):
        data = link_positions[j,:,:]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:,0], data[:,1], data[:,2]) 
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(link_names[j])
        plt.show()
        # save figure
        fig.savefig(link_names[j] + '.png', dpi=300) # save as png file
        # close figure
        plt.close()
    '''
    # np.save('link_positions.npy', link_positions) # save as numpy array
    # save as .mat file
    # import scipy.io
    # scipy.io.savemat('link_positions.mat', mdict={'link_positions': link_positions}) # save as .mat file

## Construct the reachable space of the full hand
# home configuration
ret = chain.forward_kinematics(np.zeros((1, 20)), end_only=False)
hand_base = ret['hand'].get_matrix()
finger_1_base = ret['finger_1'].get_matrix()  # tensor, size [1,4,4]
finger_2_base = ret['finger_2'].get_matrix()
finger_3_base = ret['finger_3'].get_matrix()
finger_4_base = ret['finger_4'].get_matrix()
finger_5_base = ret['finger_5'].get_matrix()

H1_inv = torch.inverse(finger_1_base)
# H2_inv = torch.inverse(finger_2_base)
# H3_inv = torch.inverse(finger_3_base)
# H4_inv = torch.inverse(finger_4_base)
# H5_inv = torch.inverse(finger_5_base)

H12 = torch.matmul(finger_2_base, H1_inv).squeeze(
    0)  # transformation matrix from finger 1 to finger 2, tensor, size [1,4,4]
H13 = torch.matmul(finger_3_base, H1_inv).squeeze(0)
H14 = torch.matmul(finger_4_base, H1_inv).squeeze(0)
H15 = torch.matmul(finger_5_base, H1_inv).squeeze(0)

# construct reachable space of finger i by transforming the reachable space of finger 1 by H1i (i = 2,3,4,5)
# Finger 1: transform sampled data to homogeneous coordinates first

finger_1_link_1 = torch.tensor(np.hstack((link_positions[-3, :, :], np.ones((N, 1)))),
                               dtype=torch.float)  # torch.Size([4000, 4])
finger_1_link_2 = torch.tensor(np.hstack((link_positions[-2, :, :], np.ones((N, 1)))),
                               dtype=torch.float)  # torch.Size([4000, 4])
finger_1_tip = torch.tensor(np.hstack((link_positions[-1, :, :], np.ones((N, 1)))),
                            dtype=torch.float)  # torch.Size([4000, 4])
# plot_finger_reachable_map(finger_1_link_1, finger_1_link_2, finger_1_tip)
# breakpoint()

# Finger 2
finger_2_link_1 = torch.matmul(finger_1_link_1, H12)
finger_2_link_2 = torch.matmul(finger_1_link_2, H12)
finger_2_tip = torch.matmul(finger_1_tip, H12)
# plot_finger_reachable_map(finger_2_link_1, finger_2_link_2, finger_2_tip)
# breakpoint()

# Finger 3
finger_3_link_1 = torch.matmul(finger_1_link_1, H13)
finger_3_link_2 = torch.matmul(finger_1_link_2, H13)
finger_3_tip = torch.matmul(finger_1_tip, H13)
# plot_finger_reachable_map(finger_3_link_1, finger_3_link_2, finger_3_tip)
# breakpoint()

# Finger 4
finger_4_link_1 = torch.matmul(finger_1_link_1, H14)
finger_4_link_2 = torch.matmul(finger_1_link_2, H14)
finger_4_tip = torch.matmul(finger_1_tip, H14)
# plot_finger_reachable_map(finger_4_link_1, finger_4_link_2, finger_4_tip)
# breakpoint()

# Finger 5
finger_5_link_1 = torch.matmul(finger_1_link_1, H15)
finger_5_link_2 = torch.matmul(finger_1_link_2, H15)
finger_5_tip = torch.matmul(finger_1_tip, H15)
# plot_finger_reachable_map(finger_5_link_1, finger_5_link_2, finger_5_tip)
# breakpoint()

torch.save({
    'finger_1_link_1': finger_1_link_1,
    'finger_1_link_2': finger_1_link_2,
    'finger_1_tip': finger_1_tip,
    'finger_2_link_1': finger_2_link_1,
    'finger_2_link_2': finger_2_link_2,
    'finger_2_tip': finger_2_tip,
    'finger_3_link_1': finger_3_link_1,
    'finger_3_link_2': finger_3_link_2,
    'finger_3_tip': finger_3_tip,
    'finger_4_link_1': finger_4_link_1,
    'finger_4_link_2': finger_4_link_2,
    'finger_4_tip': finger_4_tip,
    'finger_5_link_1': finger_5_link_1,
    'finger_5_link_2': finger_5_link_2,
    'finger_5_tip': finger_5_tip,
}, 'hand_reachability_map.pth')

# breakpoint()

# plot the reachable space of the full hand
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j in range(nF):
    # breakpoint()
    ax = plot_finger_reachable_map(eval('finger_' + str(j + 1) + '_link_1'), eval('finger_' + str(j + 1) + '_link_2'),
                                   eval('finger_' + str(j + 1) + '_tip'), ax)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Hand reachable space')
plt.axis('equal')
plt.show()
