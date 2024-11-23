import math
import pytorch_kinematics as pk
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
from scipy.optimize import minimize
from scipy.optimize import Bounds # for defining the bounds constraints
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint

global chain # robot model
global iF # finger index, iF = 1,2,3,4,5
global radius # radius of the object
global joint_names

iF = 1 # finger index, iF = 1,2,3,4,5
joint_names = ['MCP_spread_'+str(iF), 'MCP_'+str(iF), 'PIP_'+str(iF), 'DIP_'+str(iF)]

# load robot model
xml_path = os.getcwd()+'/../../five_finger_hand_ssss.xml'
chain = pk.build_chain_from_mjcf(xml_path, 'hand')

# Single object grasping
radius = 10/2
# radius = 20/2 # object 2
# radius = 30/2 # object 3

# obtain contact position of the finger
def obtain_contact_position(x):
	# print(x)
	"""obtain contact position of the finger"""
	p_finger = [x[3], x[4], x[5], x[5]] # joint angles, notice that the last two joints are coupled
	p_hand = np.zeros(20) # joint angles of the entire hand # numpy.ndarray, size: (20,)
	p_hand[3*(iF-1):3*(iF-1)+4] = p_finger # fill in the finger joint angles
	ret = chain.forward_kinematics(p_hand, end_only=False) # forward kinematics
	# obtain contact position
	p_tip = ret['finger_'+str(iF)+'_tip'].get_matrix()[:, :3, 3] # tensor, size [1,4,4]
	p_link_1 = ret['finger_'+str(iF)+'_link_1'].get_matrix()[:, :3, 3]
	p_link_2 = ret['finger_'+str(iF)+'_link_2'].get_matrix()[:, :3, 3]

	# print(p_tip, p_link_1, p_link_2)
	return p_tip, p_link_1, p_link_2

'''
def objfun(x):
	"""the objective function"""
	# return np.linalg.norm(x[-3:])
	return 1
'''

### define nonlinear constraint
# def cons_f(x):
def objfun(x):
	'''
	url: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.NonlinearConstraint.html#scipy.optimize.NonlinearConstraint
	'''
	# Nonlinear equality constraint: in contact with the object.
	# use same upper and lower bounds to define equality constraint
	ox = x[0] # x coordinate of the object center
	oy = x[1]
	oz = x[2]
	p_tip, p_link_1, p_link_2 = obtain_contact_position(x)
	d1 = np.linalg.norm(p_tip - np.array([ox, oy, oz])) - radius
	d2 = np.linalg.norm(p_link_1 - np.array([ox, oy, oz])) - radius
	d3 = np.linalg.norm(p_link_2 - np.array([ox, oy, oz])) - radius
	# breakpoint()
	# print(d1,d2,d3)
	return d1**2+d2**2+d3**2

nonlinear_constraint = NonlinearConstraint(cons_f, [0,0,0], [0,0,0], jac='2-point')

# x0 = [0,0,0,0,0,0] # joint angles (only the first 3 joints)

bound_low = [-0.2, -0.2, -0.2, 0, 0, 0] # lower bound of the optimization variables
bound_high = [0.2, 0.2, 0.2, math.pi/2, math.pi/2, math.pi/2] # upper bound of the optimization variables
bounds = Bounds(bound_low, bound_high) # bounds for joint angles

x0 = [random.uniform(low,high) for low,high in zip(bound_low,bound_high)]

'''
ineq_cons = {'type': 'ineq',
			 'fun' : lambda x: np.array([x[0] - 2*x[1] + 2,
										 -x[0] - 2*x[1] + 6,
										 -x[0] + 2*x[1] + 2])}
eq_cons = {'type': 'eq',
		   'fun' : lambda x: np.array([, , , ])}
'''

res = minimize(objfun, x0, method='Nelder-Mead',
			   constraints=nonlinear_constraint,
			#    options={'ftol': 1e-9, 'disp': True},
			   bounds=bounds)

print(res.x)
print(res.success)
print(res.message) # Positive directional derivative for linesearch
print(res.fun)
# breakpoint()
# check optimization solver and formulation: why linesearch is not working?