# this script is used to analyze the opposition space of the hand
import math
import pytorch_kinematics as pk
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

loaded_data = torch.load('hand_reachability_map.pth')

handlinks = ['finger_1_link_1', 'finger_1_link_2', 'finger_1_tip',
                 'finger_2_link_1', 'finger_2_link_2', 'finger_2_tip',
                 'finger_3_link_1', 'finger_3_link_2', 'finger_3_tip',
                 'finger_4_link_1', 'finger_4_link_2', 'finger_4_tip',
                 'finger_5_link_1', 'finger_5_link_2', 'finger_5_tip']

tick_labels = ['F1L1', 'F1L2', 'F1T', 'F2L1', 'F2L2', 'F2T', 'F3L1', 'F3L2', 'F3T', 
               'F4L1', 'F4L2', 'F4T', 'F5L1', 'F5L2', 'F5T'] # 15 links

# Create tensor_names using dictionary comprehension
tensor_names = {i: name for i, name in enumerate(tick_labels)}

def calculate_distances(tensor_a, tensor_b):
    # Convert from homogeneous coordinates ([N, 4]) to 3D coordinates ([N, 3])
    tensor_a_3d = tensor_a[:, :3]
    tensor_b_3d = tensor_b[:, :3]

    # Calculate all pairwise distances
    distances = torch.cdist(tensor_a_3d.unsqueeze(0), tensor_b_3d.unsqueeze(0)).squeeze(0)        

    # Find the minimum and maximum distances
    min_distance = torch.min(distances)
    max_distance = torch.max(distances)

    return min_distance.item(), max_distance.item()

num_tensors = len(handlinks)
distance_matrix = torch.zeros((num_tensors, num_tensors, 2))

for i in range(num_tensors):
    for j in range(num_tensors):
        if i != j:
            tensor_i = loaded_data[handlinks[i]] # 'torch.FloatTensor', torch.Size([N, 4])
            tensor_j = loaded_data[handlinks[j]]

            if tensor_i.shape[1] != tensor_j.shape[1]:
                print(handlinks[i], tensor_i.shape, handlinks[j], tensor_j.shape)
                breakpoint()
                raise ValueError("The input tensors must have the same number of columns.")

            min_dist, max_dist = calculate_distances(tensor_i, tensor_j)
            distance_matrix[i, j, 0] = min_dist
            distance_matrix[i, j, 1] = max_dist

# Assuming distance_matrix is a [15, 15, 2] tensor or NumPy array
# Convert to NumPy array if it's a tensor
if isinstance(distance_matrix, torch.Tensor):
    distance_matrix = distance_matrix.numpy()

min_distances = distance_matrix[:, :, 0]
max_distances = distance_matrix[:, :, 1]


# Visualize the minimum and maximum distances between all pairs of links
'''
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot heatmap for minimum distances
min_dist_plot = ax[0].imshow(min_distances, cmap='viridis', aspect='auto')
ax[0].set_title('Minimum Distances')
ax[0].set_xticks(range(len(tick_labels)))
ax[0].set_yticks(range(len(tick_labels)))
ax[0].set_xticklabels(tick_labels, rotation=45, ha='right')
ax[0].set_yticklabels(tick_labels)

# Plot heatmap for maximum distances
max_dist_plot = ax[1].imshow(max_distances, cmap='viridis', aspect='auto')
ax[1].set_title('Maximum Distances')
ax[1].set_xticks(range(len(tick_labels)))
ax[1].set_yticks(range(len(tick_labels)))
ax[1].set_xticklabels(tick_labels, rotation=45, ha='right')
ax[1].set_yticklabels(tick_labels)

# Set color limits for consistency
vmin = min(min_distances.min(), max_distances.min())
vmax = max(min_distances.max(), max_distances.max())
min_dist_plot.set_clim(vmin, vmax)
max_dist_plot.set_clim(vmin, vmax)

# Add a color bar
fig.colorbar(min_dist_plot, ax=ax, orientation='vertical')

plt.show()
'''

# Visualize the feasibility of fitting a sphere of radius r to an opposition space
r = 0.05 # object radius, in meter
# Initialize the fit matrix
fit_matrix = np.zeros((15, 15))

# Iterate over the distance matrix and check if the sphere fits
for i in range(15):
    for j in range(15):
        min_dist = distance_matrix[i, j, 0]
        max_dist = distance_matrix[i, j, 1]

        # Check if the sphere fits
        if min_dist <= 2*r and max_dist >= 2*r:
            fit_matrix[i, j] = 1

# Visualizing the heatmap
'''
plt.figure(figsize=(10, 8))
plt.imshow(fit_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Sphere Fit in Opposition Space")
plt.xlabel("Tensor Index")
plt.ylabel("Tensor Index")
plt.show()
'''

### Save the max distance of all pairs of links and put them in ascending order
# Initialize a list to store (index pair, maximum distance)
pairs_with_distances = []

# Iterate and store pairs with distances
for i in range(15):
    for j in range(i + 1, 15):
        max_dist = distance_matrix[i, j, 1]
        pair_name = (tick_labels[i], tick_labels[j])
        pairs_with_distances.append((max_dist, pair_name))

# Sort by maximum distance
sorted_pairs = sorted(pairs_with_distances, key=lambda x: x[0])

# Print and save the sorted pairs
sorted_pairs_for_save = [] # list
for max_dist, pair_name in sorted_pairs:
    print(f"Distance: {max_dist}, Pair: {pair_name}")
    sorted_pairs_for_save.append((max_dist, pair_name))

# Convert to numpy array and save
np.save('sorted_pairs.npy', np.array(sorted_pairs_for_save, dtype=object))
# to access the distance of the first pair, use sorted_pairs_for_save[0][0];
# to access the pair name, use sorted_pairs_for_save[0][1];
# to access the first link name, use sorted_pairs_for_save[0][1][0];