import elkai
import torch
import numpy as np


def compute_dist(p, q):
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0]) ** 2))


def get_ref_reward(pointset):
    if isinstance(pointset, torch.cuda.FloatTensor) or isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            ret_matrix[i, j] = ret_matrix[j, i] = compute_dist(
                pointset[i], pointset[j]
            )

    q = elkai.solve_float_matrix(ret_matrix)  # Output: [0, 2, 1]

    distance = 0
    for i in range(num_points):
        distance += ret_matrix[q[i], q[(i+1) % num_points]]

    return distance
