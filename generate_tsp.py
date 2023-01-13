import torch
import numpy as np
import elkai

from tqdm import tqdm
from draw_tsp import draw_tsp


def get_tsp_solution(pointset):
    """
    elkai solution for tsp
    - https://github.com/fikisipi/elkai
    - parameters : point (x,y) sets
    - return : opt solution
    """
    def get_dist(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    num_points = len(pointset)
    matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i+1, num_points):
            matrix[i, j] = get_dist(pointset[i], pointset[j])
            matrix[j, i] = matrix[i, j]

    # output : [0, 2, 1]
    tsp = elkai.solve_float_matrix(matrix, runs=10)

    return matrix, torch.LongTensor(tsp)


def generator_tsp_coord(
    num_nodes: int,
    num_samples: int,
    min_range: int = 0,
    max_range: int = 1
):
    # (x, y): (0~1, 0~1)
    # shape of coord: (num_nodes, 2)
    data_set = []
    for _ in range(num_samples):
        coord = torch.FloatTensor(num_nodes, 2).uniform_(min_range, max_range)
        data_set.append(coord)  # shape of data_set: (num_nodes, )

    return data_set


if __name__ == '__main__':
    # test code
    torch.manual_seed(2021)

    num_samples = 10
    num_nodes = 40

    data_set = []
    for l in range(num_samples):
        x = torch.FloatTensor(num_nodes, 2).uniform_(0, 1)
        data_set.append(x)

    train_data_elkai = []
    train_data_elaki_length = []
    for i, pointset in enumerate(tqdm(data_set)):
        dist, tsp = get_tsp_solution(pointset)
        train_data_elkai.append(tsp)
        train_data_elaki_length.append(dist)

    print(f"train_data_elkai : {train_data_elkai}")
    print(f"train_data_elaki_length : {train_data_elaki_length}")

    draw_tsp(data_set[0], train_data_elkai[0])
    draw_tsp(data_set[5], train_data_elkai[5])
    draw_tsp(data_set[8], train_data_elkai[8])
