import torch
from torch_cluster import knn
import numpy as np


def calc_knn(point_cloud:np.array, k_nn:int) -> np.array:
    num_points = point_cloud.shape[0]
    nn_idxs = knn(torch.tensor(point_cloud), torch.tensor(point_cloud), k_nn)[1,:].reshape(num_points, k_nn).numpy()
    return nn_idxs


def flow_median_filter(flow, nn_idxs, k_nn):
    smooth_flow = np.zeros_like(flow)
    n_points = flow.shape[0]
    for point_idx in range(n_points):
        smooth_flow[point_idx] = np.median(flow[nn_idxs[point_idx,:k_nn]], axis=0) # get median of each axis seperately
    return smooth_flow