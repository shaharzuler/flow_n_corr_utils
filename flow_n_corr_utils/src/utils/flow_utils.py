from typing import Tuple

import numpy as np

from .geometry_utils import calc_knn, flow_median_filter


def smooth_flow(point_cloud:np.array, flow:np.array, k_nn:int):
    nn_idxs = calc_knn(point_cloud, k_nn)
    smooth_flow = flow_median_filter(flow, nn_idxs, k_nn) if (k_nn > 1) else flow # [N, 3]
    return smooth_flow

def voxelize_flow(flow:np.array, point_cloud:np.array, output_constraints_shape:Tuple) -> np.array:
    template_idxs = np.round(point_cloud).astype(int)

    voxels_flow = np.empty(output_constraints_shape)
    voxels_flow[:] = np.nan

    voxels_flow[template_idxs[:,0], template_idxs[:,1], template_idxs[:,2], 0] = flow[:,0] 
    voxels_flow[template_idxs[:,0], template_idxs[:,1], template_idxs[:,2], 1] = flow[:,1] 
    voxels_flow[template_idxs[:,0], template_idxs[:,1], template_idxs[:,2], 2] = flow[:,2]

    return voxels_flow