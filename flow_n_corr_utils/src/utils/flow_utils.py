from typing import Tuple

import numpy as np

from .geometry_utils import calc_knn, flow_median_filter

def xyz3_to_3xyz(flow:np.ndarray) -> np.ndarray:
    return np.transpose(flow, (3,0,1,2))

def t3xyz_to_xyz3(flow:np.ndarray) -> np.ndarray:
    return np.transpose(flow, (1,2,3,0))

def t3xy_to_xy3(img:np.ndarray) -> np.ndarray:
    return np.transpose(img, (1,2,0))
    
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

def crop_flow_by_mask_center(center, x, y, z, flow_field_rotated:np.array, orig_vertices_mean:np.array) -> np.array:
    start = (2*center - orig_vertices_mean).astype(int)
    end = (2*center - orig_vertices_mean + np.array((x, y, z))).astype(int)
    flow_field_cropped = flow_field_rotated[ start[0,0]:end[0,0], start[0,1]:end[0,1], start[0,2]:end[0,2], : ]
    return flow_field_cropped