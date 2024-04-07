from typing import Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from .geometry_utils import calc_knn, flow_median_filter

def xyz3_to_3xyz(flow:np.ndarray) -> np.ndarray:
    return np.transpose(flow, (3,0,1,2))

def t3xyz_to_xyz3(flow:np.ndarray) -> np.ndarray:
    return np.transpose(flow, (1,2,3,0))

def corr_cloud_to_flow_cloud(point_cloud1:np.ndarray, point_cloud2:np.ndarray, correspondence12:np.ndarray) -> np.ndarray: 
    point_cloud2_in_point_cloud1_coords = point_cloud2[correspondence12] # shape: [N,3]
    flow12 = point_cloud2_in_point_cloud1_coords - point_cloud1 # shape: [N,3]

    return flow12 # shape: [N,3]

def voxelize_flow(flow:np.array, point_cloud:np.array, output_grid_shape:Tuple) -> np.array:
    template_idxs = np.round(point_cloud).astype(int)

    voxels_flow = np.empty(output_grid_shape)
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

def interpolate_from_flow_in_axis(k:int, voxelized_flow:np.array, axis:int) -> np.array:
    '''Take voxelized flow with nans/infs and interpolate flow values to nans/infs for voxels which are close enough to finite values'''
    data_mask = np.isfinite(voxelized_flow[:,:,:,axis] )
    data_coords = np.array(np.where(data_mask)).T
    data_values = voxelized_flow[:,:,:,axis][data_mask]

    nan_mask = np.isnan(voxelized_flow[:,:,:,axis] )
    nan_coords = np.array(np.where(nan_mask)).T

    kdtree = cKDTree(nan_coords)
    distances, nn_indices = kdtree.query(data_coords, k=k)
    nan_coords_for_interp = nan_coords[nn_indices].reshape(-1,3)

    interpolated_values = griddata(data_coords, data_values, nan_coords_for_interp , method='linear')
    voxelized_flow[nan_coords_for_interp[:,0], nan_coords_for_interp[:,1], nan_coords_for_interp[:,2], axis ] = interpolated_values
    return voxelized_flow

def smooth_flow(point_cloud:np.array, flow:np.array, k_nn:int):
    nn_idxs = calc_knn(point_cloud, k_nn)
    smooth_flow = flow_median_filter(flow, nn_idxs, k_nn) if (k_nn > 1) else flow # [N, 3]
    return smooth_flow
