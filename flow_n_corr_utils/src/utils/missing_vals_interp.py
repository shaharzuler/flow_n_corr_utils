from typing import Tuple

import numpy as np

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.qhull import QhullError
import patchify



def interp_missing_values(flow_field_axis:np.array, interpolator)->np.array:
    nan_indices = np.isnan(flow_field_axis)
    main_points_indices = np.logical_not(nan_indices)
    main_points_data = flow_field_axis[main_points_indices]
    if main_points_data.shape[0] == 0:
        flow_field_axis[nan_indices] = 0.
    elif main_points_data.shape[0] < 3:
        flow_field_axis[nan_indices] = main_points_data.mean()
    else:
        try: 
            interp = interpolator(list(zip(*main_points_indices.nonzero())), main_points_data) 
            flow_field_axis[nan_indices] = interp(*nan_indices.nonzero())
        except QhullError as e:
            flow_field_axis[nan_indices] = main_points_data.mean()
            print(f"Can't interpolate to fill nans: \n{e}")

    return flow_field_axis

def _interp_in_patches(flow:np.array, patches:np.array, axis:int, z_plane_i:int, unpatchify_output_x:int, unpatchify_output_y:int) -> None:
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch_with_nans = patches[i, j]
            patch_wo_nans = interp_missing_values(patch_with_nans, interpolator=LinearNDInterpolator)
            patches[i, j] = patch_wo_nans
    flow_for_axis = patchify.unpatchify(patches, (unpatchify_output_x, unpatchify_output_y))
    flow[:unpatchify_output_x,:unpatchify_output_y, z_plane_i, axis] = flow_for_axis
    return flow

def interp_to_fill_nans(flow:np.ndarray, patchify_step:int=8, patch_size_x:int=10, patch_size_y:int=10) -> None:
    
    unpatchify_output_x = flow.shape[0] - (flow.shape[0] - patch_size_x) % patchify_step
    unpatchify_output_y = flow.shape[1] - (flow.shape[1] - patch_size_y) % patchify_step

    for axis in range(3):
        print(f"axis {axis} out of 2")
        for z_plane_i in range(flow.shape[2]):
            if z_plane_i%10==0:
                print(f'z plane {z_plane_i} out of {flow.shape[2]}')
            z_plane = flow[:,:,z_plane_i,axis]
            patches = patchify.patchify(z_plane, (patch_size_x, patch_size_y), step=patchify_step)    
            flow = _interp_in_patches(flow, patches, axis, z_plane_i, unpatchify_output_x, unpatchify_output_y)
            unpatchify_dim_matches_scan_dim = (flow.shape[:2] == (unpatchify_output_x, unpatchify_output_y))
            if not(unpatchify_dim_matches_scan_dim):
                flow[unpatchify_output_x-2:, :, z_plane_i, axis] = interp_missing_values(flow[unpatchify_output_x-2:, :, z_plane_i, axis], interpolator=LinearNDInterpolator)
                flow[:, unpatchify_output_y-2:, z_plane_i, axis] = interp_missing_values(flow[:, unpatchify_output_y-2:, z_plane_i, axis], interpolator=LinearNDInterpolator)

    return flow

