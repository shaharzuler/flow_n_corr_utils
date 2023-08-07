from typing import Tuple

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from three_d_data_manager import save_arr

from .utils.h5_utils import get_point_clouds_and_corr_from_h5
from .utils.flow_utils import voxelize_flow, smooth_flow



class Corr2ConstraintsConvertor:
    def __init__(self) -> None:
        pass

    def convert_corr_to_constraints(self, correspondence_h5_path:str, k_nn:int, output_folder_path:str, output_constraints_shape:Tuple, k_interpolate_sparse_constraints_nn:int=124) -> str:
        template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled = get_point_clouds_and_corr_from_h5(correspondence_h5_path)
        flow_template_unlabeled = self._flow_from_corr(template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled)
        smooth_flow_template_unlabeled = smooth_flow(template_point_cloud, flow_template_unlabeled, k_nn)
        voxelized_flow = voxelize_flow(smooth_flow_template_unlabeled, template_point_cloud, output_constraints_shape) 

        if k_interpolate_sparse_constraints_nn>1:
            for axis in range(voxelized_flow.shape[-1]):
                voxelized_flow = self._interpolate_knn_axis(k_interpolate_sparse_constraints_nn, voxelized_flow, axis)

        output_file_path = save_arr(output_folder_path, "constraints", voxelized_flow)
        return output_file_path

    def _interpolate_knn_axis(self, k_interpolate_sparse_constraints_nn:int, voxelized_flow:np.array, axis:int) -> np.array:
        data_mask = np.isfinite(voxelized_flow[:,:,:,axis] )
        data_coords = np.array(np.where(data_mask)).T
        data_values = voxelized_flow[:,:,:,axis][data_mask]

        nan_mask = np.isnan(voxelized_flow[:,:,:,axis] )
        nan_coords = np.array(np.where(nan_mask)).T

        kdtree = cKDTree(nan_coords)
        distances, nn_indices = kdtree.query(data_coords, k=k_interpolate_sparse_constraints_nn)
        nan_coords_for_interp = nan_coords[nn_indices].reshape(-1,3)

        interpolated_values = griddata(data_coords, data_values, nan_coords_for_interp , method='linear')
        voxelized_flow[nan_coords_for_interp[:,0], nan_coords_for_interp[:,1], nan_coords_for_interp[:,2], axis ] = interpolated_values
        return voxelized_flow

    @staticmethod
    def _flow_from_corr(point_cloud1:np.array, point_cloud2:np.array, correspondence12:np.array) -> np.array:
        point_cloud2_in_point_cloud1_coords = point_cloud2[correspondence12]
        flow12 = point_cloud2_in_point_cloud1_coords - point_cloud1
        return flow12

