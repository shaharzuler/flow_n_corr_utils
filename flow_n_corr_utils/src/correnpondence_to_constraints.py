from typing import Tuple
import os

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

from three_d_data_manager import save_arr

from .utils.h5_utils import get_point_clouds_and_p_from_h5
from .utils.flow_utils import voxelize_flow, smooth_flow
from .confidence_matrix_manipulations import get_correspondence_from_p
from .visualization.flow_2d_visualization import disp_flow_as_arrows
from .utils.flow_utils import xyz3_to_3xyz, t3xy_to_xy3
from .flow_to_correspondence import get_flow_in_target_coords



class Corr2ConstraintsConvertor:
    def __init__(self) -> None:
        pass

    def convert_corr_to_constraints(
        self, correspondence_h5_path:str, k_nn:int, output_folder_path:str, 
        output_constraints_shape:Tuple, k_interpolate_sparse_constraints_nn:int=124, 
        confidence_matrix_manipulations_config={"axis":1, "remove_high_var_corr":False}, gt_flow_path:np.ndarray=None, orig_img_path:np.ndarray=None) -> str:
        
        template_point_cloud, unlabeled_point_cloud, p = get_point_clouds_and_p_from_h5(correspondence_h5_path)
                  
        correspondence_template_unlabeled, mask, variance = get_correspondence_from_p(unlabeled_point_cloud, p, confidence_matrix_manipulations_config)
        flow_template_unlabeled_naive = self.corr_to_flow(template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled)#, mask)
        flow_template_unlabeled = flow_template_unlabeled_naive.copy()
        flow_template_unlabeled[~mask] = np.nan

        smooth_flow_template_unlabeled = smooth_flow(template_point_cloud, flow_template_unlabeled, k_nn)
        voxelized_flow = voxelize_flow(smooth_flow_template_unlabeled, template_point_cloud, output_constraints_shape) 
        if orig_img_path is not None:
            orig_img = np.load(orig_img_path)
            # TODO add utils fn for normalizing img https://github.com/shaharzuler/four_d_ct_cost_unrolling/four_d_ct_cost_unrolling/src/dataset_handlers/seg_puller_cardio_dataset.py#L36
            min_ = orig_img.min()
            max_ = orig_img.max()
            img_norm = (orig_img-min_)/(max_-min_) 
        else:
            img_norm = np.zeros(shape=output_constraints_shape[:-1], dtype=float)+0.5

        self.save_constraints_sections_visualization(
            output_constraints_shape=output_constraints_shape, 
            confidence_matrix_manipulations_config=confidence_matrix_manipulations_config, 
            template_point_cloud=template_point_cloud, 
            mask=mask, 
            voxelized_flow=voxelized_flow, 
            flow_template_unlabeled_naive=flow_template_unlabeled_naive.copy(), 
            text="pred",
            orig_image=img_norm)
        
        if gt_flow_path is not None:
            gt_flow = np.load(gt_flow_path)
            _, gt_flow_in_target_coords = get_flow_in_target_coords(gt_flow, template_point_cloud)
            gt_voxelized_flow = voxelize_flow(gt_flow_in_target_coords, template_point_cloud, output_constraints_shape)
            error = np.linalg.norm((gt_flow_in_target_coords-flow_template_unlabeled_naive),2,axis=1)
            self.save_constraints_sections_visualization(
                output_constraints_shape=output_constraints_shape, 
                confidence_matrix_manipulations_config=confidence_matrix_manipulations_config, 
                template_point_cloud=template_point_cloud, 
                mask=np.ones_like(mask), 
                voxelized_flow=gt_voxelized_flow, 
                flow_template_unlabeled_naive=gt_flow_in_target_coords.copy(), 
                text="gt", 
                orig_image=img_norm)

            self.plot_error_vs_var(confidence_matrix_manipulations_config, variance, error)

            
        if k_interpolate_sparse_constraints_nn>1:
            for axis in range(voxelized_flow.shape[-1]):
                voxelized_flow = self._interpolate_knn_axis(k_interpolate_sparse_constraints_nn, voxelized_flow, axis)

        output_file_path = save_arr(output_folder_path, "constraints", voxelized_flow)
        return output_file_path

    def plot_error_vs_var(self, confidence_matrix_manipulations_config, variance, error): # TODO 
        plt.close()
        plt.scatter(variance, error)
        plt.xlabel("Variance")
        plt.ylabel("Error")
        plt.show()
        plt.savefig(os.path.join(confidence_matrix_manipulations_config["plot_folder"],"error_vs_var.jpg"))

    def save_constraints_sections_visualization(
        self, output_constraints_shape, confidence_matrix_manipulations_config, 
        template_point_cloud, mask, voxelized_flow, flow_template_unlabeled_naive, 
        text, orig_image):

        #TODO move to some utils

        arrows_disp = disp_flow_as_arrows(img=orig_image.copy(), seg=None, flow=xyz3_to_3xyz(voxelized_flow))
        flow_template_unlabeled_outliars = flow_template_unlabeled_naive.copy()
        flow_template_unlabeled_outliars[mask] = np.nan
        voxelized_flow_outliars = voxelize_flow(flow_template_unlabeled_outliars, template_point_cloud, output_constraints_shape) 
        arrows_disp_outliars = disp_flow_as_arrows(img=orig_image.copy(), seg=None, flow=xyz3_to_3xyz(voxelized_flow_outliars), arrow_scale_factor=1, emphesize=True) ### TODO extract method
        
        arrows_disp_avg = (arrows_disp+arrows_disp_outliars)/2

        plt.imshow(t3xy_to_xy3(arrows_disp_avg[0]))
        plt.savefig(os.path.join(confidence_matrix_manipulations_config["plot_folder"], f"constraints_sections_{text}.jpg"), dpi=1200)
    
    @staticmethod
    def corr_to_flow(point_cloud1:np.ndarray, point_cloud2:np.ndarray, correspondence12:np.ndarray) -> np.ndarray:
        point_cloud2_in_point_cloud1_coords = point_cloud2[correspondence12] # shape: [N,3]
        flow12 = point_cloud2_in_point_cloud1_coords - point_cloud1 # shape: [N,3]

        return flow12 # shape: [N,3]
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



