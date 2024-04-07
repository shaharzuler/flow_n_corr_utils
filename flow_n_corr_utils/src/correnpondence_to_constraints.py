from typing import Tuple
import os

import numpy as np
from matplotlib import pyplot as plt

from .utils.flow_utils_basic import corr_cloud_to_flow_cloud, voxelize_flow, smooth_flow
from .utils.voxel_vis_utils import voxelize_and_visualize_3d_vecs
from .utils.h5_utils import get_point_clouds_and_p_from_h5

from .utils.image_utils import get_norm_img
from .utils.contours_flow_visualization import save_contour_flow_sections_visualization
from .flow_to_correspondence import get_flow_in_target_coords
from .confidence_matrix_manipulations import get_correspondence_from_p


def convert_corr_to_constraints(
    correspondence_h5_path:str, k_nn:int, output_folder_path:str, 
    output_constraints_shape:Tuple, k_interpolate_sparse_constraints_nn:int=124, 
    confidence_matrix_manipulations_config={"axis":1, "remove_high_var_corr":False}, gt_flow_path:np.ndarray=None, orig_img_path:np.ndarray=None) -> str:
    
    template_point_cloud, unlabeled_point_cloud, p = get_point_clouds_and_p_from_h5(correspondence_h5_path)
                
    correspondence_template_unlabeled, mask, variance = get_correspondence_from_p(unlabeled_point_cloud, p, confidence_matrix_manipulations_config)
    flow_template_unlabeled_naive = corr_cloud_to_flow_cloud(template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled)#, mask)
    flow_template_unlabeled = flow_template_unlabeled_naive.copy()
    flow_template_unlabeled[~mask] = np.nan

    smooth_flow_template_unlabeled = smooth_flow(template_point_cloud, flow_template_unlabeled, k_nn)

    output_file_path = voxelize_and_visualize_3d_vecs(smooth_flow_template_unlabeled, template_point_cloud, output_constraints_shape, "pred", "constraints", output_folder_path, k=k_interpolate_sparse_constraints_nn, img_path=orig_img_path)
    
    if gt_flow_path is not None:
        gt_flow = np.load(gt_flow_path)
        _, gt_flow_in_target_coords = get_flow_in_target_coords(gt_flow, template_point_cloud)
        gt_voxelized_flow = voxelize_flow(gt_flow_in_target_coords, template_point_cloud, output_constraints_shape)
        error = np.linalg.norm((gt_flow_in_target_coords-flow_template_unlabeled_naive),2,axis=1)
        
        save_contour_flow_sections_visualization(
            output_shape=output_constraints_shape, 
            output_folder=confidence_matrix_manipulations_config["plot_folder"], 
            point_cloud=template_point_cloud, 
            mask=np.ones_like(mask), 
            voxelized_flow=gt_voxelized_flow, 
            flow=gt_flow_in_target_coords.copy(), 
            text="gt", 
            orig_image=get_norm_img(orig_img_path, output_constraints_shape[:-1]))

        _plot_error_vs_var(confidence_matrix_manipulations_config["plot_folder"], variance, error)

    
    return output_file_path

def _plot_error_vs_var(plot_folder, variance, error):
    plt.close()
    plt.scatter(variance, error)
    plt.xlabel("Variance")
    plt.ylabel("Error")
    plt.show()
    plt.savefig(os.path.join(plot_folder,"error_vs_var.jpg"))


