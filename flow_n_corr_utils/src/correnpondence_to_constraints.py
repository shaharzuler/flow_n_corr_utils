from typing import Tuple
import os

import numpy as np
from matplotlib import pyplot as plt

from three_d_data_manager import save_arr

from .utils.h5_utils import get_point_clouds_and_p_from_h5
from .utils.flow_utils import voxelize_flow, smooth_flow, corr_cloud_to_flow_cloud, voxelize_and_visualize_3d_vecs
from .utils.image_utils import get_norm_img
from .utils.contours_flow_visualization import save_contour_flow_sections_visualization
from .flow_to_correspondence import get_flow_in_target_coords
from .confidence_matrix_manipulations import get_correspondence_from_p


# def voxelize_and_visualize_3d_vecs(vectors_cloud, point_cloud, output_shape, text_vis, output_arr_filename, output_folder, k=1, img_path=None):
#     voxelized_vectors = voxelize_flow(vectors_cloud, point_cloud, output_shape) 

#     img_norm = get_norm_img(img_path, output_shape)

#     save_contour_flow_sections_visualization(
#         output_constraints_shape=output_shape, 
#         plot_folder=output_folder, 
#         point_cloud=point_cloud, 
#         mask=None, 
#         voxelized_flow=voxelized_vectors, 
#         flow=vectors_cloud, 
#         text=text_vis,
#         orig_image=img_norm)

#     if k>1:
#         for axis in range(voxelized_vectors.shape[-1]):
#             voxelized_vectors = interpolate_from_flow_in_axis(k, voxelized_vectors, axis)
#     output_file_path = save_arr(output_folder, output_arr_filename, voxelized_vectors)

#     return output_file_path

# def get_norm_img(img_path, output_shape):
#     if img_path is not None:
#         orig_img = np.load(img_path)
#         img_norm = min_max_norm(orig_img)
#     else:
#         img_norm = np.zeros(shape=output_shape[:-1], dtype=float) + 0.5
#     return img_norm


class Corr2ConstraintsConvertor:
    def __init__(self) -> None:
        pass

    def convert_normals_cloud_to_voxelized_normals(self, normals, point_cloud, output_shape, plot_folder, img_path=None): # TODO: after save_constraints_sections_visualization is moved to some utils, this method should be in it's own class.
        
        return voxelize_and_visualize_3d_vecs(normals, point_cloud, output_shape, "normals", "normals", plot_folder, img_path=img_path)
        
        # voxelized_normals = voxelize_flow(normals, point_cloud, output_shape) 
        # if img_path is not None:
        #     orig_img = np.load(img_path)
        #     img_norm = min_max_norm(orig_img)
        # else:
        #     img_norm = np.zeros(shape=output_shape[:-1], dtype=float) + 0.5

        # save_contour_flow_sections_visualization(
        #     output_constraints_shape=output_shape, 
        #     plot_folder=plot_folder, 
        #     point_cloud=point_cloud, 
        #     mask=None, 
        #     voxelized_flow=voxelized_normals, 
        #     flow=normals, 
        #     text="normals",
        #     orig_image=img_norm)

        # output_file_path = save_arr(plot_folder, "normals", voxelized_normals)
        # return output_file_path

    def convert_corr_to_constraints(
        self, correspondence_h5_path:str, k_nn:int, output_folder_path:str, 
        output_constraints_shape:Tuple, k_interpolate_sparse_constraints_nn:int=124, 
        confidence_matrix_manipulations_config={"axis":1, "remove_high_var_corr":False}, gt_flow_path:np.ndarray=None, orig_img_path:np.ndarray=None) -> str:
        
        template_point_cloud, unlabeled_point_cloud, p = get_point_clouds_and_p_from_h5(correspondence_h5_path)
                  
        correspondence_template_unlabeled, mask, variance = get_correspondence_from_p(unlabeled_point_cloud, p, confidence_matrix_manipulations_config)
        flow_template_unlabeled_naive = corr_cloud_to_flow_cloud(template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled)#, mask)
        flow_template_unlabeled = flow_template_unlabeled_naive.copy()
        flow_template_unlabeled[~mask] = np.nan

        smooth_flow_template_unlabeled = smooth_flow(template_point_cloud, flow_template_unlabeled, k_nn)

        output_file_path = voxelize_and_visualize_3d_vecs(smooth_flow_template_unlabeled, template_point_cloud, output_constraints_shape, "pred", "constraints", output_folder_path, k=k_interpolate_sparse_constraints_nn, img_path=orig_img_path)
        
        # voxelized_flow = voxelize_flow(smooth_flow_template_unlabeled, template_point_cloud, output_constraints_shape) 
        # if orig_img_path is not None:
        #     orig_img = np.load(orig_img_path)
        #     img_norm = min_max_norm(orig_img) 
        # else:
        #     img_norm = np.zeros(shape=output_constraints_shape[:-1], dtype=float) + 0.5

        # save_contour_flow_sections_visualization(
        #     output_constraints_shape=output_constraints_shape, 
        #     plot_folder=confidence_matrix_manipulations_config["plot_folder"], 
        #     point_cloud=template_point_cloud, 
        #     mask=mask, 
        #     voxelized_flow=voxelized_flow, 
        #     flow=flow_template_unlabeled_naive.copy(), 
        #     text="pred",
        #     orig_image=img_norm)

        # if k_interpolate_sparse_constraints_nn>1:
        #     for axis in range(voxelized_flow.shape[-1]):
        #         voxelized_flow = interpolate_from_flow_in_axis(k_interpolate_sparse_constraints_nn, voxelized_flow, axis)

        # output_file_path = save_arr(output_folder_path, "constraints", voxelized_flow)


        if gt_flow_path is not None:
            gt_flow = np.load(gt_flow_path)
            _, gt_flow_in_target_coords = get_flow_in_target_coords(gt_flow, template_point_cloud)
            gt_voxelized_flow = voxelize_flow(gt_flow_in_target_coords, template_point_cloud, output_constraints_shape)
            error = np.linalg.norm((gt_flow_in_target_coords-flow_template_unlabeled_naive),2,axis=1)
            
            save_contour_flow_sections_visualization(
                output_constraints_shape=output_constraints_shape, 
                plot_folder=confidence_matrix_manipulations_config["plot_folder"], 
                point_cloud=template_point_cloud, 
                mask=np.ones_like(mask), 
                voxelized_flow=gt_voxelized_flow, 
                flow=gt_flow_in_target_coords.copy(), 
                text="gt", 
                orig_image=get_norm_img(orig_img_path, output_constraints_shape[:-1]))

            self.plot_error_vs_var(confidence_matrix_manipulations_config["plot_folder"], variance, error)

        
        return output_file_path

    def plot_error_vs_var(self, plot_folder, variance, error): # TODO 
        plt.close()
        plt.scatter(variance, error)
        plt.xlabel("Variance")
        plt.ylabel("Error")
        plt.show()
        plt.savefig(os.path.join(plot_folder,"error_vs_var.jpg"))

    # def save_constraints_sections_visualization(
    #     self, output_constraints_shape, plot_folder, 
    #     template_point_cloud, mask, voxelized_flow, flow, 
    #     text, orig_image):

    #     #TODO move to some utils

    #     arrows_disp = disp_flow_as_arrows(img=orig_image.copy(), seg=None, flow=xyz3_to_3xyz(voxelized_flow))
    #     flow_template_unlabeled_outliars = flow.copy()
    #     if mask is not None:
    #         flow_template_unlabeled_outliars[mask] = np.nan
    #     voxelized_flow_outliars = voxelize_flow(flow_template_unlabeled_outliars, template_point_cloud, output_constraints_shape) 
    #     arrows_disp_outliars = disp_flow_as_arrows(img=orig_image.copy(), seg=None, flow=xyz3_to_3xyz(voxelized_flow_outliars), arrow_scale_factor=1, emphesize=True) ### TODO extract method
        
    #     arrows_disp_avg = (arrows_disp+arrows_disp_outliars)/2

    #     plt.imshow(t3xy_to_xy3(arrows_disp_avg[0]))
    #     plt.savefig(os.path.join(plot_folder, f"constraints_sections_{text}.jpg"), dpi=1200)
    
    # @staticmethod
    # def _corr_to_flow(point_cloud1:np.ndarray, point_cloud2:np.ndarray, correspondence12:np.ndarray) -> np.ndarray: #TODO move to some utils
    #     point_cloud2_in_point_cloud1_coords = point_cloud2[correspondence12] # shape: [N,3]
    #     flow12 = point_cloud2_in_point_cloud1_coords - point_cloud1 # shape: [N,3]

    #     return flow12 # shape: [N,3]
        
    # def _interpolate_knn_axis(self, k_interpolate_sparse_constraints_nn:int, voxelized_flow:np.array, axis:int) -> np.array: #TODO duplication w 4dcostunrolling and 2ts_corr_main
    #     data_mask = np.isfinite(voxelized_flow[:,:,:,axis] )
    #     data_coords = np.array(np.where(data_mask)).T
    #     data_values = voxelized_flow[:,:,:,axis][data_mask]

    #     nan_mask = np.isnan(voxelized_flow[:,:,:,axis] )
    #     nan_coords = np.array(np.where(nan_mask)).T

    #     kdtree = cKDTree(nan_coords)
    #     distances, nn_indices = kdtree.query(data_coords, k=k_interpolate_sparse_constraints_nn)
    #     nan_coords_for_interp = nan_coords[nn_indices].reshape(-1,3)

    #     interpolated_values = griddata(data_coords, data_values, nan_coords_for_interp , method='linear')
    #     voxelized_flow[nan_coords_for_interp[:,0], nan_coords_for_interp[:,1], nan_coords_for_interp[:,2], axis ] = interpolated_values
    #     return voxelized_flow



