import numpy as np

from three_d_data_manager import save_arr

from .flow_utils_basic import interpolate_from_flow_in_axis, voxelize_flow

from .image_utils import get_norm_img
from .contours_flow_visualization import save_contour_flow_sections_visualization



def voxelize_and_visualize_3d_vecs(vectors_cloud, point_cloud, output_shape, text_vis, output_arr_filename, output_folder, k=1, img_path=None):
    voxelized_vectors = voxelize_flow(vectors_cloud, point_cloud, output_shape) 

    img_norm = get_norm_img(img_path, output_shape)

    save_contour_flow_sections_visualization(
        output_shape=output_shape, 
        output_folder=output_folder, 
        point_cloud=point_cloud, 
        mask=None, 
        voxelized_flow=voxelized_vectors, 
        flow=vectors_cloud, 
        text=text_vis,
        orig_image=img_norm)

    if k>1:
        for axis in range(voxelized_vectors.shape[-1]):
            voxelized_vectors = interpolate_from_flow_in_axis(k, voxelized_vectors, axis)
    output_file_path = save_arr(output_folder, output_arr_filename, voxelized_vectors)

    return output_file_path
