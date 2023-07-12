from typing import Tuple

import numpy as np

from three_d_data_manager import save_arr

from .utils.h5_utils import get_point_clouds_and_corr_from_h5
from .utils.flow_utils import voxelize_flow, smooth_flow



class Corr2ConstraintsConvertor:
    def __init__(self) -> None:
        pass

    def convert_corr_to_constraints(self, correspondence_h5_path:str, k_nn:int, output_folder_path:str, output_constraints_shape:Tuple) -> str:
        template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled = get_point_clouds_and_corr_from_h5(correspondence_h5_path)
        flow_template_unlabeled = self.flow_from_corr(template_point_cloud, unlabeled_point_cloud, correspondence_template_unlabeled)
        smooth_flow_template_unlabeled = smooth_flow(template_point_cloud, flow_template_unlabeled, k_nn)
        voxelized_flow = voxelize_flow(smooth_flow_template_unlabeled, template_point_cloud, output_constraints_shape) 
        output_file_path = save_arr(output_folder_path, "constraints", voxelized_flow)

        return output_file_path

    @staticmethod
    def flow_from_corr(point_cloud1:np.array, point_cloud2:np.array, correspondence12:np.array) -> np.array:
        point_cloud2_in_point_cloud1_coords = point_cloud2[correspondence12]
        flow12 = point_cloud2_in_point_cloud1_coords - point_cloud1
        return flow12

