import os

from matplotlib import pyplot as plt
import numpy as np

from ..visualization.flow_2d_visualization import disp_flow_as_arrows
from .flow_utils import xyz3_to_3xyz, voxelize_flow
from .image_utils import t3xy_to_xy3


def save_contour_flow_sections_visualization(output_shape, output_folder, point_cloud, mask, voxelized_flow, flow, text, orig_image):
    arrows_disp = disp_flow_as_arrows(img=orig_image.copy(), seg=None, flow=xyz3_to_3xyz(voxelized_flow))
    flow_template_unlabeled_outliars = flow.copy()
    if mask is not None:
        flow_template_unlabeled_outliars[mask] = np.nan
    voxelized_flow_outliars = voxelize_flow(flow_template_unlabeled_outliars, point_cloud, output_shape) 
    arrows_disp_outliars = disp_flow_as_arrows(img=orig_image.copy(), seg=None, flow=xyz3_to_3xyz(voxelized_flow_outliars), arrow_scale_factor=1, emphesize=True) ### TODO extract method
    
    arrows_disp_avg = (arrows_disp+arrows_disp_outliars)/2

    plt.imshow(t3xy_to_xy3(arrows_disp_avg[0]))
    plt.savefig(os.path.join(output_folder, f"constraints_sections_{text}.jpg"), dpi=1200)
