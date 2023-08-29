import os
from typing import Dict, Tuple

import scipy
from matplotlib import pyplot as plt
import numpy as np
import torch

from .visualization.flow_2d_visualization import disp_flow_as_arrows
from .utils.geometry_utils import knn

def get_correspondence_from_p(point_cloud:np.ndarray, p:np.ndarray, confidence_matrix_manipulations_config:Dict) -> Tuple[np.ndarray, np.ndarray]:
    if "remove_high_var_corr" in confidence_matrix_manipulations_config.keys() and confidence_matrix_manipulations_config["remove_high_var_corr"]:
        correspondence, mask = variance_based_argmax(
            point_cloud, 
            p, 
            axis=confidence_matrix_manipulations_config["axis"], 
            k=confidence_matrix_manipulations_config["k"], 
            variance_threshold=confidence_matrix_manipulations_config["variance_threshold"],
            plot_folder=confidence_matrix_manipulations_config["plot_folder"]
            )
    else:
        correspondence = np.argmax(p, axis=confidence_matrix_manipulations_config["axis"])
        mask = np.ones_like(correspondence).astype(bool)
    
    return correspondence, mask

def variance_based_argmax(point_cloud:np.ndarray, p:np.ndarray, axis:int, k:int, variance_threshold:float, plot_folder:str=None, num_hist_bins:int=250) -> np.ndarray:
    """
    Given p [NxN] matrix, the confidence of all argmax points of point_cloud [Nx3] is replaced by 
        np.nan if the variance between the point confidence and it's k nearest neighbors'
             confidence is above variance_threshold.
                The argmax with nans is returned.
    """

    k_nn = knn(torch.tensor(point_cloud), torch.tensor(point_cloud), k) # shape: [2, N*k]
    k_nn_2d = k_nn[1].reshape([p.shape[0], k]) # shape: [N, k]
    orig_indices_2d = k_nn[0].reshape([p.shape[0],k]) # shape: [N, k]

    naive_correspondence = np.argmax(p, axis=axis) # shape: N
    indices_closed_to_argmaxes_2d = k_nn_2d[naive_correspondence] # shape: [N, k]

    if axis == 0:
        nns_confs = p[indices_closed_to_argmaxes_2d, orig_indices_2d] # shape: [N, k]
    elif axis==1:
        nns_confs = p[orig_indices_2d ,indices_closed_to_argmaxes_2d] # shape: [N, k]
    
    nn_probs = scipy.special.softmax(nns_confs ,1)
    variance = np.var(nn_probs,1)

    var_based_mask = np.where(variance<variance_threshold, True, False)

    if plot_folder is not None:
        print(f"Confidence mean: {nns_confs.mean()}")
        print(f"Unique argmax values: {np.unique(naive_correspondence).shape[0]}")
        print(f"removed {(~var_based_mask).sum()} out of {naive_correspondence.shape[0]} indices from correspondence results")
        plt.close()
        plt.hist(variance, bins=num_hist_bins)
        plt.show()
        plt.savefig(os.path.join(plot_folder, "variance_histogram.jpg"), dpi=1200)
    
    return naive_correspondence, var_based_mask