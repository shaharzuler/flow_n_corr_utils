import numpy as np
import torch

from .utils.geometry_utils import knn

def flow_to_corr(flow:np.ndarray, target_pc:np.ndarray, source_pc:np.ndarray) -> np.ndarray: # shapes [x,y,z,3], [N, 3], [N, 3]
    """
    Extracts correspondence from pc pair and flow field.
    flow.shape = [x,y,z,3]
    target_pc.shape = [N, 3]
    source_pc.shape = [N, 3]
    The returned indices are the indices of the target_pc corresponding to source_pc, meaning: 
    source_estimated_coords[indices] =~ source_pc
    target_pc[indices] corresponds to source_pc
    """
    target_as_int = np.round(target_pc).astype(int)
    flow_in_target_coords = flow[target_as_int[:,0], target_as_int[:,1], target_as_int[:,2], :]
    source_estimated_coords = target_as_int + flow_in_target_coords
        
    indices = knn(torch.tensor(source_estimated_coords), torch.tensor(source_pc), 1)[1]
    
    return indices

