import numpy as np
from sklearn.neighbors import NearestNeighbors

def flow_to_corr(flow:np.ndarray, target_pc:np.ndarray, source_pc:np.ndarray) -> np.ndarray: # shapes [x,y,z,3], [N, 3], [N, 3]
    """
    Extracts correspondence from pc pair and flow field.
    flow.shape = [x,y,z,3]
    target_pc.shape = [N, 3]
    source_pc.shape = [N, 3]
    The returned indices are the indices of the source_pc corresponding to target_pc
    """
    source_as_int = np.round(source_pc).astype(int)
    flow_in_source_coords = flow[source_as_int[:,0], source_as_int[:,1], source_as_int[:,2], :]
    target_estimated_coords = source_as_int + flow_in_source_coords
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_estimated_coords)
    distances, indices = nbrs.kneighbors(target_pc)

    # now target_estimated_coords[indices][:,0,:] =~ target_pc
    # now source_pc[indices][:,0,:] corresponds to target_pc

    return indices[:,0]

