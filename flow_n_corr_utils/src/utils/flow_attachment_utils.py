import numpy as np
from sklearn.neighbors import NearestNeighbors

from three_d_data_manager import extract_segmentation_envelope


def _get_indices_closest_to_populated_vals(populated_indices:np.ndarray, envelope_indices:np.ndarray) -> np.ndarray:
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(populated_indices)
    nn_ind = neigh.kneighbors(envelope_indices, return_distance=False) # shape (N2, 1)
    closest_indices = populated_indices[nn_ind][:,0,:] # shape (N2, 3)
    return closest_indices

def _attach_flow_to_indices(sparse_flows_arr:np.ndarray, envelope_indices:np.ndarray, indices_closest_to_populated_vals:np.ndarray) -> np.ndarray:
    restored_flow_arr = np.empty(sparse_flows_arr.shape)
    restored_flow_arr[:] = np.nan
    for axis in range(3):
        restored_flow_arr[envelope_indices[:,0],envelope_indices[:,1],envelope_indices[:,2],axis] = sparse_flows_arr[indices_closest_to_populated_vals[:,0],indices_closest_to_populated_vals[:,1],indices_closest_to_populated_vals[:,2],axis]
    return restored_flow_arr

def attach_flow_between_segs(sparse_flows_arr:np.ndarray, seg_arr:np.ndarray) -> np.ndarray:
    """
    Takes arr with spars flow vals (2d constraints for example) based on one segmentation map (for example a smooth seg map),
    and moves it to each index closest neighbour on the second segmentation map.
    """
    envelope = extract_segmentation_envelope(seg_arr)
    populated_indices = np.array([*np.where(~np.isnan(sparse_flows_arr[:,:,:,0]))]).T # shape N1, 3
    envelope_indices = np.array([*np.where(envelope)]).T # shape N2, 3
    indices_closest_to_populated_vals = _get_indices_closest_to_populated_vals(populated_indices, envelope_indices) # shape N2, 3
    attached_flow_arr = _attach_flow_to_indices(sparse_flows_arr, envelope_indices, indices_closest_to_populated_vals)
    return attached_flow_arr
