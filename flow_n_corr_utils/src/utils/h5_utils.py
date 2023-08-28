from typing import Tuple

import numpy as np
import h5py

def get_key(h5_correnpondence_path):
    h5_data = h5py.File(h5_correnpondence_path, 'r') 
    key_arr = h5_data['key'][:]
    key_str = f"{key_arr[0]}_{key_arr[1]}"
    return key_str


def get_point_clouds_and_p_from_h5(correspondence_h5_path:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = get_key(correspondence_h5_path)
    h5_data = h5py.File(correspondence_h5_path, 'r') 
    p = h5_data[f"p_{key}"][:]
    template_point_cloud = h5_data[f"source_{key}"][:]
    unlabeled_point_cloud = h5_data[f"target_{key}"][:]
    return template_point_cloud, unlabeled_point_cloud, p 