import h5py

def get_key(h5_correnpondence_path):
    h5_data = h5py.File(h5_correnpondence_path, 'r') 
    key_arr = h5_data['key'][:]
    key_str = f"{key_arr[0]}_{key_arr[1]}"
    return key_str