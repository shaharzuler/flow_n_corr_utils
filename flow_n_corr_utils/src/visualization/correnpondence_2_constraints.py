import tempfile
import numpy as np
import os
from torch_cluster import knn
import torch
import re
import h5py
import scipy.ndimage

        
def flow_median_filter(flow, source_nn, k_nn):
    smooth_flow = np.zeros_like(flow)
    for point_idx in range(flow.shape[0]):
        smooth_flow[point_idx] = np.median(flow[source_nn[point_idx,:k_nn]], axis=0) #take median of each axis seperately
    return smooth_flow

def load_file_by_name(filename):
    full_file_path = os.path.join(INFERENCE_FOLDER_PATH, filename+".npy")
    # with h5py.File(os.path.join(INFERENCE_FOLDER_PATH,"model_inference.hdf5"), 'r') as f:
    #     arr = f[filename][:]
    arr = np.load(full_file_path)
    return arr

def load_file_by_key(key):
    with h5py.File(os.path.join(INFERENCE_FOLDER_PATH,"model_inference.hdf5"), 'r') as f:
        arr = f[key][:]
    return arr

def recalc_knn(source, k_nn):
    # print("recalc knn")
    source_neigh_idxs = knn(torch.tensor(source), torch.tensor(source), k_nn)[1,:].reshape(source.shape[0],k_nn).numpy()
    return source_neigh_idxs

from typing import List, Tuple


class Vector:
    ''' This should actually be a dataclass if Python version was >= 3.7 '''
    def __init__(self, x: Tuple[float,float], y: Tuple[float,float], z: Tuple[float,float]):
        self.x = x
        self.y = y
        self.z = z


import plotly.graph_objects as go
from typing import List
import open3d as o3d

def vis_o3d_w_plotly(
    pcds:List[o3d.open3d.geometry.PointCloud], 
    html_path:str, 
    meshes:List[o3d.geometry.TriangleMesh]=[], 
    vectors:List[Vector]=[]
    ):

    data = []
    
    for pcd in pcds:
        points=np.asarray(pcd.points)
        data.append(go.Scatter3d(
                x=points[:,0], y=points[:,1], z=points[:,2], 
                mode='markers',
                marker=dict(size=1)#, color=colors)
            ),)

    for mesh in meshes:
        mesh_points = np.asarray(mesh.vertices) 
        faces = np.asarray(mesh.triangles) 
        data.append(
            go.Mesh3d(x=mesh_points[:,0], 
                y=mesh_points[:,1],
                z=mesh_points[:,2],
                i=faces[:,0],
                j=faces[:,1],
                k=faces[:,2],)
        )

    for vector in vectors:
        data.append(
            go.Scatter3d(
                x=vector.x,
                y=vector.y,
                z=vector.z,
                marker = dict( size = 1,color = "rgb(84,48,5)"),
                line = dict( color = "rgb(84,48,5)", width = 6)
                )
                )


    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.write_html(f"{html_path}.html")


from ..h5_utils import get_key
from ..geometry_utils import calc_knn, flow_median_filter
import three_d_data_manager 

class Corr2ConstraintsConvertor:
    def __init__(self) -> None:
        pass

    def convert_corr_to_constraints(self, correspondence_h5_path:str, k_nn:int, output_path:str):
        os.makedirs(output_path, exist_ok=True)
        key = get_key(correspondence_h5_path)
        h5_data = h5py.File(correspondence_h5_path, 'r') 
        correspondence_template_unlabeled = h5_data[f"p_{key}"][:].argmax(axis=1)
        template_point_cloud = h5_data[f"source_{key}"][:]
        unlabeled_point_cloud = h5_data[f"source_{key}"][:]
        template_nn_idxs = calc_knn(template_point_cloud, K_NN)
        unlabeled_in_template_coords = unlabeled_point_cloud[correspondence_template_unlabeled]
        flow_template_unlabeled = unlabeled_in_template_coords - template_point_cloud
        smooth_flow_template_unlabeled = flow_median_filter(flow_template_unlabeled, template_nn_idxs, k_nn) # [N, 3]

        ftmp = tempfile.NamedTemporaryFile(suffix='.off', prefix='tmp', delete=False)
        tmp_fpath = ftmp.name
        ftmp.close()

        three_d_data_manager.write_off(tmp_fpath, template_point_cloud)
        voxels = three_d_data_manager.Mesh2VoxelsConvertor(tmp_fpath).padded_voxelized
        
        os.remove(tmp_fpath)




def main():
    output_dir = os.path.join(INFERENCE_FOLDER_PATH,"voxel_surface_flow_w_blur")
    h5_out_filename = os.path.join(output_dir, "voxel_surface_flow_w_blur.hdf5")
    os.makedirs(output_dir, exist_ok=True)
    f_out =  h5py.File(h5_out_filename, 'w')
    print(f"writing to {h5_out_filename}")
    for filename in os.listdir(INFERENCE_FOLDER_PATH):
        # full_file_path = os.path.join(INFERENCE_FOLDER_PATH, filename)
        # if filename.endswith(".npy"):
        if filename.endswith(".hdf5"):
            f_in = h5py.File(os.path.join(INFERENCE_FOLDER_PATH,filename), 'r')
            for key in f_in.keys():
                if "p_" in key and "28" in key and "25" in key:
            # if "p_" in filename:
                    print(filename, key)
                    # corr_ab = load_file_by_name(filename.replace(".npy", "")).argmax(axis=1) 
                    corr_ab = load_file_by_key(key).argmax(axis=1) 
                    # a = load_file_by_name(filename.replace(".npy", "").replace("p", "source"))
                    a = load_file_by_key(key.replace("p", "source"))
                    # b = load_file_by_name(filename.replace(".npy", "").replace("p", "target"))
                    b = load_file_by_key(key.replace("p", "target"))
                    # a_neigh_idxs = load_file_by_name(filename.replace(".npy", "").replace("p", "source_neigh_idxs")) # shape = 7168,27 [N,K]
                    if K_NN <= 27:
                        a_neigh_idxs = load_file_by_key(key.replace("p", "source_neigh_idxs")) # shape = 7168,27 [N,K]
                    else:
                        a_neigh_idxs = recalc_knn(a, K_NN)

                    b_in_a_coords = b[corr_ab]
                    flow_ab = b_in_a_coords - a
                    smooth_flow_ab = flow_median_filter(flow_ab, a_neigh_idxs, K_NN) # [7168, 3]


                    a_idxs = np.round(a).astype(int)
                    voxels_flow = np.zeros([3, 136, 200, 200]) # [3, 192,192,192]) 
                    voxels_flow[0, a_idxs[:,0], a_idxs[:,1], a_idxs[:,2]] = flow_ab[:,0] # smooth_flow_ab[:,0] ############# TESTINGGGGG TEMP TODO
                    voxels_flow[1, a_idxs[:,0], a_idxs[:,1], a_idxs[:,2]] = flow_ab[:,1] # smooth_flow_ab[:,1]
                    voxels_flow[2, a_idxs[:,0], a_idxs[:,1], a_idxs[:,2]] = flow_ab[:,2] # smooth_flow_ab[:,2]

                    f_out.create_dataset(
                        name=key.replace("p_", "voxel_surface_flow_").split(".")[0], 
                        data=voxels_flow, 
                        compression="gzip")

                    # blurred_voxels_flow = np.zeros_like(voxels_flow)
                    # for dim in range(3):
                    #     blurred_voxels_flow[dim,:,:,:] = scipy.ndimage.filters.gaussian_filter(input=voxels_flow[dim,:,:,:], sigma=3, truncate=6) #sigma=5, truncate=10)
                    #     avg_ratio = np.average(blurred_voxels_flow[dim,:,:,:][voxels_flow[dim,:,:,:]!=0]/voxels_flow[dim,:,:,:][voxels_flow[dim,:,:,:]!=0])
                    #     blurred_voxels_flow[dim,:,:,:] = blurred_voxels_flow[dim,:,:,:]/avg_ratio

                    # f_out.create_dataset(
                    #     name=key.replace("p_", "blurred_voxel_surface_flow_").split(".")[0], 
                    #     data=blurred_voxels_flow, 
                    #     compression="gzip")

                    # np.save(os.path.join(output_dir,filename.replace("p_", "voxel_surface_flow_")),voxels_flow)

                    # corr_ba = load_file_by_name(filename.replace(".npy", "")).argmax(axis=0) 
                    corr_ba = load_file_by_key(key).argmax(axis=0) 
                    
                    b_neigh_idxs = recalc_knn(b, K_NN)

                    a_in_b_coords = a[corr_ba]
                    flow_ba = a_in_b_coords - b
                    smooth_flow_ba = flow_median_filter(flow_ba, b_neigh_idxs, K_NN) # [7168, 3]

                    b_idxs = np.round(b).astype(int)
                    voxels_flow = np.zeros([3, 136, 200, 200])# [3, 192,192,192]) #### [3, 136, 200, 200] ? 
                    voxels_flow[0, b_idxs[:,0], b_idxs[:,1], b_idxs[:,2]] = flow_ba[:,0] # smooth_flow_ba[:,0] #TESTTTTT TEMP TODO
                    voxels_flow[1, b_idxs[:,0], b_idxs[:,1], b_idxs[:,2]] = flow_ba[:,1] # smooth_flow_ba[:,1]
                    voxels_flow[2, b_idxs[:,0], b_idxs[:,1], b_idxs[:,2]] = flow_ba[:,2] # smooth_flow_ba[:,2]

                    # nums = re.findall('[0-9]+', filename)
                    nums = re.findall('[0-9]+', key)

                    key_reverse = f"voxel_surface_flow_{nums[1]}_{nums[0]}.npy"

                    # with h5py.File(h5_filename, 'w') as f:
                    f_out.create_dataset(
                        name=key_reverse.split(".")[0], 
                        data=voxels_flow, 
                        compression="gzip")

                    # np.save(os.path.join(output_dir,filename_reverse),voxels_flow)

                    # blurred_voxels_flow = np.zeros_like(voxels_flow)
                    # for dim in range(3):
                    #     blurred_voxels_flow[dim,:,:,:] = scipy.ndimage.filters.gaussian_filter(voxels_flow[dim,:,:,:], sigma=3, truncate=6) #sigma=5, truncate=10)
                    #     avg_ratio = np.average(blurred_voxels_flow[dim,:,:,:][voxels_flow[dim,:,:,:]!=0]/voxels_flow[dim,:,:,:][voxels_flow[dim,:,:,:]!=0])
                    #     blurred_voxels_flow[dim,:,:,:] = blurred_voxels_flow[dim,:,:,:]/avg_ratio

                    # f_out.create_dataset(
                    #     name="blurred_"+key_reverse.split(".")[0], 
                    #     data=blurred_voxels_flow, 
                    #     compression="gzip")
    f_out.close()
    f_in.close()













K_NN = 1
INFERENCE_FOLDER_PATH = '/home/shahar/projects/omri/dfaust_allign/output/shape_corr/inference_results/2022_12_08_13_34_29_cube'
main()