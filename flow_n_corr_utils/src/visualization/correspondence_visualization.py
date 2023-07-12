# TODO refactor from 
# /home/shahar/projects/omri/dfaust_allign/other_models_git/DPC/offline_post_processing_inference/post_process_inference_by_knn_surface_acc_consecutive_final_sep22.py
# once I have an actual sequence!

from typing import List
import os

import h5py

class CorrAnimationCreator:
    def __init__(self, args) -> None:
        pass

    def create_animation(self, output_path:str):
        os.makedirs(output_path, exist_ok=True)
        

class CorrKNNAnimationCreator(CorrAnimationCreator):
    def __init__(self, corr_results_h5_paths:List[str], knn:int) -> None:
        self.corr_results_h5_paths = corr_results_h5_paths #all paths are currently assumed to have the ame source shape and are in the correct order.
        self.knn = knn
    
    def create_animation(self, output_path:str):
        super().create_animation(output_path)
        all_p = {}
        keys = []
        for h5_path in self.corr_results_h5_paths:
            h5_data = h5py.File(h5_path, 'r') 
            key_arr = h5_data['key'][:]
            key_str = f"{key_arr[0]}_{key_arr[1]}"
            all_p[key_str] = h5_data[f"p_{key_str}"][:]
            keys.append(key_str)


        print(1)


def create_corresponcence_animation(method:str, method_args:dict, output_path:str) -> str:
    if method == "knn":
        CorrKNNAnimationCreator(**method_args).create_animation(output_path)

