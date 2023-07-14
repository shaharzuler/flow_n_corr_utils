from flow_n_corr_utils import Corr2ConstraintsConvertor


output_constraints_arr_path = Corr2ConstraintsConvertor().convert_corr_to_constraints(
    correspondence_h5_path="/home/shahar/cardio_corr/outputs/outputs_20230714_005246/sts_training_output_20230714_005849/inference_20230714_005909/model_inference.hdf5",
     k_nn=10, 
     output_folder_path="/home/shahar/cardio_corr/my_packages/flow_n_corr_utils_project/flow_n_corr_utils/outputs",
     output_constraints_shape=(200, 200, 136, 3),
     k_interpolate_sparse_constraints_nn=124 #a 5x5x5 pixels cube
     )


# TODO:
# method_args={
#     "corr_results_h5_paths": ["/home/shahar/cardio_corr/my_packages/sts_project/STS/outputs_20230712_121917/inference_20230712_121934/model_inference.hdf5"],
#     "knn": 100
#     }
# create_corresponcence_animation(method="knn", method_args=method_args, output_path="/home/shahar/cardio_corr/my_packages/flow_n_corr_utils_project/flow_n_corr_utils/outputs") 
