from flow_n_corr_utils import convert_corr_to_constraints


output_constraints_arr_path = convert_corr_to_constraints(
    correspondence_h5_path="/home/shahar/cardio_corr/outputs/outputs_20230817_164704/sts_training_output_20230817_170949/inference_20230818_062138/model_inference.hdf5",
    k_nn=1, 
    output_folder_path="/home/shahar/cardio_corr/my_packages/flow_n_corr_utils_project/flow_n_corr_utils/outputs2",
    output_constraints_shape=(200, 200, 136, 3),
    k_interpolate_sparse_constraints_nn=26, # a 3x3x3 pixels cube,
    confidence_matrix_manipulations_config={
                                "remove_high_var_corr": True,
                                "axis": 1,
                                "k": 5,#20,
                                "variance_threshold": 0.5E-5, # 0.00005,#
                                "plot_folder": ""
                            },
                            gt_flow_path= "/home/shahar/cardio_corr/outputs/synthetic_dataset42/thetas_0.0_0.0_rs_0.6_0.6_h_0.6_random_3_0.4_mask_True_blur_radious_7/flow_for_mask_thetas_0.0_0.0_rs_0.6_0.6_h_0.6_random_3_0.4_mask_True_blur_radious_7.npy",
                            orig_img_path="/home/shahar/cardio_corr/outputs/synthetic_dataset42/thetas_0.0_0.0_rs_0.6_0.6_h_0.6_random_3_0.4_mask_True_blur_radious_7/image_skewed_thetas_0.0_0.0_rs_0.6_0.6_h_0.6_random_3_0.4_mask_True_blur_radious_7.npy"
    )
