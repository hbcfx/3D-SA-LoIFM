# 3D-SA-LoIFM
A Three-Dimensional Framework for Segmentation-Assisted Learning of Informative Feature Matching for Rigid Registration

The method requires at least 15GB GPU memory. For python environment, please see the requirement.txt file.

For the test example, please follow the steps as:
1. first download the model file:https://huggingface.co/hbcfx/3D-SA-LoIFM/resolve/main/last.ckpt, copy it int the example folder
 
2. cd 3D-SA-LoIFM

3. first run: [python test_demo.py --input_file_list example/input_info_test_0.json --ckpt_path example/last.ckpt --dump_dir output --gpus 1] to get corresponding points

4. then using [LiverUSCT_outlier.exe example/case00_US_ap.nii.gz  case00_CT.nii.gz output/testUS_00_var_sort.mps output/testCT_00_var_sort.mp output/test_00_C2U.tfm  output/test_00_C2U.nii.gz] to perform outliers rejection and get the final transform

5. add test_tfm.txt file, write the output/test_00_C2U.tfm in it and then run [computeTREFigure.exe example/test_moving.txt example/test_fixed.txt output/test_tfm.txt output/test_TRE.txt] to get the evaluation result


For your data used original CT and US data, please first preprocess as:

1. run the pose initialization: [LiverUSCT_pose.exe input_US_filename  input_US_mask_filename pose_NO outputdir/caseXX_US_ap.nii.gz outputdir/caseXX_US_mask_ap.nii.gz]. For pose_number parameters: 1:intercostal; 2subcostal; 3: left-lobe. If you do not know your test data's pose_number, you can try all three poses and finally choose the best one after registration.
2. put all the ct files and initialized us files in input_folder. Then cd dataset, run space and intensity normalization:[python preprocess_usct.py input_folder spcae_output_folder output_folder dataset_properties_file]. The dataset_properties_file is provided in the config folder
3. prepare your own input_info_test.json file as the example shows, that is you just need to write your test number in val_input_info.json file 

Finally, run the registration just like the test example steps

