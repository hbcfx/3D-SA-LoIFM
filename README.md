# 3D-SA-LoIFM
A Three-Dimensional Framework for Segmentation-Assisted Learning of Informative Feature Matching for Rigid Registration

1. first run test_demo.py --input_file_list  --ckpt_path --dump_dir --gpus 1 to get corresponding points

2. then using LiverUSCT_outlier.exe us_img ct_img us_points ct_points output_tfm_filename  output_ct_name to perform outliers rejection and get the final transform

3. run computeTRE.exe  to get the evaluation result
