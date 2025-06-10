# 3D-SA-LoIFM
A Three-Dimensional Framework for Segmentation-Assisted Learning of Informative Feature Matching for Rigid Registration
cd 3D0SA-LoIFM

1. first run: [python test_demo.py --input_file_list example/input_info_test_0.json --ckpt_path --dump_dir output --gpus 1] to get corresponding points

2. then using [LiverUSCT_outlier.exe example/case00_US_ap.nii.gz  case00_CT.nii.gz output/testUS_00_var_sort.mps output/testCT_00_var_sort.mp output/test_00_C2U.tfm  utput/test_00_C2U.nii.gz] to perform outliers rejection and get the final transform

3. run computeTREFigure.exe example/test_moving.txt example/test_fixed.txt output/test_tfm.txt output/test_00_TRE.txt to get the evaluation result
