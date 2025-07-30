import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
import os
import sys
from utils.util import *



def SpacingNormalization(input_folder,data_identifiers,seg_identifiers,output_folder,patch_size):

    dataset = {i:sitk.ReadImage(join(input_folder,i)) for i in data_identifiers}

    data_itk = [dataset[i] for i in data_identifiers]

    spacings = [np.array(d.GetSpacing()) for d in data_itk]

    sizes = [np.array(d.GetSize()) for d in data_itk]

    new_spacings = [np.array(i)* np.array(j).astype(np.float32) / np.array(patch_size).astype(np.float32)  for i, j in zip(spacings, sizes)]

    target_spacing = np.percentile(np.vstack(new_spacings), 50, 0)


    for i,i_seg in zip(data_identifiers,seg_identifiers):
        print(i,i_seg)
        img = dataset[i]


        original_spacing = img.GetSpacing()

        #img_array = sitk.GetArrayFromImage(img).astype(np.uint8)

        new_shape = np.round(np.array(original_spacing) * np.array(img.GetSize()).astype(float)/target_spacing)
        print(i,new_shape)

        ref = sitk.Image(int(new_shape[0]),int(new_shape[1]),int(new_shape[2]),sitk.sitkUInt8)

        ref.SetOrigin(img.GetOrigin())
        ref.SetSpacing(target_spacing)
        ref.SetDirection(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkBSpline)
        crop_resample_img = resampler.Execute(img)

        writer = sitk.ImageFileWriter()

        writer.SetFileName(join(output_folder,i))

        writer.Execute(crop_resample_img)





        seg = sitk.ReadImage(join(input_folder, i_seg))
        seg = sitk.Cast(seg, sitk.sitkUInt8)


        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        crop_resample_seg = resampler.Execute(seg)
        writer.SetFileName(join(output_folder,i_seg))

        writer.Execute(crop_resample_seg)

def SpacingNormalizationForTest(input_folder,data_identifiers,seg_identifiers,output_folder,target_spacing):


    dataset = {i:sitk.ReadImage(join(input_folder,i)) for i in data_identifiers}

    for i,i_seg in zip(data_identifiers,seg_identifiers):
        print(i,i_seg)
        img = dataset[i]


        original_spacing = img.GetSpacing()

        #img_array = sitk.GetArrayFromImage(img).astype(np.uint8)

        new_shape = np.round(np.array(original_spacing) * np.array(img.GetSize()).astype(float)/target_spacing)
        print(i,new_shape)

        ref = sitk.Image(int(new_shape[0]),int(new_shape[1]),int(new_shape[2]),sitk.sitkUInt8)

        ref.SetOrigin(img.GetOrigin())
        ref.SetSpacing(target_spacing)
        ref.SetDirection(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetInterpolator(sitk.sitkBSpline)
        crop_resample_img = resampler.Execute(img)

        writer = sitk.ImageFileWriter()

        writer.SetFileName(join(output_folder,i))

        writer.Execute(crop_resample_img)

        seg = sitk.ReadImage(join(input_folder, i_seg))
        seg = sitk.Cast(seg, sitk.sitkUInt8)


        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        crop_resample_seg = resampler.Execute(seg)
        writer.SetFileName(join(output_folder,i_seg))

        writer.Execute(crop_resample_seg)
def IntensityNormalization(input_folder,data_identifiers,seg_identifiers,output_folder,dataset_properties, scheme = "CT",use_nonzero_mask=False):

    dataset = {i: [sitk.ReadImage(join(input_folder,i)), sitk.ReadImage(join(input_folder,j))] for i,j in zip(data_identifiers,seg_identifiers)}

    if not use_nonzero_mask:
        properties = load_pickle(dataset_properties)
        intensityproperties = properties['intensityproperties']

    for i,j in zip(data_identifiers,seg_identifiers):
        print(i)
        raw_img = dataset[i][0]
        label_img = dataset[i][1]
        data = sitk.GetArrayFromImage(raw_img)[np.newaxis, :, :, :].astype(np.float32)
        seg = sitk.GetArrayFromImage(label_img)[np.newaxis, :, :, :].astype(np.float32)

        original_spacing = raw_img.GetSpacing()

        print("normalization...")

        for c in range(len(data)):
            if scheme == "CT":
                # clip to lb and ub from train data foreground and use foreground mn and sd from training data
                assert intensityproperties is not None, "if there is a CT then we need intensity properties"
                mean_intensity = intensityproperties[c]['mean']
                std_intensity = intensityproperties[c]['sd']
                lower_bound = intensityproperties[c]['percentile_00_5']
                upper_bound = intensityproperties[c]['percentile_99_5']
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                data[c] = (data[c] - mean_intensity) / std_intensity
                pad_value = (lower_bound - mean_intensity) / std_intensity

            elif scheme == "CT2":
                # clip to lb and ub from train data foreground, use mn and sd form each case for normalization
                assert intensityproperties is not None, "if there is a CT then we need intensity properties"
                lower_bound = intensityproperties[c]['percentile_00_5']
                upper_bound = intensityproperties[c]['percentile_99_5']
                mask = (data[c] > lower_bound) & (data[c] < upper_bound)
                data[c] = np.clip(data[c], lower_bound, upper_bound)
                mn = data[c][mask].mean()
                sd = data[c][mask].std()
                data[c] = (data[c] - mn) / sd
                pad_value = (lower_bound - mn) / sd

                if use_nonzero_mask:
                    data[c][seg[-1] < 0] = 0
            else:
                if use_nonzero_mask:
                    mask = data[c] != 0
                    mask = binary_fill_holes(mask)
                else:
                    mask = np.ones(seg.shape[1:], dtype=bool)
                #data[c][mask] = (data[c][mask] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
                #data[c][mask == 0] = 0
                print(data[c][mask].mean(),data[c][mask].std())
                data[c] = (data[c] - data[c][mask].mean()) / (data[c][mask].std() + 1e-8)
        print("normalization done")

        data_final = sitk.GetImageFromArray(data[0,:,:,:])


        data_final.SetSpacing(raw_img.GetSpacing())
        data_final.SetOrigin(raw_img.GetOrigin())
        data_final.SetDirection(raw_img.GetDirection())



        sitk.WriteImage(data_final, os.path.join(output_folder,i))
        sitk.WriteImage(label_img, os.path.join(output_folder,j))


if __name__ == "__main__":
    input_folder = sys.argv[1]
    space_output_folder = sys.argv[2]
    output_folder = sys.argv[3]
    dataset_properties = sys.argv[4]

    maybe_mkdir_p()

    ct_data_identifiers = np.sort(list(set([x for x in listdir(input_folder) if '_CT.nii.gz' in x and 'mask' not in x])))
    ct_seg_identifiers = np.sort(list(set([x for x in listdir(input_folder) if '_CT_mask' in x])))
    SpacingNormalizationForTest(input_folder,ct_data_identifiers,ct_seg_identifiers,space_output_folder,target_spacing=[1.84684,1.84694,1.67969]) # cannot use parrel

    IntensityNormalization(space_output_folder,ct_data_identifiers,ct_seg_identifiers,output_folder,dataset_properties,scheme='CT',use_nonzero_mask=False)

    us_data_identifiers = np.sort(list(set([x for x in listdir(input_folder) if '_US_ap.nii.gz' in x and 'mask' not in x])))
    us_seg_identifiers = np.sort(list(set([x for x in listdir(input_folder) if '_US_mask_ap' in x])))
    SpacingNormalizationForTest(input_folder,us_data_identifiers,us_seg_identifiers,space_output_folder,target_spacing=[1.42495,1.70439, 1.64621]) # cannot use parrel

    #the ultrasound image normalization no not deed intensity_property file
    print(us_data_identifiers)
    IntensityNormalization(space_output_folder,us_data_identifiers,us_seg_identifiers,output_folder,dataset_properties,scheme='MR',use_nonzero_mask=True)