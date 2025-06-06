import torch
import torch.nn.functional as F
import numpy as np
from batchgenerators.transforms import AbstractTransform
from einops.einops import rearrange

class GenerateCorrTransform3Test(AbstractTransform):
    def __init__(self,patch_size,scale, beta, data_key="data",seg_key="seg"):    #us_seg_size: [128/8*128/8*128/8]
        us_patch_size = patch_size[1]
        self.patch_size = patch_size
        self.new_ct_patch = (np.array(self.patch_size[0]) * np.array(scale[0])).astype(int).tolist()
        self.new_us_patch = (np.array(us_patch_size) * np.array(scale[1])).astype(int).tolist()
        self.us_size = self.new_us_patch[0] * self.new_us_patch[1] * self.new_us_patch[2]
        self.scale = scale

        indices = [torch.arange(s) for s in self.new_us_patch]
        indices = torch.meshgrid(*indices)

        indices_tensor = torch.stack(indices, dim=3).reshape(-1, 3)

        self.us_gird_points = indices_tensor

        self.beta = beta

        self.type = "GT3T"

    def __call__(self, **data_dict):
        aug_affine = {}
        #print(data_dict.keys())
        if 'aug_c' in data_dict.keys():
            aug_affine['aug_c'] = data_dict['aug_c']
        if 'aug_u' in data_dict.keys():
            aug_affine['aug_u'] = data_dict['aug_u']

        #print(data_dict['spacing'])
        new_spacing = data_dict['spacing']/torch.tensor(self.scale)

        mask1 = data_dict['mask1'].reshape(-1)


        us_gird_points_input = self.us_gird_points[mask1>0]
        data_dict.update({'gt2u': us_gird_points_input})  # 'hwd1_c'*1  stack input must be tuple of tensors


        us_grid_points = us_gird_points_input[:,0]*self.new_us_patch[1]*self.new_us_patch[2]+ us_gird_points_input[:,1]*self.new_us_patch[2]+us_gird_points_input[:,2]


        data_dict.update({'gt':torch.stack((us_grid_points,us_grid_points))})  # 'hwd1_c'*1  stack input must be tuple of tensors

        data_mask0 = rearrange(data_dict['mask0'], 'h w d -> (h w d)')

        var_flatten = rearrange(data_dict['var'], 'h w d -> (h w d)')[us_grid_points.long()]
        data_dict.update({'var_flatten': var_flatten})

        mask0_ids = torch.where(data_mask0 > 0)
        data_dict.update({'mask0_ids': mask0_ids})
        return data_dict

class ResampleSegTransform(AbstractTransform):
    def __init__(self, scale,seg_key="seg"):    #attention:: __init__ not _init_
        self.scale = scale
        self.seg_key = seg_key
        self.type = "ResampleSeg"

    def __call__(self, **data_dict):
        mask =[]

        if self.scale[0][0] == 1:
            data_dict['mask'] = data_dict[self.seg_key]
            return data_dict


        for m in range(len(data_dict[self.seg_key])):  #number of modalities
            mask.append(F.interpolate(torch.from_numpy(data_dict[self.seg_key][m])[None],scale_factor=self.scale[m],mode = 'nearest')[0])  #interpolate input must be NCDHW

        if 'var' in data_dict.keys():
            data_dict['var'] = F.interpolate(torch.from_numpy(data_dict['var'])[None][None],scale_factor=self.scale[1],mode = 'trilinear')[0][0]

        data_dict['mask'] = mask


        return data_dict

def buildAugmentor(augment_name,config):
    if augment_name == 'ResampleSeg':
        return ResampleSegTransform(config['AUG']["SCALE"])
    if augment_name == 'GT3T':
        return GenerateCorrTransform3Test(patch_size=config['LOADER']['PATCH_SIZE'],scale=config['AUG']["SCALE"],beta=config['LOSS']['BETA'])



