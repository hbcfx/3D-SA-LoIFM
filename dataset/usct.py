from torch.utils.data import Dataset,DataLoader
from collections import OrderedDict
from scipy.ndimage import uniform_filter
import pytorch_lightning as pl
from dataset.augmentor import buildAugmentor
from utils.util import *
import SimpleITK as sitk
import numpy as np
def load_final_test_sample_from_id(input_path, id):
    ct_img_file = os.path.join(input_path, "case"+id+ "_CT.nii.gz")
    us_img_file = os.path.join(input_path, "case"+id+ "_US_ap.nii.gz")
    ct_img_mask_file = os.path.join(input_path, "case"+id+ "_CT_mask.nii.gz")
    us_img_mask_file = os.path.join(input_path, "case"+id+ "_US_mask_ap_c.nii.gz")

    ct_img = sitk.ReadImage(ct_img_file)
    us_img = sitk.ReadImage(us_img_file)
    ct_img_mask = sitk.ReadImage(ct_img_mask_file)
    us_img_mask = sitk.ReadImage(us_img_mask_file)

    ct_img_array = sitk.GetArrayFromImage(ct_img).transpose(2,1,0)
    us_img_array = sitk.GetArrayFromImage(us_img).transpose(2,1,0)

    window_size = (9, 9, 9)
    local_mean = uniform_filter(us_img_array, window_size)
    squared_diff = (us_img_array - local_mean) ** 2
    local_var_array = uniform_filter(squared_diff, window_size).astype(np.float32)
    local_var_array = (local_var_array>0.3).astype(np.float32)



    ct_img_array_mask = sitk.GetArrayFromImage(ct_img_mask).transpose(2,1,0)
    us_img_array_mask = sitk.GetArrayFromImage(us_img_mask).transpose(2,1,0)
    #!!!!!!!!!!!!!!!!!!!!!
    #local_var_array = (us_img_array_mask>0).astype(np.float32) #for comparison experiment without variance

    data = [ct_img_array[np.newaxis,:,:,:],us_img_array[np.newaxis,:,:,:]]  #input to datasetloader must be c h d w
    seg = [ct_img_array_mask[np.newaxis,:,:,:],us_img_array_mask[np.newaxis,:,:,:]]

    origin = [ct_img.GetOrigin(),us_img.GetOrigin()]
    spacing = [ct_img.GetSpacing(),us_img.GetSpacing()]


    data_dict = {'data':data,'seg':seg,'var':local_var_array,'origin':origin,'spacing':spacing,'key':id}

    return data_dict
def load_final_test_dataset(data_file):
    para = load_json(data_file)
    input_path = para['input_path']
    ids = para['data']


    dataset= []

    ids.sort()

    id_dict = {}
    for i in range(len(ids)):
        id_dict[ids[i]] = i

    tasks = [delayed(load_final_test_sample_from_id)( input_path, x) for x in ids]

    res = Parallel(n_jobs=-1)(tasks)

    print("load data successfully")

    for id in ids:
        dataset.append(res[id_dict[id]])
    return dataset
class USCTTestDataset(Dataset):
    def __init__(self,
                 data_file,
                 config,
                 pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None,
                 augment_fn=None):
        """
        Manage one scene(npz_path) of MegaDepth dataset.

        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()

        self._data = load_final_test_dataset(data_file)

        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.final_patch_size = config['LOADER']['FINAL_PATCH_SIZE']  # patch size after augmentation
        self.patch_size = config['LOADER']['PATCH_SIZE']
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = []
        for m in range(2):
            self.need_to_pad.append((np.array(self.patch_size[m]) - np.array(self.final_patch_size[m])).astype(int))

            if pad_sides is not None:
                if not isinstance(pad_sides, np.ndarray):
                    pad_sides = np.array(pad_sides)
                self.need_to_pad[m] += pad_sides
            self.pad_sides = pad_sides

        self.aug_funcs = augment_fn

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):

        data = []
        seg = []
        origin = []

        data_selected = self._data[idx]
        current_variance = data_selected['var']

        for j in range(2):
            current_patch_size = self.patch_size[j]

            current_data = data_selected['data'][j]
            current_data_seg = data_selected['seg'][j]
            need_to_pad = self.need_to_pad[j]
            current_origin = data_selected['origin'][j]
            spacing = data_selected['spacing'][j]

            # print("current_data:",current_data.shape)

            for d in range(3):
                if need_to_pad[d] + current_data.shape[d + 1] < current_patch_size[d]:
                    need_to_pad[d] = current_patch_size[d] - current_data.shape[d + 1]

            shape = current_data.shape[1:]

            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - current_patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - current_patch_size[1]
            lb_z = - need_to_pad[2] // 2
            ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - current_patch_size[2]

            # If the selected class is indeed not present then we fall back to random cropping. We can do that
            # because this case is extremely rare.
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

            bbox_x_ub = bbox_x_lb + current_patch_size[0]
            bbox_y_ub = bbox_y_lb + current_patch_size[1]
            bbox_z_ub = bbox_z_lb + current_patch_size[2]

            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)
            valid_bbox_z_lb = max(0, bbox_z_lb)
            valid_bbox_z_ub = min(shape[2], bbox_z_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            current_data = current_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                           valid_bbox_y_lb:valid_bbox_y_ub,
                           valid_bbox_z_lb:valid_bbox_z_ub]

            current_data_seg = current_data_seg[:, valid_bbox_x_lb:valid_bbox_x_ub,
                               valid_bbox_y_lb:valid_bbox_y_ub,
                               valid_bbox_z_lb:valid_bbox_z_ub]

            if j == 1:
                # print(current_variance.shape)
                current_variance = current_variance[valid_bbox_x_lb:valid_bbox_x_ub,
                                   valid_bbox_y_lb:valid_bbox_y_ub,
                                   valid_bbox_z_lb:valid_bbox_z_ub]

            # crop = [valid_bbox_z_lb,valid_bbox_y_lb,valid_bbox_x_lb] #as getarrayfromImage get an array of z y x
            crop = [valid_bbox_x_lb, valid_bbox_y_lb, valid_bbox_z_lb]  # as getarrayfromImage get an array of z y x

            # print("origin:",current_origin)

            current_origin = originalAfterCrop(current_origin, spacing, crop)
            # print("current_origin:",current_origin,spacing, crop)
            # print("current_data2:",current_data.shape) #for debug, debug resolved for

            case_all_data_donly = np.pad(current_data, ((0, 0),
                                                        (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                        (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                        (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                         self.pad_mode,
                                         **self.pad_kwargs_data)  # pad_mode must be edge for image data, or images will be zero!!!

            case_all_data_segonly = np.pad(current_data_seg, ((0, 0),
                                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                           'constant', **{'constant_values': 0})
            if j == 1:
                current_variance = np.pad(current_variance, ((-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                             (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                                          self.pad_mode,
                                          **self.pad_kwargs_data)  # pad_mode must be edge for image data!!!

            # crop = [-min(0, bbox_z_lb), -min(0, bbox_y_lb), -min(0, bbox_x_lb)]#as getarrayfromImage get an array of z y x
            crop = [min(0, bbox_x_lb), min(0, bbox_y_lb),
                    min(0, bbox_z_lb)]  # attention!!! not [-min(0, bbox_x_lb), -min(0, bbox_y_lb),-min(0, bbox_z_lb)]

            current_origin = originalAfterCrop(current_origin, spacing, crop)
            # print("current_origin2:",current_origin,spacing, crop,[bbox_x_lb, bbox_y_lb,bbox_z_lb],[lb_x,lb_y,lb_z])

            data.append(case_all_data_donly)
            seg.append(case_all_data_segonly)
            origin.append(current_origin)


        # this is bad. Slow. Better preallocate data and set with np.zeros(). But this way we don't have to know how
        # many color channels data has and how many seg channels there are

        data_cropped = {'data': data, 'seg': seg, 'var': current_variance, 'origin': origin,
                        'spacing': data_selected['spacing'], 'key': data_selected['key']}

        ###start data augmentation
        GT_fun = None
        for trans in self.aug_funcs:
            if 'GT' in trans.type:  # is "GT" or trans.type is "GT2" or trans.type is 'GTP' or trans.type is 'GT2P':
                # print("GenerateCorrTransform")
                GT_fun = trans
                continue
            data_cropped = trans(**data_cropped)  # not  trans(data_cropped)

        data_a = data_cropped['data']
        mask_a = data_cropped['mask']
        seg_a = data_cropped['seg']
        var_tensor = data_cropped['var']
        # weight = data_cropped['weight']
        if not isinstance(var_tensor, torch.Tensor):
            var_tensor = torch.from_numpy(var_tensor).float()

        for m in range(2):
            if not isinstance(data_a[m], torch.Tensor):
                data_a[m] = torch.from_numpy(data_a[m]).float()
            if not isinstance(seg_a[m], torch.Tensor):
                seg_a[m] = torch.from_numpy(seg_a[m]).float()
            if not isinstance(mask_a[m], torch.Tensor):
                mask_a[m] = torch.from_numpy(mask_a[m]).float()

        # if not isinstance(weight, torch.Tensor):
        # weight = torch.from_numpy(weight).float()

        data_dict_input = {}
        data_dict_input['image0'] = data_a[0]
        data_dict_input['image1'] = data_a[1]

        data_dict_input['seg0'] = torch.cat(((seg_a[0] == 1), (seg_a[0] == 2), (seg_a[0] == 3)), dim=0)  # cHDW
        data_dict_input['seg1'] = torch.cat(((seg_a[1] == 1), (seg_a[1] == 2), (seg_a[1] == 3)), dim=0)
        # print(data_dict_input['seg0'].shape,data_dict_input['seg1'].shape)# cHDW

        data_dict_input['mask0'] = mask_a[0][0]  # HDW
        data_dict_input['mask1'] = mask_a[1][0]  # HDW
        data_dict_input['var'] = var_tensor  # HDW

        ##################for fine =========
        if 'mask_f' in data_cropped.keys():
            seg_b = data_cropped['mask_f']
            for l in range(len(seg_b)):
                for m in range(2):
                    if not isinstance(seg_b[l, m], torch.Tensor):
                        seg_b[l, m] = torch.from_numpy(seg_b[l, m]).float()
                        # seg_b = torch.tensor(seg_b)

            # data_dict_input['mask0_f'] = seg_b[:, 0, 0]  # LHDW
            # data_dict_input['mask1_f'] = seg_b[:, 1, 0]  # LHDW
            data_dict_input['mask0_f'] = seg_b[0][0][0]  # LHDW
            data_dict_input['mask1_f'] = seg_b[0][1][0]  # LHDW
        ##################for fine =========

        # data_dict_input['weight'] = weight

        data_dict_input['origin'] = torch.tensor(data_cropped['origin']).type(torch.float32)
        data_dict_input['spacing'] = torch.tensor(data_cropped['spacing']).type(torch.float32)
        data_dict_input['key'] = data_cropped['key']

        if GT_fun is not None:
            data_dict_input = GT_fun(**data_dict_input)
        return data_dict_input

class USCTDataModule(pl.LightningDataModule):
    """
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config):
        super().__init__()

        # 1. data config
        # Train and Val should from the same data source
        file_list = load_json(args.input_file_list)
        self.test_input_file = None
        if 'test' in file_list.keys():
            self.test_input_file = file_list['test']

        self.config = config

        self.augment_fn_test = []
        for fn in config.AUG.TEST:
            self.augment_fn_test.append(buildAugmentor(fn, config))

        # 3.loader parameters
        self.test_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': True
        }
        # (optional) RandomSampler for debugging

        # misc configurations
        #self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.parallel_load_data = False
        self.seed = config.TRAINER.SEED  # 66

    def setup(self,stage='test'):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """
        self.test_dataset = USCTTestDataset(
            self.test_input_file,
            self.config,
            augment_fn=self.augment_fn_test)

    def test_dataloader(self, *args, **kwargs):
        return DataLoader(self.test_dataset, sampler=None, **self.test_loader_params)
