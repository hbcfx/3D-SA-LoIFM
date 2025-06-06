import torch
import torch.nn as nn
from einops.einops import rearrange

from model.networks.lofmr.backbone import build_backbone
from model.networks.lofmr.transformer import LocalFeatureTransformer
from model.networks.lofmr.coarse_matching_mask import CoarseMatching
from model.networks.lofmr.position_encoding import PositionEncodingSine
class LoFMRCNNPoint(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone_0 = build_backbone(config)
        self.backbone_1 = build_backbone(config)
        self.matcher = CoarseMatching(config['MATCH_COARSE'])

        self.pos_encoding = PositionEncodingSine(config['RESNETFPN']['FINAL_DIMS'])

        self.transformer_attention = LocalFeatureTransformer(config['COARSE'])


    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W,D)
                'image1': (torch.Tensor): (N, 1, H, W,D)
                'mask0'(optional) : (torch.Tensor): (N, H, W,D) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W,D)
            }
        """

        #print(data['image0'].shape)
        #print(data['def'].shape)

        #save original image size
        data.update({
            'bs': data['image0'].size(0),
            'hwd0_i': data['image0'].shape[2:], 'hwd1_i': data['image1'].shape[2:]
        })

        input_0 = torch.cat((data['image0'],data['seg0']),dim=1)
        input_1 = torch.cat((data['image1'],data['seg1']),dim=1)
        feats_c0 = self.backbone_0(input_0)
        feats_c1 = self.backbone_1(input_1)
        # 1. Local Feature CNN
        #with torch.no_grad():

        #feats_c0 = self.backbone_0(data['image0'])
        #feats_c1 = self.backbone_1(data['image1'])

        feats_c0 = self.pos_encoding(feats_c0)
        feats_c1 = self.pos_encoding(feats_c1)
        #save coarse image size
        data.update({
            'hwd0_c': feats_c0.shape[2:], 'hwd1_c': feats_c1.shape[2:]
        })

        # 2. transformer module
        # add featmap with positional encoding, then flatten it to sequence [N, HWD, C]
        feat_c0 = rearrange(feats_c0, 'n c h w d -> n (h w d) c')
        feat_c1 = rearrange(feats_c1, 'n c h w d -> n (h w d) c')

        point0_tensor = []
        point1_tensor = []
        b_id_tensor = []

        for b in range(data['bs']):
            point_feat_0 = feat_c0[b, data['mask0_ids'][0][0], :]  # only fit for one batch size
            point_feat_1 = feat_c1[b, data['gt'][0][1], :]
            point0_tensor.append(point_feat_0)
            point1_tensor.append(point_feat_1)
            b_id_tensor.append(torch.zeros(size=[point_feat_0.shape[0]]).fill_(b))

        point0_tensor = torch.cat(point0_tensor).unsqueeze(0)
        point1_tensor = torch.cat(point1_tensor).unsqueeze(0)

        point0_tensor, point1_tensor = self.transformer_attention(point0_tensor, point1_tensor)#point_num is the second channel, not batchsize


        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-3), data['mask1'].flatten(-3)
            #print(mask_c0.shape,mask_c1.shape)

        # 3. match coarse-level
        self.matcher(point0_tensor, point1_tensor, data, mask_c0=mask_c0, mask_c1=mask_c1)



    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
