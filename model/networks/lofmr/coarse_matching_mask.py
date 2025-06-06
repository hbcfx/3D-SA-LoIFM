import torch
import torch.nn as nn
from einops.einops import rearrange
from utils.util import *
INF = 1e9
import copy

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, :, :, :,:, :b] = v
    m[:, :, :, :,:,:, :b] = v

    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v
    m[:, :, :, :,:, -b:] = v
    m[:, :, :, :,:,:, -b:] = v

def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v
    m[:, :, :, :, :, :bd] = v
    m[:, :, :, :,:, :, :bd] = v

    h0s, w0s, d0s = p_m0.sum(1).max(-1)[0].max(-1)[0].int(), p_m0.sum(2).max(-1)[0].max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].max(-1)[0].int()  #[0] represents values
    h1s, w1s, d1s = p_m1.sum(1).max(-1)[0].max(-1)[0].int(), p_m1.sum(2).max(-1)[0].max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].max(-1)[0].int()

    #print( h0s, w0s, d0s,h1s, w1s, d1s)
    for b_idx, (h0, w0,d0, h1, w1, d1) in enumerate(zip(h0s, w0s, d0s, h1s, w1s,d1s)):
        #print(b_idx,(h0, w0,d0, h1, w1, d1))

        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :,:, d0 - bd:] = v
        m[b_idx, :, :,:, h1 - bd:] = v
        m[b_idx, :, :,:, :, w1 - bd:] = v
        m[b_idx, :, :,:, :, :,d1 - bd:] = v

#no useage
def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s, d0s = p_m0.sum(1).max(-1)[0].max(-1)[0],p_m0.sum(2).max(-1)[0].max(-1)[0], p_m0.sum(-1).max(-1)[0].max(-1)[0]
    h1s, w1s, d1s = p_m1.sum(1).max(-1)[0].max(-1)[0],p_m1.sum(2).max(-1)[0].max(-1)[0], p_m1.sum(-1).max(-1)[0].max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s* d0s, h1s * w1s *d1s], -1), -1)[0])
    return max_cand

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z
class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['THR']   #probability threshold
        self.border_rm = config['BORDER_RM']
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['TRAIN_COARSE_PERCENT']
        self.train_pad_num_gt_min = config['TRAIN_PAD_NUM_GT_MIN']

        # we provide 2 options for differentiable matching
        self.match_type = config['MATCH_TYPE']
        if self.match_type == 'dual_softmax' or self.match_type == 'softmax' or self.match_type == 'sigmoid':
            self.temperature = config['DSMAX_TEMPERATURE']

        elif self.match_type == 'sinkhorn':
            '''try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")'''
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['SKH_INIT_BIN_SCORE'], requires_grad=True))
            self.skh_iters = config['SKH_ITERS']
            self.skh_prefilter = config['SKH_PREFILTER']
        else:
            raise NotImplementedError()

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        #print(data['mask0_ids'][0].shape,feat_c0.shape,feat_c1.shape)


        #print(feat_c0.shape,feat_c1.shape)

        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        '''if mask_c1 is not None:
            mask_nonvalid = ~(torch.ones(mask_c0[..., None].shape).to(mask_c1.device) * mask_c1[:, None]).bool() #for training
            mask_nonvalid = ~(mask_c0[..., None].to(mask_c1.device) * mask_c1[:, None]).bool() #for testing

            data.update({'mask_nonvalid': mask_nonvalid.clone()})'''

        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature

            for b in range(N):
                data_mask0 = rearrange(data['mask0'][b], 'h w d -> (h w d)')[mask_c0[b]]
                var_mask = rearrange(data['var'][b], 'h w d -> (h w d)')[data['gt'][0][1]]

                #print(data['key'],data['gt'][0][1].shape,torch.sum(mask_c1[b]> 0))
                #print(torch.sum(data['mask1'][b] > 0))
                var_filter_mask = ((data_mask0.unsqueeze(1) * var_mask)<1)  #only for threshold mask
                sim_matrix[b].masked_fill_(
                    var_filter_mask,
                    -INF)

            conf_matrix_match = conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            conf_matrix_2 = conf_matrix_match
            conf_matrix_match = conf_matrix_match
            #conf_matrix.masked_fill_(mask_nonvalid,1e-10) #we add it because the  log(conf) in the loss function, the zero value is not allowed
        elif self.match_type == 'softmax':
            #print(feat_c0.shape)
            #print("softmax is used")
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            negtive_mask = data['negtive_mask']
            corres_mask = data['corres_mask']
            neg_var_weight = data['neg_weight']
            corr_var_weight = data['corres_weight']

            var_filter_mask = copy.deepcopy(negtive_mask).fill_(1)
            var_filter_mask[negtive_mask] *= neg_var_weight
            var_filter_mask[corres_mask] *= corr_var_weight
            var_filter_mask = (var_filter_mask<1)
            sim_matrix.masked_fill_(
                var_filter_mask,
                -INF)

            conf_matrix = F.softmax(sim_matrix, 1)
            conf_matrix_2 = F.softmax(sim_matrix, 2)
            conf_matrix_match = conf_matrix*conf_matrix_2

            #conf_matrix_match = conf_matrix

        elif self.match_type == 'sigmoid':
            # print("softmax is used")
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature

            conf_matrix = F.sigmoid(sim_matrix)
            conf_matrix_match = conf_matrix
            conf_matrix_2 = conf_matrix

        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included
            #print('sinkhorn')
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            '''if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    1e-10)'''

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp() # [N, L+1,S+1]
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            conf_matrix_match = conf_matrix
            conf_matrix_2 = conf_matrix



        data.update({'conf_matrix': conf_matrix})# conf_matrix NLS  = Ai,j in Global multi-modal 2D/3D Registration via local descriptors learning

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix,conf_matrix_match,conf_matrix_2,sim_matrix, data))

        # estimate pose transform from the matches

        landmark_phy_0 = []
        landmark_phy_1 = []

        for bs in range(conf_matrix.shape[0]):
            mask = (data['m_bids'] == bs)
            # transform indexes to physical point:
            #coarse_matches['mkpts0_c'][mask]
            mkpts0_c = data['mkpts0_c'][mask]
            mkpts1_c = data['mkpts1_c'][mask]


            mkpts0_c_phy = mkpts0_c*data['spacing'][bs][0]+data['origin'][bs][0]
            mkpts1_c_phy = mkpts1_c*data['spacing'][bs][1]+data['origin'][bs][1]

            #print(mkpts0_c_phy.shape,mkpts1_c_phy.shape)
            landmark_phy_0.append(mkpts0_c_phy)
            landmark_phy_1.append(mkpts1_c_phy)

        data.update({'landmarkphy_0': torch.stack(landmark_phy_0)})
        data.update({'landmarkphy_1': torch.stack(landmark_phy_1)})


    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, conf_matrix_match,conf_matrix_2,sim_matrix,data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hwd0_c'][0],
            'w0c': data['hwd0_c'][1],
            'd0c': data['hwd0_c'][2],
            'h1c': data['hwd1_c'][0],
            'w1c': data['hwd1_c'][1],
            'd1c': data['hwd1_c'][2]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding
        #conf_matrix = data['mask0'][None] * data['mask1'].unsqueeze(1)
        #conf_matrix[conf_matrix > 0] = conf_matrix_input
        #mask =(conf_matrix>0.5)*(conf_matrix_match>0.25)  # 2. mutual nearest
        data.update({'conf_matrix_image': conf_matrix})# conf_matrix NLS  = Ai,j in Global multi-modal 2D/3D Registration via local descriptors learning

        mask = (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])*(conf_matrix_2 == conf_matrix_2.max(dim=2, keepdim=True)[0])

        #mask = mask * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])# unique

        '''mask = mask \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])'''



        #max function: for each x in dim,  data[x] = max[x,::::] and also return indices

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
       # mask_v, all_i_ids = mask.max(dim=1)
        b_ids, i_ids, j_ids = torch.where(mask)
        #i_ids = all_i_ids[b_ids, j_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]
        msim  = sim_matrix[b_ids, i_ids, j_ids]


        #print("m_conf",mconf)

        #print(b_ids, i_ids, j_ids)

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale = data['hwd0_i'][0] / data['hwd0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale

        i_ids_image = data['mask0_ids'][0][0][i_ids]
        j_ids_image = data['gt'][b_ids,1,j_ids]

        #for images
        mkpts0_c = torch.stack(
            [i_ids_image // (data['hwd0_c'][1]*data['hwd0_c'][2]),\
             (i_ids_image % (data['hwd0_c'][1]*data['hwd0_c'][2]))//data['hwd0_c'][2],\
             (i_ids_image % (data['hwd0_c'][1]*data['hwd0_c'][2])) %data['hwd0_c'][2]],\
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids_image // (data['hwd1_c'][1] * data['hwd1_c'][2]), \
             (j_ids_image % (data['hwd1_c'][1] * data['hwd1_c'][2])) // data['hwd1_c'][2], \
             (j_ids_image % (data['hwd1_c'][1] * data['hwd1_c'][2])) % data['hwd1_c'][2]],\
            dim=1) * scale1

        #print(mkpts0_c, mkpts1_c)




        #print(data['mask1'][b_ids,(mkpts1_c[:,0]/8).long(),(mkpts1_c[:,1]/8).long(),(mkpts1_c[:,2]/8).long()])

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'm_bids': b_ids,  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c,
            'mkpts1_c': mkpts1_c,
            'mconf': mconf,
            'msim': msim

        })
        return coarse_matches
