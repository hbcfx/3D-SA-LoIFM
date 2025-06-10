import pytorch_lightning as pl
from model.networks.lofmr.lofmrCNNPoint_trans import LoFMRCNNPoint
from utils.profiler import PassThroughProfiler
from utils.xmlpointIO import pointxml
from utils.util import *
class PL_LoFMRCNNPoint(pl.LightningModule):
    def __init__(self, config,  loss=0,pretrained_ckpt=None, pretrained_ckpt2=None,profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = LoFMRCNNPoint(config=self.config['LOFMR'])

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        # Testing
        self.dump_dir = dump_dir
        if self.dump_dir is not None:
            maybe_mkdir_p( self.dump_dir)

    def nogt_test(self, batch):
        aug_affine = {}
        if 'aug_c' in batch.keys():
            aug_affine['aug_c'] = batch['aug_c']
        if 'aug_u' in batch.keys():
            aug_affine['aug_u'] = batch['aug_u']

        for b in range(len(batch['key'])):
            b_mask = (batch['m_bids'] == b)
            index2 = torch.where(batch['var_flatten'][b][batch['j_ids'][b_mask]] > 0.3)
            a, a_index = torch.sort(batch['mconf'][b_mask][index2].cpu())
            l0_list = batch['landmarkphy_0'][b][index2][a_index]
            l1_list = batch['landmarkphy_1'][b][index2][a_index]
            pointxml(l0_list, join(self.dump_dir,
                                   "testCT_" + batch['key'][b] + "_var_sort.mps"))
            pointxml(l1_list, join(self.dump_dir,
                                   "testUS_" + batch['key'][b] + "_var_sort.mps"))

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFMR"):
            self.matcher(batch)
            self.nogt_test(batch)
        return {
            'l0': batch['landmarkphy_0'],
            'l1': batch['landmarkphy_1'],
            'key': batch['key']
        }

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        return

