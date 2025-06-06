from yacs.config import CfgNode as CN
_CN = CN()

##########################
_CN.PREPROCESS = CN()
_CN.PREPROCESS.CTLABEL = [1,2,3]
_CN.PREPROCESS.USLABEL = [2,3]

########################
_CN.LOADER = CN()
_CN.LOADER.BATCHSIZE = 2
_CN.LOADER.FINAL_PATCH_SIZE =[[192, 192, 128],[128, 128, 128]]#must be divided by SCALE
_CN.LOADER.PATCH_SIZE = [[192, 192, 128],[128, 128, 128]]
#################DATA AUGMENTATION##################
_CN.AUG = CN()
_CN.AUG.TEST = ['ResampleSeg','GT3T']
_CN.AUG.AFFINE = True
_CN.AUG.STRENGTH = 0.05
_CN.AUG.GAMMA_RANGE = (0.7, 1.5)
_CN.AUG.GAMMA_RETAIN = True
_CN.AUG.P_GAMMA = 0.3
_CN.AUG.NUM_THREADS = 12
_CN.AUG.NUM_CACHED_PER_THREAD = 1
_CN.AUG.SCALE = [[1.0/4.0,1.0/4.0,1.0/4.0],[1.0/4.0,1.0/4.0,1.0/4.0]]  #correponds to _CN.LOFMR.RESOLUTION

############  ↓  LOFMR Pipeline  ↓  ##############
_CN.LOFMR = CN()
_CN.LOFMR.BACKBONE_TYPE = 'ResNetFPN'
_CN.LOFMR.RESOLUTION = (4)  # options: [(8, 2), (16, 4)]
_CN.LOFMR.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.LOFMR.FINE_CONCAT_COARSE_FEAT = True

# 1. LOFMR-backbone (local feature CNN) config
_CN.LOFMR.RESNETFPN = CN()
_CN.LOFMR.RESNETFPN.INITIAL_DIM = 126
_CN.LOFMR.RESNETFPN.BLOCK_DIMS = [126, 192, 252]  # s1, s2, s3 #must be divided by 6, must be equal to the output of the _CN.LOFMR.RESNETFPN.BLOCK_DIMS
_CN.LOFMR.RESNETFPN.FINAL_DIMS = 192  # for final feature dims

_CN.LOFMR.UNET = CN()
_CN.LOFMR.UNET.INITIAL_DIM = 16
_CN.LOFMR.UNET.FINAL_DIMS = 192  # for final feature dims

# 2. LOFMR-coarse module config
_CN.LOFMR.COARSE = CN()
_CN.LOFMR.COARSE.D_MODEL = 192  #must be divided by 6, must be equal to the output of the _CN.LOFMR.RESNETFPN.BLOCK_DIMS
_CN.LOFMR.COARSE.NHEAD = 6  #must be divided by 6
_CN.LOFMR.COARSE.LAYER_NAMES = ['self','cross'] * 1
_CN.LOFMR.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.LOFMR.COARSE.TEMP_BUG_FIX = True
# 3. Coarse-Matching config
_CN.LOFMR.MATCH_COARSE = CN()
_CN.LOFMR.MATCH_COARSE.THR = 0.2 #dual_softmax: 0.5*0.5 = 0.25 0.2 for dual_softmax
_CN.LOFMR.MATCH_COARSE.BORDER_RM = 2
_CN.LOFMR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['sigmoid','softmax',dual_softmax, 'sinkhorn']
_CN.LOFMR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFMR.MATCH_COARSE.SKH_ITERS = 3
_CN.LOFMR.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.LOFMR.MATCH_COARSE.SKH_PREFILTER = False
_CN.LOFMR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LOFMR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LOFMR.MATCH_COARSE.SPARSE_SPVS = False
# 4. LoFTR-fine module config
_CN.LOFMR.FINE = CN()
_CN.LOFMR.FINE.D_MODEL = 128 #must be divided by 6 if using attention block
_CN.LOFMR.FINE.D_FFN = 128
_CN.LOFMR.FINE.NHEAD = 8
_CN.LOFMR.FINE.LAYER_NAMES = ['cross'] * 1 #['self', 'cross']
_CN.LOFMR.FINE.ATTENTION = 'linear'
# 5. LOFMR Losses
# -- # coarse-level
_CN.LOSS = CN()
_CN.LOSS.BETA = 0.5  #e(-1.0*0.1*1) = 0.909
##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 6e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.
_CN.TRAINER.WARMUP_STEP = 4800

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12]  # MSLR: MultiStepLR
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver

# data sampler for train_dataloader
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults_ce_point():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
