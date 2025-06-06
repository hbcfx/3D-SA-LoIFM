import json
import torch.nn.functional as F
import torch
import os
import pickle
from joblib import Parallel,delayed
from yacs.config import CfgNode as CN
from itertools import chain
from loguru import _Logger, logger
from pytorch_lightning.utilities import rank_zero_only
from typing import Union
from einops.einops import rearrange

def flattenList(x):
    return list(chain(*x))

def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level,
                    lambda x: None)
        logger._log = lambda x: None
    return logger

def setup_gpus(gpus: Union[str, int]) -> int:
    """ must set up CUDA_VISIBLE_DEVICES before python train.py, then the visible devices will be numbered from 0 """
    gpus = str(gpus)
    gpu_ids = []

    if ',' not in gpus:
        n_gpus = int(gpus)
        return n_gpus if n_gpus != -1 else torch.cuda.device_count()
    else:
        gpu_ids = [i.strip() for i in gpus.split(',') if i != '']
    return len(gpu_ids)

def load_json(file):
    with open(file,'r') as f:
        a=json.load(f)
    return a
def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)
write_json = save_json

##transform the config file keys to lowwer case (xiaoxie)
def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}

#transform the physical point to its grid coordinates
def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}

def pardir(path):
    return os.path.join(path, os.pardir)

def maybe_mkdir_p(directory):
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i+1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i+1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file, mode='wb'):
    with open(file, mode) as f:
        pickle.dump(obj, f)

save_pickle = write_pickle
# I'm tired of typing these out
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir

def originalAfterCrop(origin, spacing, crop):
    origin_crop = [origin[0]+crop[0]*spacing[0],origin[1]+crop[1]*spacing[1],origin[2]+crop[2]*spacing[2]]
    return origin_crop