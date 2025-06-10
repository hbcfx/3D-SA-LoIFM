import pytorch_lightning as pl
import argparse
import pprint
from loguru import logger as loguru_logger

from config.defaultCEPoint import get_cfg_defaults_ce_point
from utils.profiler import build_profiler
from dataset.usct import USCTDataModule
from model.lightning_lofmrCNNPoint import PL_LoFMRCNNPoint

from utils.util import setup_gpus
def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input_file_list', type=str, help='data input path file')
    parser.add_argument(
        '--ckpt_path', type=str, default="weights/indoor_ds.ckpt", help='path to the checkpoint')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")   
    parser.add_argument(
        '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
   
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    pprint.pprint(vars(args))

    # init default-cfg and merge it with the main- and data-cfg

    config = get_cfg_defaults_ce_point()
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    loguru_logger.info(f"Args and config initialized!")

    # lightning module
    profiler = build_profiler(args.profiler_name)

    model = PL_LoFMRCNNPoint(config,pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)

    loguru_logger.info(f"LoFMR-lightning initialized!")
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    # lightning data

    data_module = USCTDataModule(args, config)

    loguru_logger.info(f"DataModule initialized!")

    # lightning trainer
    trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)
    
    loguru_logger.info(f"Start testing!")
    trainer.test(model, datamodule=data_module, verbose=False)
