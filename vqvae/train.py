import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union, Tuple

import torch
import pytorch_lightning as pl
import numpy as np


import sys
new_path = '/home/eezzchen/TransPro'
sys.path.append(new_path)
from vqvae.model import VQVAE
from options.train_options import TrainOptions
from utils import CTDataModule



def parse_arguments():
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)

    parser.add_argument('--rescale-input', type=int, nargs='+')
    parser.add_argument("--batch-size", type=int)
    #parser.add_argument("dataset_path", type=Path)

    parser.set_defaults(
        gpus="1",
        accelerator='ddp',

        benchmark=True,

        num_sanity_val_steps=0,
        precision=16,

        log_every_n_steps=50,
        val_check_interval=0.5,
        flush_logs_every_n_steps=100,
        weights_summary='full',

        max_epochs=int(1e5),
    )

    args = parser.parse_args()

    return args


def main(args):
    torch.cuda.empty_cache()

    pl.trainer.seed_everything(seed=42)

    #datamodule = CTDataModule(path=args.dataset_path, batch_size=args.batch_size, num_workers=5, rescale_input=args.rescale_input)

    datamodule = create_dataset(opt, phase="train")  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(datamodule)
    print('#training images = %d' % dataset_size)

    while(1):True

    model = VQVAE(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, save_last=True, monitor='val_recon_loss_mean')

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    args = parse_arguments()

    main(args)