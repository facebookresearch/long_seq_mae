# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.crop import RandomResizedCrop as BYOLRandomResizedCrop
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.long_seq_patch_loader import SampleVisiblePatchIndices, MAEIndexCollator

import models_mae

from engine_pretrain import train_one_epoch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
except ImportError:
    xm = xmp = pl = xu = None


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--effective_batch_size', default=-1, type=int,
                        help='Effective batch size (set to -1 to ignore and use --batch_size)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--ckpt_interval', default=20, type=int,
                        help='The interval (in epochs) to save a checkpoint')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--decoder_embed_dim', default=-1, type=int)
    parser.add_argument('--decoder_depth', default=-1, type=int)
    parser.add_argument('--no_k_bias_in_vit', action='store_true', dest='no_k_bias_in_vit',
                        help="Use a variant of ViT without k_bias in ViT self-attention (as in BEiT)")
    parser.set_defaults(no_k_bias_in_vit=False)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=-1, type=int,
                        help='ViT patch size (-1 means it will be automatically inferred from `model`')
    parser.add_argument('--min_crop', default=0.2, type=float,
                        help='minimum crop ratio in random resized crop')
    parser.add_argument('--max_crop', default=1.0, type=float,
                        help='maximum crop ratio in random resized crop')
    parser.add_argument('--use_byol_crop', action='store_true',
                        help='Use BYOL random resized crop')
    parser.set_defaults(use_byol_crop=False)

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_downsampling', default=1, type=int,
                        help='Downsampling ratio of masks (e.g. 2 means using 32x32 mask patches for 16x16 image patches).')
    parser.add_argument('--decoder_downsampling', default=1, type=int,
                        help='Downsampling ratio in the MAE decoder (e.g. 2 means using a 2x2 conv w/ stride 2 '
                             'to downsample the decoder input, giving a smaller decoder sequence length than encoder).')
    parser.add_argument('--pred_downsampling', default=1, type=int,
                        help='Downsampling ratio of prediction target image grid compared to the encoder grid '
                             '(e.g. 2 means predicting in 32x32 patch size when the encoder patch size is 16x16')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # PyTorch XLA parameters
    parser.add_argument('--use_xla', action='store_true',
                        help='Use PyTorch XLA on TPUs')
    parser.set_defaults(use_xla=False)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    if misc.XLA_CFG["is_xla"]:
        device = xm.xla_device()
    else:
        device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    world_size = misc.get_world_size()
    assert (args.batch_size > 0) != (args.effective_batch_size > 0) or (
        args.batch_size == args.effective_batch_size // world_size // args.accum_iter), \
        "only one of --batch_size and --effective_batch_size should be specified (set to -1 to unspecify)"
    if args.effective_batch_size > 0:
        assert args.effective_batch_size % (world_size * args.accum_iter) == 0
        args.batch_size = args.effective_batch_size // world_size // args.accum_iter

    # simple augmentation
    MAECrop = BYOLRandomResizedCrop if args.use_byol_crop else transforms.RandomResizedCrop
    transform_train = transforms.Compose([
            MAECrop(args.input_size, scale=(args.min_crop, args.max_crop), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.patch_size == -1:
        # automatically infer the patch size from model names
        if "patch14" in args.model:
            args.patch_size = 14
        elif "patch64" in args.model:
            args.patch_size = 64
        elif "patch32" in args.model:
            args.patch_size = 32
        elif "patch24" in args.model:
            args.patch_size = 24
        elif "patch16" in args.model:
            args.patch_size = 16
        elif "patch8" in args.model:
            args.patch_size = 8
        elif "patch4" in args.model:
            args.patch_size = 4
        else:
            raise Exception("cannot automatically infer patch size from args.model")
    assert args.input_size % args.patch_size == 0
    num_patches = (args.input_size // args.patch_size) ** 2
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'),
        transform=SampleVisiblePatchIndices(
            transform_train, num_patches, args.mask_ratio, args.mask_downsampling,
        ),
    )
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None and not misc.XLA_CFG["is_xla"]:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
        collate_fn=MAEIndexCollator(),
    )
    data_loader_train_sampler = data_loader_train.sampler
    if misc.XLA_CFG["is_xla"]:
        data_loader_train = pl.MpDeviceLoader(data_loader_train, device)
    
    # define the model
    model = models_mae.__dict__[args.model](
        args=args,
        img_size=args.input_size,
        patch_size=args.patch_size,
        norm_pix_loss=args.norm_pix_loss,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if misc.XLA_CFG["is_xla"]:
        misc.broadcast_xla_master_model_param(model)
    elif args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % args.ckpt_interval == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def xla_main(index, args):
    misc.XLA_CFG["is_xla"] = True
    main(args)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.use_xla:
        xmp.spawn(xla_main, args=(args,))
    else:
        main(args)
