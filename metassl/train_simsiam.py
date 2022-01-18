#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# code taken from https://github.com/facebookresearch/simsiam

import argparse
import builtins
import math
import os
import pathlib
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import yaml
from torch.utils.tensorboard import SummaryWriter

try:
    # For execution in PyCharm
    from metassl.utils.data import get_train_valid_loader
    from metassl.utils.config import AttrDict
    from metassl.utils.meters import AverageMeter, ProgressMeter
    from metassl.utils.simsiam import SimSiam
    from metassl.utils.summary import write_to_summary_writer
    import metassl.models.resnet_cifar as our_cifar_resnets
    from metassl.utils.torch_utils import accuracy
    from knn_validation import knn_classifier

except ImportError:
    # For execution in command line
    from .utils.data import get_train_valid_loader
    from .utils.config import AttrDict
    from .utils.meters import AverageMeter, ProgressMeter
    from .utils.simsiam import SimSiam
    from .utils.summary import write_to_summary_writer
    from .models import resnet_cifar as our_cifar_resnets
    from .utils.torch_utils import accuracy
    from .knn_validation import knn_classifier

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)


def main(config, expt_dir):
    # args = parser.parse_args()
    
    # Define master port (for preventing 'Address already in use error' when submitting more than 1 jobs on 1 node)
    # Code from: https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    master_port = find_free_port()
    config.expt.dist_url = "tcp://localhost:" + str(master_port)
    # if this should still fail: do it via filesystem initialization
    # https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization
    
    if config.expt.seed is not None:
        random.seed(config.expt.seed)
        torch.manual_seed(config.expt.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    if config.expt.gpu is not None:
        warnings.warn(
            'You have chosen a specific GPU. This will completely '
            'disable data parallelism.'
        )

    if config.expt.dist_url == "env://" and config.expt.world_size == -1:
        config.expt.world_size = int(os.environ["WORLD_SIZE"])

    config.expt.distributed = config.expt.world_size > 1 or config.expt.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if config.expt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.expt.world_size = ngpus_per_node * config.expt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, expt_dir))
    else:
        # Simply call main_worker function
        main_worker(config.expt.gpu, ngpus_per_node, config, expt_dir)


def main_worker(gpu, ngpus_per_node, config, expt_dir):
    config.expt.gpu = gpu

    # suppress printing if not master
    if config.expt.multiprocessing_distributed and config.expt.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if config.expt.gpu is not None:
        print(f"Use GPU: {config.expt.gpu} for training")

    if config.expt.distributed:
        if config.expt.dist_url == "env://" and config.expt.rank == -1:
            config.expt.rank = int(os.environ["RANK"])
        if config.expt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            config.expt.rank = config.expt.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config.expt.dist_backend, init_method=config.expt.dist_url,
            world_size=config.expt.world_size, rank=config.expt.rank
        )
        torch.distributed.barrier()
    # create model
    print(f"=> creating model '{config.model.model_type}'")
    if config.data.dataset == 'CIFAR10':
        if config.use_our_resnet:
            # Use model from our model folder instead from torchvision!
            print("Our resnet18")
            model = SimSiam(our_cifar_resnets.resnet18, config.simsiam.dim, config.simsiam.pred_dim)
        else:
            print("torchvision resnet18")
            model = SimSiam(models.__dict__[config.model.model_type], config.simsiam.dim, config.simsiam.pred_dim)
    else:
        model = SimSiam(models.__dict__[config.model.model_type], config.simsiam.dim, config.simsiam.pred_dim)

    # infer learning rate before changing batch size
    init_lr = config.train.lr * config.train.batch_size / 256

    if config.expt.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.expt.gpu is not None:
            torch.cuda.set_device(config.expt.gpu)
            model.cuda(config.expt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            config.train.batch_size = int(config.train.batch_size / ngpus_per_node)
            config.workers = int((config.expt.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.expt.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.expt.gpu is not None:
        torch.cuda.set_device(config.expt.gpu)
        model = model.cuda(config.expt.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # print(model) # print model after SyncBatchNorm

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(config.expt.gpu)

    if config.simsiam.fix_pred_lr:
        optim_params = [{
            'params': model.module.encoder.parameters(),
            'fix_lr': False
        },
            {
                'params': model.module.predictor.parameters(),
                'fix_lr': True
            }]
    else:
        optim_params = model.parameters()

    # Data loading code
    traindir = os.path.join(config.data.dataset, 'train')

    train_loader, test_loader, train_sampler, _ = get_train_valid_loader(
        data_dir=traindir,
        batch_size=config.train.batch_size,
        random_seed=config.expt.seed,
        valid_size=0.1,
        dataset_name=config.data.dataset,
        shuffle=True,
        num_workers=config.expt.workers,
        pin_memory=True,
        download=False,
        distributed=config.expt.distributed,
        drop_last=False,
        get_fine_tuning_loaders=False,
    )

    optimizer = torch.optim.SGD(
        optim_params, init_lr,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay
    )

    # optionally resume from a checkpoint
    if config.expt.ssl_model_checkpoint_path:
        if os.path.isfile(config.expt.ssl_model_checkpoint_path):
            print(f"=> loading checkpoint '{config.expt.ssl_model_checkpoint_path}'")
            if config.expt.gpu is None:
                checkpoint = torch.load(config.expt.ssl_model_checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'cuda:{config.expt.gpu}'
                checkpoint = torch.load(config.expt.ssl_model_checkpoint_path, map_location=loc)
            config.train.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{config.expt.ssl_model_checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")

    cudnn.benchmark = True
    writer = None

    if config.expt.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(expt_dir, f"tensorboard_pretraining_{config.train.epochs}_{init_lr}"))

    print(f"=> BEGIN PRE-TRAINING with config {config}")
    best_acc = 0.0

    for epoch in range(config.train.start_epoch, config.train.epochs):
        warmup = config.expt.warmup_epochs > epoch

        if config.expt.distributed:
            train_sampler.set_epoch(epoch)
        if warmup:
            from metassl.utils.torch_utils import adjust_learning_rate as adjust_learning_rate_warmup
            cur_lr = adjust_learning_rate_warmup(optimizer, init_lr, epoch, total_epochs=config.expt.warmup_epochs, warmup=True, multiplier=config.expt.warmup_multiplier)
            print(f"warming up phase (PT)")
        else:
            cur_lr = adjust_learning_rate(optimizer, init_lr, epoch, config.train.epochs, config)
        print(f"Current LR: {cur_lr}")

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, config, writer)

        # evaluate on validation set if run_knn_val flag set

        if (epoch % config.train.val_freq == 0) and config.run_knn_val :
            if config.expt.rank == 0:
                top1_avg = knn_classifier(net=model.module.encoder, batch_size=config.train.batch_size,
                                          workers=config.expt.workers, epoch=epoch, datatset=config.data.dataset)
                writer.add_scalar('Pre-training/Accuracy@1', top1_avg, epoch)
                print(f"=> Validation '{top1_avg}'")

                # save the best model
                if top1_avg > best_acc:
                    best_acc = top1_avg
                    if not config.expt.multiprocessing_distributed or (config.expt.multiprocessing_distributed
                                                                       and config.expt.rank % ngpus_per_node == 0):
                        save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': config.model.model_type,
                                'state_dict': model.state_dict(),
                                'top1_best':top1_avg,
                                'optimizer': optimizer.state_dict(),
                            }, is_best=True, filename=os.path.join(expt_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                        )

    # shut down writer at end of training
    if config.expt.rank == 0:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, config, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix=f"Epoch: [{epoch}]"
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.expt.gpu is not None:
            images[0] = images[0].cuda(config.expt.gpu, non_blocking=True)
            images[1] = images[1].cuda(config.expt.gpu, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.expt.print_freq == 0:
            progress.display(i)
        # write log epoch wise
        if config.expt.rank == 0:
            writer.add_scalar('Pre-training/Loss', loss.item(), epoch + 1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, init_lr, epoch, total_epochs, config):
    """Decay the learning rate based on schedule"""
    if config.train.schedule == "cosine":
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / total_epochs))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
                return init_lr
            else:
                param_group['lr'] = cur_lr
                return cur_lr

def find_free_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == '__main__':
    user = os.environ.get('USER')

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--expt_name', default='pre-training-fix-lr-100-256', type=str, help='experiment name')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial (base) learning rate', dest='lr')
    parser.add_argument('--ssl_model_checkpoint_path', default=None, type=str, help='pretrained model checkpoint path')
    parser.add_argument('--expt_mode', default="ImageNet", choices=["ImageNet", "CIFAR10"],
                        help='Define which dataset to use to select the correct yaml file.')
    parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--val_freq', default=10, type=int, metavar='N',
                        help='Frequency to evaluate kNN classifier accuracy # epochs')
    parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed of numpy and torch')
    parser.add_argument('--scheduler_epochs', default=100, type=int, metavar='N', help='denotes when scheduler should '
                                                                                       'step')
    parser.add_argument('--run_knn_val', action='store_true')  # if needed run knn validation
    parser.add_argument('--use_our_resnet', action='store_true', help='Set this flag to use our resnet18. Default: Use torchvision resnet18.')
    parser.add_argument('--warmup_epochs', default=3, type=int, metavar='N', help='denotes the number of epochs that we only pre-train without finetuning afterwards; warmup is turned off when set to 0; we use a linear incremental schedule during warmup')
    parser.add_argument('--warmup_multiplier', default=2., type=float, metavar='N', help='A factor that is multiplied with the pretraining lr used in the linear incremental learning rate scheduler during warmup. The final lr is multiplier * pre-training lr')

    args = parser.parse_args()

    expt_name = args.expt_name
    epochs = args.epochs
    lr = args.lr
    ssl_model_checkpoint_path = args.ssl_model_checkpoint_path

    # Saving checkpoint and config pased on experiment mode
    if args.expt_mode == "ImageNet":
        expt_dir = f"/home/{user}/workspace/experiments/metassl"
    elif args.expt_mode == "CIFAR10":
        if user == "wagnerd":
            expt_dir = "/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10"
        else:
            expt_dir = "experiments"
    else:
        raise ValueError(f"Experiment mode {args.expt_mode} is undefined!")
    expt_sub_dir = os.path.join(expt_dir, expt_name)

    expt_dir = pathlib.Path(expt_dir)

    if not os.path.exists(expt_sub_dir):
        os.makedirs(expt_sub_dir)

    # Select which yaml file to use depending on the selected experiment mode
    if args.expt_mode == "ImageNet":
        config_path = "metassl/default_metassl_config.yaml"
    elif args.expt_mode == "CIFAR10":
        config_path = "metassl/default_metassl_config_cifar10.yaml"
    else:
        raise ValueError(f"Experiment mode {args.expt_mode} is undefined!")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.expt_mode == "ImageNet":
        config['data']['data_dir'] = f'/home/{user}/workspace/data/metassl'
        config['expt']['expt_name'] = expt_name
        config['expt']['ssl_model_checkpoint_path'] = ssl_model_checkpoint_path
        config['train']['epochs'] = epochs
        config['train']['lr'] = lr


    print(expt_name, ssl_model_checkpoint_path, epochs, lr)
    print(f"batch size {config['train']['batch_size']}")

    with open(os.path.join(expt_sub_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
        print(f"copied config to {expt_sub_dir}")

    # overwrite config yaml values from command line
    config['train']['epochs'] = epochs
    config['train']['lr'] = lr
    config['expt']['workers'] = args.workers
    config['train']['val_freq'] = args.val_freq
    config['expt']['seed'] = args.seed
    config['train']['scheduler_epochs'] = args.scheduler_epochs
    config['run_knn_val'] = args.run_knn_val
    config['use_our_resnet'] = args.use_our_resnet
    config['expt']['warmup_epochs'] = args.warmup_epochs
    config['expt']['warmup_multiplier'] = args.warmup_multiplier

    config = AttrDict(config)

    main(config=config, expt_dir=expt_sub_dir)
