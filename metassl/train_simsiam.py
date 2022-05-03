#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# code taken from https://github.com/facebookresearch/simsiam

import builtins
import logging
import math
import os
import random
import time
import warnings
from collections import OrderedDict
from copy import deepcopy

import numpy as np
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
from torch.utils.tensorboard import SummaryWriter

from metassl.hyperparameter_optimization.configspaces import get_neps_pipeline_space
from metassl.parser_flags import get_parsed_config
from metassl.utils.lars_optimizer import LARS

warnings.filterwarnings("ignore", category=UserWarning)

try:
    # For execution in PyCharm
    import metassl.models.resnet_cifar as our_cifar_resnets
    from metassl.utils.data import get_loaders
    from metassl.utils.io import (
        find_free_port,
        get_expt_dir_with_bohb_config_id,
        organize_experiment_saving,
    )
    from metassl.utils.knn_validation import knn_classifier
    from metassl.utils.meters import ProgressMeter, initialize_all_meters, update_grad_stats_meters
    from metassl.utils.simsiam_alternating import SimSiam
    from metassl.utils.summary import write_to_summary_writer
    from metassl.utils.torch_utils import (
        adjust_learning_rate,
        check_and_save_checkpoint,
        deactivate_bn,
        get_newest_model,
    )

except ImportError:
    # For execution in command line
    from .models import resnet_cifar as our_cifar_resnets
    from .utils.data import get_loaders
    from .utils.io import (
        find_free_port,
        get_expt_dir_with_bohb_config_id,
        organize_experiment_saving,
    )
    from .utils.knn_validation import knn_classifier
    from .utils.meters import ProgressMeter, initialize_all_meters, update_grad_stats_meters
    from .utils.simsiam_alternating import SimSiam
    from .utils.summary import write_to_summary_writer
    from .utils.torch_utils import (
        adjust_learning_rate,
        check_and_save_checkpoint,
        deactivate_bn,
        get_newest_model,
    )


def main(working_directory, config, bohb_infos=None, **hyperparameters):
    config = deepcopy(
        config
    )  # Important for NEPS (for not overwriting pretraining stuff in finetuning and vice versa)
    print("\n\n\nPRETRAINING\n\n\n")
    # NEPS implementation requires "working_directory" as the first argument
    expt_dir = working_directory

    # BOHB only ------------------------------------------------------------------------------------
    if bohb_infos is not None:
        # Integrate budget based on budget_mode
        if config.bohb.budget_mode == "epochs":
            # TODO: @Diane - Check out how to handle the case where #epochs_pt != #epochs_ft
            config.train.epochs = int(bohb_infos["bohb_budget"])
            config.finetuning.epochs = int(bohb_infos["bohb_budget"])
        else:
            raise ValueError(f"Budget mode '{config.bohb.budget_mode}' not implemented yet!")

        # Add --bohb.configspace_mode to bohb_infos
        bohb_infos["bohb_configspace"] = config.bohb.configspace_mode

        # Create subfoler for each config_id
        # (directory where tensorboard and checkpoints are being saved)
        expt_dir_id = get_expt_dir_with_bohb_config_id(expt_dir, bohb_infos["bohb_config_id"])
        expt_dir = expt_dir_id

        # Define master port (for preventing 'Address already in use error' when more than 1
        # submitted worker on 1 node)
        # TODO: @Diane - Think for another strategy to handle the problem
        # str_config_id = "".join(str(sub_id) for sub_id in bohb_infos['bohb_config_id'])
        # master_port = str(int(bohb_infos['bohb_budget'])) + str_config_id
        # print(f"{master_port=}")
        # if len(master_port) < 5:
        #     master_port = master_port + str(0)
        # print(f"{master_port=}")
        # config.expt.dist_url = "tcp://localhost:" + master_port
        # print(f"{config.expt.dist_url=}")

        print(f"\n\n\n\n\n\nbohb_infos: {bohb_infos}\n\n\n\n\n\n")
    # ----------------------------------------------------------------------------------------------
    neps_hyperparameters = None
    # NEPS only ------------------------------------------------------------------------------------
    if config.neps.is_neps_run:
        neps_hyperparameters = hyperparameters

        if "pt_learning_rate" in neps_hyperparameters:
            config.train.lr = neps_hyperparameters["pt_learning_rate"]
            print("PT LR: ", config.train.lr)

        if "warmup_epochs" in neps_hyperparameters:
            config.expt.warmup_epochs = neps_hyperparameters["warmup_epochs"]
            print("Warmup Epochs: ", config.expt.warmup_epochs)
    # ----------------------------------------------------------------------------------------------

    if config.expt.seed is not None:
        random.seed(config.expt.seed)
        torch.manual_seed(config.expt.seed)
        np.random.seed(config.expt.seed)
        cudnn.deterministic = True  # TODO: @Diane - checkout
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if config.expt.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely " "disable data parallelism."
        )

    if config.expt.dist_url == "env://" and config.expt.world_size == -1:
        config.expt.world_size = int(os.environ["WORLD_SIZE"])

    config.expt.distributed = config.expt.world_size > 1 or config.expt.multiprocessing_distributed

    if (
        config.data.dataset == "CIFAR10" or config.data.dataset == "CIFAR100"
    ) and config.expt.distributed:
        # Define master port (for preventing 'Address already in use error' when submitting more
        # than 1 jobs on 1 node)
        # Code from:
        # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        master_port = find_free_port()
        config.expt.dist_url = "tcp://localhost:" + str(master_port)
        # if this should still fail: do it via filesystem initialization
        # https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization

    ngpus_per_node = torch.cuda.device_count()
    if config.expt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config.expt.world_size = ngpus_per_node * config.expt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, config, expt_dir, bohb_infos, neps_hyperparameters),
        )
    else:
        # Simply call main_worker function
        try:
            main_worker(
                config.expt.gpu,
                ngpus_per_node,
                config,
                expt_dir,
                bohb_infos,
                neps_hyperparameters,
            )
        except KeyError as e:
            print("\n\n RETURN 0 \n\n", e)
            return 0

    # BOHB only ------------------------------------------------------------------------------------
    # Read validation metric from the .txt (as for mp.spawn returning values is not trivial)
    if bohb_infos is not None:
        with open(expt_dir + "/current_val_metric.txt", "r") as f:
            val_metric = f.read()
        print(f"val_metric: {val_metric}")
        return float(val_metric)
    # ----------------------------------------------------------------------------------------------
    # NEPS only ------------------------------------------------------------------------------------
    # Read validation metric from the .txt (as for mp.spawn returning values is not trivial)
    if config.neps.is_neps_run:
        from metassl.train_linear_classifier_simsiam import main as main_ft

        val_metric = main_ft(
            config=config,
            expt_dir=expt_dir,
            bohb_infos=None,
            hyperparameters=neps_hyperparameters,
        )
        return float(val_metric)
    # ----------------------------------------------------------------------------------------------


def main_worker(gpu, ngpus_per_node, config, expt_dir, bohb_infos, neps_hyperparameters):
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
            backend=config.expt.dist_backend,
            init_method=config.expt.dist_url,
            world_size=config.expt.world_size,
            rank=config.expt.rank,
        )
        torch.distributed.barrier()

    # create model
    # TODO: @Diane - Check out and compare against baseline code
    if config.data.dataset == "CIFAR10":
        # Use model from our model folder instead from torchvision!
        print("=> creating model resnet18")
        model = SimSiam(
            our_cifar_resnets.resnet18,
            config.simsiam.dim,
            config.simsiam.pred_dim,
            num_classes=10,
            neps_hyperparameters=neps_hyperparameters,
        )

    elif config.data.dataset == "CIFAR100":
        # Use model from our model folder instead from torchvision!
        print("=> creating model resnet18")
        model = SimSiam(
            our_cifar_resnets.resnet18,
            config.simsiam.dim,
            config.simsiam.pred_dim,
            num_classes=100,
        )

    elif config.data.dataset == "ImageNet":
        print(f"=> creating model {config.model.model_type}")
        model = SimSiam(
            models.__dict__[config.model.model_type],
            config.simsiam.dim,
            config.simsiam.pred_dim,
            num_classes=1000,
        )

    else:
        raise NotImplementedError

    # print(model)
    print("\n\n\nNEPS HYPERPARAMETERS: ", neps_hyperparameters, "\n\n\n")

    if config.model.turn_off_bn:
        print("Turning off BatchNorm in entire model.")
        deactivate_bn(model)
        model.encoder_head[6].bias.requires_grad = True

    # infer learning rate !before changing batch size! > see lines below
    divisor = 256 if config.data.dataset == "ImageNet" else config.train.batch_size
    init_lr_pt = config.train.lr * config.train.batch_size / divisor

    if config.expt.distributed:
        # Apply SyncBN TODO: @Fabio - keep for CIFAR10? (metassl code)
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

            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.expt.gpu],
                find_unused_parameters=True,  # TODO: @Fabio - keep for CIFAR10? (metassl code)
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.expt.gpu is not None:
        torch.cuda.set_device(config.expt.gpu)
        model = model.cuda(config.expt.gpu)
        # comment out the following line for debugging  # TODO: delete? (metassl code)
        # TODO: delete line below (metassl code)
        # raise NotImplementedError("Only DistributedDataParallel or gpu mode is supported.")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        # TODO: Integrate lines below? (baselines code)
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        #     model = torch.nn.DataParallel(model).cuda()

    # TODO: delete? (metassl code)
    # else:
    #     # AllGather implementation (batch shuffle, queue update, etc.) in
    #     # this code only supports DistributedDataParallel.
    #     raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer

    criterion_pt = nn.CosineSimilarity(dim=1).cuda(
        config.expt.gpu
    )  # TODO: @Diane - Check out and compare against baseline code
    # TODO: delete as it is unused?
    # criterion_ft = nn.CrossEntropyLoss().cuda(config.expt.gpu)

    if (
        config.expt.distributed
    ):  # TODO: @Fabio - This is not working for not config.expt.distributed (metassl code)
        optim_params_pt = [
            {
                "params": model.module.backbone.parameters()
                if config.expt.distributed
                else model.backbone.parameters(),
                "fix_lr": False,
            },
            {
                "params": model.module.encoder_head.parameters()
                if config.expt.distributed
                else model.encoder_head.parameters(),
                "fix_lr": False,
            },
            {
                "params": model.module.predictor.parameters()
                if config.expt.distributed
                else model.predictor.parameters(),
                "fix_lr": config.simsiam.fix_pred_lr,
            },
        ]
    if neps_hyperparameters is not None and "pt_optimizer" in neps_hyperparameters:
        # choices=['adamw', 'sgd', 'lars']
        if neps_hyperparameters["pt_optimizer"] == "sgd":
            optimizer_pt = torch.optim.SGD(
                params=optim_params_pt if config.expt.distributed else model.parameters(),
                lr=init_lr_pt,
                momentum=config.train.momentum,
                weight_decay=config.train.weight_decay,
            )
        elif neps_hyperparameters["pt_optimizer"] == "adamw":
            optimizer_pt = torch.optim.AdamW(
                params=optim_params_pt if config.expt.distributed else model.parameters(),
                lr=init_lr_pt,
                weight_decay=config.train.weight_decay,
            )
        elif neps_hyperparameters["pt_optimizer"] == "lars":
            optimizer_pt = LARS(
                params=model.parameters(),
                lr=init_lr_pt,
                weight_decay=config.train.weight_decay,
                momentum=config.train.momentum,
            )
        else:
            raise NotImplementedError(
                "{neps_hyperparameters['pt_optimizer']} is not implemented yet!"
            )
        print("\nOptimizer (NEPS): ", optimizer_pt)

    else:
        optimizer_pt = torch.optim.SGD(
            params=optim_params_pt if config.expt.distributed else model.parameters(),
            lr=init_lr_pt,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )

    # in case a dumped model exist and ssl_model_checkpoint is not set, load that dumped model
    newest_model = get_newest_model(expt_dir, suffix="checkpoint*.pth.tar")
    if newest_model and config.expt.ssl_model_checkpoint_path is None:
        config.expt.ssl_model_checkpoint_path = newest_model

    total_iter = 0
    meters = None

    # optionally resume from a checkpoint
    if config.expt.ssl_model_checkpoint_path:
        if os.path.isfile(config.expt.ssl_model_checkpoint_path):
            print(f"=> loading checkpoint '{config.expt.ssl_model_checkpoint_path}'")
            if config.expt.gpu is None:
                checkpoint = torch.load(config.expt.ssl_model_checkpoint_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f"cuda:{config.expt.gpu}"
                checkpoint = torch.load(config.expt.ssl_model_checkpoint_path, map_location=loc)
            config.train.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer_pt.load_state_dict(checkpoint["optimizer_pt"])
            total_iter = checkpoint["total_iter"]
            meters = checkpoint["meters"]
            print(
                f"=> loaded checkpoint '{config.expt.ssl_model_checkpoint_path}' "
                f"(epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{config.expt.ssl_model_checkpoint_path}'")

    if config.finetuning.valid_size > 0:
        (
            train_loader_pt,
            train_sampler_pt,
            train_loader_ft,
            train_sampler_ft,
            valid_loader_ft,
            test_loader_ft,
        ) = get_loaders(
            config,
            parameterize_augmentation=False,
            bohb_infos=bohb_infos,
            neps_hyperparameters=neps_hyperparameters,
        )
    else:  # TODO: @Diane - Checkout and test on *parameterized_aug*
        (
            train_loader_pt,
            train_sampler_pt,
            train_loader_ft,
            train_sampler_ft,
            test_loader_ft,
        ) = get_loaders(
            config,
            parameterize_augmentation=False,
            bohb_infos=bohb_infos,
            neps_hyperparameters=neps_hyperparameters,
        )

    cudnn.benchmark = True
    writer = None

    if config.expt.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(expt_dir, "tensorboard_pt"))
    if not meters:
        meters = initialize_all_meters()

    for epoch in range(config.train.start_epoch, config.train.epochs):

        if config.expt.distributed:
            train_sampler_pt.set_epoch(epoch)
            train_sampler_ft.set_epoch(epoch)

        warmup = config.expt.warmup_epochs > epoch

        if warmup:
            print(f"warming up: epoch {epoch} / {config.expt.warmup_epochs}")
            cur_lr_pt = adjust_learning_rate(
                optimizer=optimizer_pt,
                init_lr=init_lr_pt,
                epoch=epoch,
                total_epochs=config.expt.warmup_epochs,
                warmup=True,
                multiplier=config.expt.warmup_multiplier,
                neps_hyperparameters=neps_hyperparameters,
            )
            if not config.expt.warmup_epochs > epoch + 1:
                init_lr_pt = cur_lr_pt

        else:
            cur_lr_pt = adjust_learning_rate(
                optimizer=optimizer_pt,
                init_lr=init_lr_pt,
                epoch=epoch,
                total_epochs=config.train.epochs,
                warmup=False,
            )

        print(f"Current LR: {cur_lr_pt}")

        if config.expt.wd_decay_pt or (
            neps_hyperparameters is not None and "pt_weight_decay_start" in neps_hyperparameters
        ):
            if neps_hyperparameters is not None and "pt_weight_decay_start" in neps_hyperparameters:
                config.train.wd_start = neps_hyperparameters["pt_weight_decay_start"]
                config.train.wd_end = neps_hyperparameters["pt_weight_decay_end"]
                print("\nAnnealing Weight Decay (NEPS)")

            # Do annealing
            if epoch == 0:
                for group in optimizer_pt.param_groups:
                    group["weight_decay"] = config.train.wd_start
                    current_weight_decay = group["weight_decay"]
            else:
                for group in optimizer_pt.param_groups:
                    group["weight_decay"] = config.train.wd_end + 1 / 2 * (
                        config.train.wd_start - config.train.wd_end
                    ) * (1 + math.cos(epoch / (config.train.epochs - 1) * math.pi))
                    current_weight_decay = group["weight_decay"]
            print(f"current weight decay (pt): {current_weight_decay}\n")

        total_iter = train_one_epoch(
            train_loader_pt=train_loader_pt,
            model=model,
            criterion_pt=criterion_pt,
            optimizer_pt=optimizer_pt,
            epoch=epoch,
            total_iter=total_iter,
            config=config,
            writer=writer,
            meters=meters,
            warmup=warmup,
        )

        if (
            (epoch % config.expt.eval_freq == 0)
            and config.expt.run_knn_val
            and config.data.dataset != "ImageNet"
        ):
            if config.expt.rank == 0:
                top1_avg = knn_classifier(
                    net=model.module.backbone,
                    batch_size=config.train.batch_size,
                    workers=config.expt.workers,
                    dataset=config.data.dataset,
                    hide_progress=True,
                    download=False,
                )
                writer.add_scalar("Pre-training/kNN test acc@1", top1_avg, epoch)
                print(f"=> kNN test acc@1 '{top1_avg}'")

        # make sure to always save at the end of training
        is_last_epoch = epoch + 1 >= config.train.epochs
        if bohb_infos is not None:
            if (
                config.bohb.budget_mode == "epochs"
                and epoch % int(bohb_infos["bohb_budget"] - 1) == 0
            ):
                check_and_save_checkpoint(
                    config=config,
                    ngpus_per_node=ngpus_per_node,
                    total_iter=total_iter,
                    epoch=epoch,
                    model=model,
                    optimizer_pt=optimizer_pt,
                    optimizer_ft=None,
                    expt_dir=expt_dir,
                    meters=meters,
                )
            elif config.bohb.budget_mode != "epochs":
                raise ValueError("Not implemented yet!")

        elif (config.expt.save_model and epoch % config.expt.save_model_frequency == 0) or (
            config.expt.save_model and is_last_epoch
        ):

            check_and_save_checkpoint(
                config=config,
                ngpus_per_node=ngpus_per_node,
                total_iter=total_iter,
                epoch=epoch,
                model=model,
                optimizer_pt=optimizer_pt,
                optimizer_ft=None,
                expt_dir=expt_dir,
                meters=meters,
            )

    # shut down writer at end of training
    if config.expt.rank == 0:
        writer.close()

        batch_time_meter = meters["batch_time_meter"]
        print(f"total batch time elapsed: {batch_time_meter.sum:.2f}s")


def train_one_epoch(
    train_loader_pt,
    model,
    criterion_pt,
    optimizer_pt,
    epoch,
    total_iter,
    config,
    writer,
    meters,
    warmup=False,
):
    # general meters
    batch_time_meter = meters["batch_time_meter"]
    data_time_meter = meters["data_time_meter"]
    losses_pt_meter = meters["losses_pt_meter"]
    top1_meter = meters["top1_meter"]
    top5_meter = meters["top5_meter"]

    norm_pt_meter_global = meters["norm_pt_meter_global"]
    norm_pt_avg_meter_lw = meters["norm_pt_avg_meter_lw"]
    norm_pt_std_meter_lw = meters["norm_pt_std_meter_lw"]
    target_std_meter = meters["target_std_meter"]

    meters_to_print = [
        batch_time_meter,
        losses_pt_meter,
        top1_meter,
        norm_pt_meter_global,
        norm_pt_avg_meter_lw,
        target_std_meter,
    ]

    progress = ProgressMeter(
        num_batches=len(train_loader_pt),
        meters=meters_to_print,
        prefix=f"Epoch: [{epoch}]",
    )

    end = time.time()

    for i, (images_pt, _) in enumerate(train_loader_pt):

        total_iter += 1

        if config.expt.gpu is not None:
            images_pt[0] = images_pt[0].cuda(config.expt.gpu, non_blocking=True)
            images_pt[1] = images_pt[1].cuda(config.expt.gpu, non_blocking=True)

        get_gradients = False if config.expt.is_non_grad_based else True  # default is True
        loss_pt, backbone_grads_pt_lw, backbone_grads_pt_global, z1, z2 = pretrain(
            model,
            images_pt,
            criterion_pt,
            optimizer_pt,
            losses_pt_meter,
            data_time_meter,
            end,
            config=config,
            get_gradients=get_gradients,
        )

        z_std_normalized = np.std(
            z1.cpu().numpy() / (torch.linalg.norm(z1, 2).cpu().numpy()) + 1e-9
        )
        target_std_meter.update(z_std_normalized)

        grads = {
            "backbone_grads_ft_lw": None,
            "backbone_grads_ft_global": None,
            "backbone_grads_pt_lw": backbone_grads_pt_lw,
            "backbone_grads_pt_global": backbone_grads_pt_global,
        }

        update_grad_stats_meters(grads=grads, meters=meters, warmup=warmup)

        main_stats_meters = [
            norm_pt_meter_global,
            norm_pt_avg_meter_lw,
            target_std_meter,
        ]

        additional_stats_meters = [
            norm_pt_std_meter_lw,
        ]

        meters_to_plot = {
            "main_meters": main_stats_meters,
            "additional_stats_meters": additional_stats_meters,
        }

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if i % config.expt.print_freq == 0:
            progress.display(i)
        if config.expt.rank == 0 and i % config.expt.write_summary_frequency:
            write_to_summary_writer(
                total_iter=total_iter,
                loss_pt=loss_pt,
                loss_ft=None,
                data_time=data_time_meter,
                batch_time=batch_time_meter,
                optimizer_pt=optimizer_pt,
                optimizer_ft=None,
                top1=top1_meter,
                top5=top5_meter,
                meters_to_plot=meters_to_plot,
                writer=writer,
            )

    return total_iter


def pretrain(
    model,
    images_pt,
    criterion_pt,
    optimizer_pt,
    losses_pt,
    data_time,
    end,
    config,
    get_gradients=False,
):
    backbone_grads_lw = OrderedDict()
    backbone_grads_global = torch.Tensor().cuda()

    model.requires_grad_(True)

    # switch to train mode
    model.train()

    # measure data loading time
    data_time.update(time.time() - end)

    # pre-training
    # compute outputs
    p1, p2, z1, z2 = model(x1=images_pt[0], x2=images_pt[1], finetuning=False)

    # compute losses
    loss_pt = -(criterion_pt(p1, z2).mean() + criterion_pt(p2, z1).mean()) * 0.5
    losses_pt.update(loss_pt.item(), images_pt[0].size(0))

    # compute gradient and do SGD step
    optimizer_pt.zero_grad()
    loss_pt.backward()
    # step does not change .grad field of the parameters.
    optimizer_pt.step()

    if get_gradients:
        backbone = model.module.backbone

        for key, param in backbone.named_parameters():
            grad_tensor = param.grad.detach_().clone().flatten()
            backbone_grads_lw[key] = torch.tensor(grad_tensor)
            backbone_grads_global = torch.cat([backbone_grads_global, grad_tensor], dim=0)

    return loss_pt, backbone_grads_lw, backbone_grads_global, z1, z2


if __name__ == "__main__":
    user = os.environ.get("USER")
    config = get_parsed_config()

    print("\n\n\n\nConfig:\n", config, "\n\n\n\n")

    # Check whether it is a NEPS run or not + organize expt_dir accordingly
    is_bohb_run = True if config.expt.expt_mode.endswith("BOHB") else False
    expt_dir = organize_experiment_saving(
        user=user, config=config, is_neps_run=config.neps.is_neps_run
    )

    # Error check
    if (
        config.finetuning.data_augmentation != "none"
        and config.expt.data_augmentation_mode != "probability_augment"
    ):
        raise ValueError(
            "If you use data augmentation for finetuning, 'probability_augment' is required as the "
            "data_augmentation_mode!"
        )
    if is_bohb_run:
        assert config.finetuning.valid_size > 0.0, "BOHB requires a valid_size > 0.0"
        if (
            config.bohb.configspace_mode == "probability_augment"
            and config.expt.data_augmentation_mode != "probability_augment"
        ):
            raise ValueError(
                "If you run a BOHB experiment with 'probability_augment' configspace mode, "
                "you also need to select 'probability_augment' as data augmentation mode!"
            )
        if (
            config.bohb.configspace_mode == "double_probability_augment"
            and config.finetuning.data_augmentation != "p_probability_augment_ft"
        ):
            raise ValueError(
                "If you run a BOHB experiment with 'double_probability_augment' configspace mode, "
                "you also need to select 'p_probability_augment_ft' as finetuning data "
                "augmentation mode!"
            )

    if config.neps.is_neps_run:  # -----------------------------------------------------------------
        from functools import partial

        import neps

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        )

        # Checks
        if config.data.dataset != "CIFAR10":
            raise NotImplementedError(
                "Please fix some CIFAR10 hardcoding in pipeline_space to start NEPS experiments "
                "with another dataset!"
            )
        if config.neps.optimize_backbone_only and not (
            config.neps.config_space == "hierarchical_nas" or config.neps.config_space == "combined"
        ):  # TODO: Flag only for "hierarchical_nas" or for both "hierarchical_nas" and "combined"?
            raise ValueError(
                "The optimize_backebone_only flag makes only sense for the following configspaces: "
                "'hierarchical_nas' and 'combined'"
            )

        # Get NEPS config space with/without user prior
        # Config spaces choices: ["data_augmentation", "hierarchical_nas", "training", "combined"]
        pipeline_space = get_neps_pipeline_space(config=config)

        # Start NEPS
        main = partial(main, config=config)
        neps.run(
            run_pipeline=main,
            pipeline_space=pipeline_space,
            working_directory=expt_dir,
            max_evaluations_total=500,
            max_evaluations_per_run=1,
            overwrite_working_directory=False,  # Set True for debuggingq
        )
        # ------------------------------------------------------------------------------------------

    else:
        # Run BOHB / main
        if is_bohb_run:
            from metassl.hyperparameter_optimization.master import start_bohb_master

            start_bohb_master(yaml_config=config, expt_dir=expt_dir)
        else:
            main(working_directory=expt_dir, config=config)
