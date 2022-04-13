# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# code taken from https://github.com/facebookresearch/simsiam

import builtins
import os
import random
import time
import warnings
from collections import OrderedDict

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

from metassl.parser_flags import get_parsed_config

warnings.filterwarnings("ignore", category=UserWarning)

try:
    # For execution in PyCharm
    import metassl.models.resnet_cifar as our_cifar_resnets
    from metassl.utils.augment import DataAugmentation
    from metassl.utils.data import get_loaders
    from metassl.utils.io import (
        find_free_port,
        get_expt_dir_with_bohb_config_id,
        organize_experiment_saving,
    )
    from metassl.utils.meters import ProgressMeter, initialize_all_meters, update_grad_stats_meters
    from metassl.utils.simsiam_alternating import SimSiam
    from metassl.utils.summary import write_to_summary_writer
    from metassl.utils.torch_utils import get_image_data_to_plot  # noqa: 401
    from metassl.utils.torch_utils import (
        accuracy,
        adjust_learning_rate,
        check_and_save_checkpoint,
        deactivate_bn,
        get_newest_model,
        hist_to_image,
        validate,
    )

except ImportError:
    # For execution in command line
    from .models import resnet_cifar as our_cifar_resnets
    from .utils.augment import DataAugmentation
    from .utils.data import get_loaders
    from .utils.io import (
        find_free_port,
        get_expt_dir_with_bohb_config_id,
        organize_experiment_saving,
    )
    from .utils.meters import ProgressMeter, initialize_all_meters, update_grad_stats_meters
    from .utils.simsiam_alternating import SimSiam
    from .utils.summary import write_to_summary_writer
    from .utils.torch_utils import (
        accuracy,
        adjust_learning_rate,
        check_and_save_checkpoint,
        deactivate_bn,
        get_newest_model,
        hist_to_image,
        validate,
    )

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def main(config, expt_dir, bohb_infos=None):
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

    if config.data.dataset == "CIFAR10":
        # Define master port (for preventing 'Address already in use error' when submitting more
        # than 1 jobs on 1 node)
        # Code from:
        # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        master_port = find_free_port()
        config.expt.dist_url = "tcp://localhost:" + str(master_port)
        # if this should still fail: do it via filesystem initialization
        # https://pytorch.org/docs/stable/distributed.html#shared-file-system-initialization

    if config.expt.seed is not None:
        random.seed(config.expt.seed)
        torch.manual_seed(config.expt.seed)
        cudnn.deterministic = True
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
            args=(ngpus_per_node, config, expt_dir, bohb_infos),
        )
    else:
        # Simply call main_worker function
        main_worker(config.expt.gpu, ngpus_per_node, config, expt_dir, bohb_infos)

    # BOHB only ------------------------------------------------------------------------------------
    # Read validation metric from the .txt (as for mp.spawn returning values is not trivial)
    if bohb_infos is not None:
        with open(expt_dir + "/current_val_metric.txt", "r") as f:
            val_metric = f.read()
        print(f"val_metric: {val_metric}")
        return float(val_metric)
    # ----------------------------------------------------------------------------------------------


def main_worker(gpu, ngpus_per_node, config, expt_dir, bohb_infos):
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
    print(f"=> creating model '{config.model.model_type}'")
    if config.data.dataset == "CIFAR10":
        # Use model from our model folder instead from torchvision!
        model = SimSiam(
            our_cifar_resnets.resnet18,
            config.simsiam.dim,
            config.simsiam.pred_dim,
            num_classes=10,
        )
    else:
        model = SimSiam(
            models.__dict__[config.model.model_type],
            config.simsiam.dim,
            config.simsiam.pred_dim,
            num_classes=1000,
        )

    # todo: check backpack + ddp + resnet with sam;
    #  backpack raises errors when using inplace operations
    # if config.expt.image_wise_gradients:
    #     for module in model.modules():
    #         if hasattr(module, "inplace"):
    #             module.inplace = False

    if config.model.turn_off_bn:
        print("Turning off BatchNorm in entire model.")
        deactivate_bn(model)
        model.encoder_head[6].bias.requires_grad = True

    # infer learning rate before changing batch size
    init_lr_pt = config.train.lr * config.train.batch_size / 256
    init_lr_ft = config.finetuning.lr * config.finetuning.batch_size / 256

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
            config.finetuning.batch_size = int(config.finetuning.batch_size / ngpus_per_node)
            config.train.batch_size = int(config.train.batch_size / ngpus_per_node)

            # if config.expt.image_wise_gradients:
            #     model = extend(model)
            #     print("using backpack")

            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[config.expt.gpu], find_unused_parameters=True
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.expt.gpu is not None:
        torch.cuda.set_device(config.expt.gpu)
        model = model.cuda(config.expt.gpu)
        # comment out the following line for debugging
        # TODO: delete line below (metassl code)
        # raise NotImplementedError("Only DistributedDataParallel or gpu mode is supported.")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        # TODO: delete line below (metassl code)
        # raise NotImplementedError("Only DistributedDataParallel or gpu mode is supported.")

    # define loss function (criterion) and optimizer
    criterion_pt = nn.CosineSimilarity(dim=1).cuda(config.expt.gpu)
    criterion_ft = nn.CrossEntropyLoss().cuda(config.expt.gpu)

    optim_params_pt = [
        {"params": model.module.backbone.parameters(), "fix_lr": False},
        {"params": model.module.encoder_head.parameters(), "fix_lr": False},
        {
            "params": model.module.predictor.parameters(),
            "fix_lr": config.simsiam.fix_pred_lr,
        },
    ]

    if config.expt.distributed:
        print(f"world size: {torch.distributed.get_world_size()}")
    print(f"finetuning bs: {config.finetuning.batch_size}")
    print(f"finetuning lr: {config.finetuning.lr}")
    print(f"init_lr_ft: {init_lr_ft}")

    print(f"pre-training bs: {config.train.batch_size}")
    print(f"pre-training lr: {config.train.lr}")
    print(f"init_lr_pt: {init_lr_pt}")

    optimizer_pt = torch.optim.SGD(
        params=optim_params_pt,
        lr=init_lr_pt,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay,
    )

    optimizer_ft = torch.optim.SGD(
        params=model.module.classifier_head.parameters(),
        lr=init_lr_ft,
        momentum=config.finetuning.momentum,
        weight_decay=config.finetuning.weight_decay,
    )

    aug_model, optimizer_aug_model, meters, total_iter = None, None, None, 0
    parameterize_augmentations = False

    if config.learnaug.type == "colorjitter":
        aug_model = DataAugmentation(config)
        optimizer_aug_model = torch.optim.Adam(aug_model.parameters(), 0.001)
        parameterize_augmentations = True

    print(f"Parameterization mode: {parameterize_augmentations} ({config.learnaug.type})")

    # in case a dumped model exist and ssl_model_checkpoint is not set, load that dumped model
    newest_model = get_newest_model(expt_dir)
    if newest_model and config.expt.ssl_model_checkpoint_path is None:
        config.expt.ssl_model_checkpoint_path = newest_model

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
            optimizer_ft.load_state_dict(checkpoint["optimizer_ft"])
            if config.learnaug.type == "colorjitter":
                optimizer_aug_model.load_state_dict(checkpoint["optimizer_aug_model"])
                aug_model = checkpoint["aug_model"]
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
            parameterize_augmentation=parameterize_augmentations,
            bohb_infos=bohb_infos,
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
            parameterize_augmentation=parameterize_augmentations,
            bohb_infos=bohb_infos,
        )

    cudnn.benchmark = True
    writer = None

    if config.expt.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(expt_dir, "tensorboard"))

    if not meters:
        meters = initialize_all_meters()

    assert (
        config.finetuning.epochs == config.train.epochs
    ), "in alternating mode, the number of train and finetuning epochs are assumed to be equal"

    for epoch in range(config.train.start_epoch, config.train.epochs):

        if config.expt.distributed:
            train_sampler_pt.set_epoch(epoch)
            train_sampler_ft.set_epoch(epoch)

        warmup = config.expt.warmup_epochs > epoch

        if warmup:
            print(f"warming up: epoch {epoch} / {config.expt.warmup_epochs}")
            cur_lr_pt = adjust_learning_rate(
                optimizer_pt,
                init_lr=init_lr_pt,
                epoch=epoch,
                total_epochs=config.expt.warmup_epochs,
                warmup=True,
                target_lr=config.expt.warmup_target_lr,
                multiplier=config.expt.warmup_multiplier,
            )
            if not config.expt.warmup_epochs > epoch+1:
                init_lr_pt = cur_lr_pt
        else:
            cur_lr_pt = adjust_learning_rate(
                optimizer_pt, init_lr_pt, epoch, total_epochs=config.train.epochs
            )

        cur_lr_ft = adjust_learning_rate(
            optimizer_ft, init_lr_ft, epoch, total_epochs=config.finetuning.epochs
        )

        # reset ft meter when transitioning from warmup to normal training
        if not warmup and config.expt.warmup_epochs > epoch - 1:
            meters["losses_ft_meter"].reset()

        print(f"current pretrain lr: {cur_lr_pt}, finetune lr: {cur_lr_ft}")

        total_iter = train_one_epoch(
            train_loader_pt=train_loader_pt,
            train_loader_ft=train_loader_ft,
            model=model,
            criterion_pt=criterion_pt,
            criterion_ft=criterion_ft,
            optimizer_pt=optimizer_pt,
            optimizer_ft=optimizer_ft,
            optimizer_aug_model=optimizer_aug_model,
            aug_model=aug_model,
            epoch=epoch,
            total_iter=total_iter,
            config=config,
            writer=writer,
            meters=meters,
            warmup=warmup,
            parameterize_augmentations=parameterize_augmentations,
        )

        # Determine wheter to evaluate on the validation or test set
        if config.finetuning.valid_size > 0:
            loader_ft = valid_loader_ft
            writer_scalar_mode = "Valid"
        else:
            loader_ft = test_loader_ft
            writer_scalar_mode = "Test"
        # BOHB only --------------------------------------------------------------------------------
        # TODO: @Diane - Refactor - no priority
        if bohb_infos is not None and config.bohb.budget_mode == "epochs":
            if (
                epoch % config.expt.eval_freq == 0
                or epoch % int(bohb_infos["bohb_budget"] - 1) == 0
            ):
                top1_avg = validate(loader_ft, model, criterion_ft, config, finetuning=True)
                if config.expt.rank == 0:
                    writer.add_scalar(writer_scalar_mode + "/Accuracy@1", top1_avg, total_iter)
        # ------------------------------------------------------------------------------------------
        else:
            # evaluate on validation/test set
            if epoch % config.expt.eval_freq == 0 or epoch % (config.train.epochs - 1) == 0:
                top1_avg = validate(loader_ft, model, criterion_ft, config, finetuning=True)
                if config.expt.rank == 0:
                    writer.add_scalar(writer_scalar_mode + "/Accuracy@1", top1_avg, total_iter)

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
                    optimizer_ft=optimizer_ft,
                    expt_dir=expt_dir,
                    meters=meters,
                    optimizer_aug_model=optimizer_aug_model,
                    aug_model=aug_model,
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
                optimizer_ft=optimizer_ft,
                expt_dir=expt_dir,
                meters=meters,
                optimizer_aug_model=optimizer_aug_model,
                aug_model=aug_model,
            )

    if config.expt.rank == 0:
        writer.close()

        batch_time_meter = meters["batch_time_meter"]
        print(f"total batch time elapsed: {batch_time_meter.sum:.2f}s")

    # BOHB only ------------------------------------------------------------------------------------
    # Save validation metric in a .txt (as for mp.spawn returning values is not trivial)
    if bohb_infos is not None:
        with open(expt_dir + "/current_val_metric.txt", "w+") as f:
            f.write(f"{top1_avg.item()}\n")
    # ----------------------------------------------------------------------------------------------


def train_one_epoch(
    train_loader_pt,
    train_loader_ft,
    model,
    criterion_pt,
    criterion_ft,
    optimizer_pt,
    optimizer_ft,
    optimizer_aug_model,
    aug_model,
    epoch,
    total_iter,
    config,
    writer,
    meters,
    warmup=False,
    parameterize_augmentations=False,
):
    # general meters
    batch_time_meter = meters["batch_time_meter"]
    data_time_meter = meters["data_time_meter"]
    losses_pt_meter = meters["losses_pt_meter"]
    losses_ft_meter = meters["losses_ft_meter"]
    top1_meter = meters["top1_meter"]
    top5_meter = meters["top5_meter"]

    # global meters
    cos_sim_ema_meter_global = meters["cos_sim_ema_meter_global"]
    cos_sim_ema_meter_standardized_global = meters["cos_sim_ema_meter_standardized_global"]

    dot_prod_meter_global = meters["dot_prod_meter_global"]
    eucl_dis_meter_global = meters["eucl_dis_meter_global"]
    norm_pt_meter_global = meters["norm_pt_meter_global"]
    norm_ft_meter_global = meters["norm_ft_meter_global"]
    target_std_meter = meters["target_std_meter"]

    # layer-wise meters
    cos_sim_ema_meter_lw = meters["cos_sim_ema_meter_lw"]
    cos_sim_std_meter_lw = meters["cos_sim_std_meter_lw"]
    cos_sim_ema_meter_standardized_lw = meters["cos_sim_ema_meter_standardized_lw"]
    cos_sim_std_meter_standardized_lw = meters["cos_sim_std_meter_standardized_lw"]

    dot_prod_avg_meter_lw = meters["dot_prod_avg_meter_lw"]
    dot_prod_std_meter_lw = meters["dot_prod_std_meter_lw"]
    eucl_dis_avg_meter_lw = meters["eucl_dis_avg_meter_lw"]
    eucl_dis_std_meter_lw = meters["eucl_dis_std_meter_lw"]
    norm_pt_avg_meter_lw = meters["norm_pt_avg_meter_lw"]
    norm_pt_std_meter_lw = meters["norm_pt_std_meter_lw"]
    norm_ft_avg_meter_lw = meters["norm_ft_avg_meter_lw"]
    norm_ft_std_meter_lw = meters["norm_ft_std_meter_lw"]

    reward_meter = meters["reward_meter"]
    norm_aug_brightness_grad_meter = meters["norm_aug_brightness_grad_meter"]
    norm_aug_contrast_grad_meter = meters["norm_aug_contrast_grad_meter"]
    norm_aug_saturation_grad_meter = meters["norm_aug_saturation_grad_meter"]
    norm_aug_hue_grad_meter = meters["norm_aug_hue_grad_meter"]

    meters_to_print = [
        batch_time_meter,
        losses_pt_meter,
        losses_ft_meter,
        top1_meter,
        cos_sim_ema_meter_lw,
        cos_sim_ema_meter_standardized_lw,
        cos_sim_ema_meter_global,
        cos_sim_ema_meter_standardized_global,
        target_std_meter,
    ]

    progress = ProgressMeter(
        num_batches=len(train_loader_pt),
        meters=meters_to_print,
        prefix=f"Epoch: [{epoch}]",
    )

    end = time.time()
    if config.finetuning.valid_size > 0:
        # As we only need a validation set for finetuning:
        # len(train_loader_pt) < len(train_loader_ft)
        # TODO @Diane - Check out if this won't cause any problems
        pass
    else:
        assert len(train_loader_pt) <= len(train_loader_ft), (
            "So since this seems to break, we should write code to run multiple finetune epoch "
            "per pretrain epoch"
        )

    for i, ((images_pt, _), (images_ft, target_ft)) in enumerate(
        zip(train_loader_pt, train_loader_ft)
    ):

        total_iter += 1

        if parameterize_augmentations:
            # if config.expt.rank == 0 and i % (config.expt.print_freq * 1000) == 0:
            #     rand_int = torch.randint(high=images_pt.shape[0], size=(1,))
            #     # permute from CHW to HWC for pyplot
            #     untransformed_image = torch.permute(images_pt[rand_int].squeeze(), (1, 2, 0)).
            #     cpu()

            if config.expt.gpu is not None and not isinstance(images_pt, list):
                images_pt = images_pt.cuda(config.expt.gpu, non_blocking=True)

            indices, logprobs, strengths = aug_model.sample_logprobs()
            images_pt = aug_model(
                images_pt,
                idx_b=indices["idx_b"],
                idx_c=indices["idx_c"],
                idx_s=indices["idx_s"],
                idx_h=indices["idx_h"],
            )

        if config.expt.gpu is not None:
            images_pt[0] = images_pt[0].contiguous()
            images_pt[1] = images_pt[1].contiguous()
            images_pt[0] = images_pt[0].cuda(config.expt.gpu, non_blocking=True)
            images_pt[1] = images_pt[1].cuda(config.expt.gpu, non_blocking=True)
            images_ft = images_ft.cuda(config.expt.gpu, non_blocking=True)
        target_ft = target_ft.cuda(config.expt.gpu, non_blocking=True)

        if (
            parameterize_augmentations
            and config.expt.rank == 0
            and i % (config.expt.print_freq * 1000) == 0
        ):
            # fmt: off
            # image_data_to_plot_dict = get_image_data_to_plot(rand_int=rand_int, untransformed_image=untransformed_image, images_pt=images_pt, images_ft=images_ft, target_ft=target_ft, strengths=strengths)  # noqa: E501, E241
            # fmt: on
            pass

        alternating_mode = False if config.expt.is_non_grad_based else True  # default is True
        loss_pt, backbone_grads_pt_lw, backbone_grads_pt_global, z1, z2 = pretrain(
            model,
            images_pt,
            criterion_pt,
            optimizer_pt,
            losses_pt_meter,
            data_time_meter,
            end,
            alternating_mode=alternating_mode,
        )

        z_std_normalized = np.std(
            z1.cpu().numpy() / (torch.linalg.norm(z1, 2).cpu().numpy() + 1e-9)
        )
        target_std_meter.update(z_std_normalized)

        backbone_grads_ft_lw, backbone_grads_ft_global, reward = None, None, None
        if not warmup:
            loss_ft, backbone_grads_ft_lw, backbone_grads_ft_global = finetune(
                model,
                images_ft,
                target_ft,
                criterion_ft,
                optimizer_ft,
                losses_ft_meter,
                top1_meter,
                top5_meter,
                config=config,
                alternating_mode=alternating_mode,
            )

            if parameterize_augmentations:
                optimizer_aug_model.zero_grad()
                reward = cos_sim_ema_meter_lw.val - cos_sim_ema_meter_lw.ema

                color_jitter_logprob_b = -logprobs["logprob_b"] * reward
                color_jitter_logprob_c = -logprobs["logprob_c"] * reward
                color_jitter_logprob_s = -logprobs["logprob_s"] * reward
                color_jitter_logprob_h = -logprobs["logprob_h"] * reward

                aug_loss = (
                    color_jitter_logprob_b
                    + color_jitter_logprob_c
                    + color_jitter_logprob_s
                    + color_jitter_logprob_h
                )
                aug_loss.backward()

                optimizer_aug_model.step()

        else:
            losses_ft_meter.update(np.inf)
            loss_ft = np.inf

        grads = {
            "backbone_grads_pt_lw": backbone_grads_pt_lw,
            "backbone_grads_pt_global": backbone_grads_pt_global,
            "backbone_grads_ft_lw": backbone_grads_ft_lw,
            "backbone_grads_ft_global": backbone_grads_ft_global,
            "reward": reward,
        }

        update_grad_stats_meters(
            grads=grads,
            meters=meters,
            warmup=warmup,
            parameterize_augmentations=parameterize_augmentations,
            aug_model=aug_model,
        )

        main_stats_meters = [
            cos_sim_ema_meter_global,
            cos_sim_ema_meter_standardized_global,
            cos_sim_ema_meter_lw,
            cos_sim_ema_meter_standardized_lw,
            cos_sim_std_meter_standardized_lw,
            dot_prod_meter_global,
            dot_prod_avg_meter_lw,
            eucl_dis_meter_global,
            eucl_dis_avg_meter_lw,
            norm_pt_meter_global,
            norm_pt_avg_meter_lw,
            norm_ft_meter_global,
            norm_ft_avg_meter_lw,
            target_std_meter,
        ]

        additional_stats_meters = [
            cos_sim_std_meter_lw,
            dot_prod_std_meter_lw,
            eucl_dis_std_meter_lw,
            norm_pt_std_meter_lw,
            norm_ft_std_meter_lw,
        ]

        meters_to_plot = {
            "main_meters": main_stats_meters,
            "additional_stats_meters": additional_stats_meters,
        }

        if parameterize_augmentations:
            main_stats_meters.append(reward_meter)
            aug_param_meters = [
                norm_aug_brightness_grad_meter,
                norm_aug_contrast_grad_meter,
                norm_aug_saturation_grad_meter,
                norm_aug_hue_grad_meter,
            ]
            meters_to_plot["aug_param_meters"] = aug_param_meters

        # measure elapsed time
        batch_time_meter.update(time.time() - end)
        end = time.time()

        if i % config.expt.print_freq == 0:
            progress.display(i)
        if config.expt.rank == 0 and i % config.expt.write_summary_frequency:
            write_to_summary_writer(
                total_iter,
                loss_pt,
                loss_ft,
                data_time_meter,
                batch_time_meter,
                optimizer_pt,
                optimizer_ft,
                top1_meter,
                top5_meter,
                meters_to_plot,
                writer,
            )

        # expensive stats
        if (
            parameterize_augmentations
            and config.expt.rank == 0
            and i % (config.expt.print_freq * 1000) == 0
        ):
            img = hist_to_image(
                aug_model.color_jitter_histogram_brightness,
                "Color Jitter Strength Brightness Counts",
            )
            writer.add_image(
                tag="Advanced Stats/color jitter strength brightness",
                img_tensor=img,
                global_step=total_iter,
            )

            img = hist_to_image(
                aug_model.color_jitter_histogram_contrast,
                "Color Jitter Strength Contrast Counts",
            )
            writer.add_image(
                tag="Advanced Stats/color jitter strength contrast",
                img_tensor=img,
                global_step=total_iter,
            )

            img = hist_to_image(
                aug_model.color_jitter_histogram_saturation,
                "Color Jitter Strength Saturation Counts",
            )
            writer.add_image(
                tag="Advanced Stats/color jitter strength saturation",
                img_tensor=img,
                global_step=total_iter,
            )

            img = hist_to_image(
                aug_model.color_jitter_histogram_hue, "Color Jitter Strength Hue Counts"
            )
            writer.add_image(
                tag="Advanced Stats/color jitter strength hue",
                img_tensor=img,
                global_step=total_iter,
            )

            # fmt: off
            # img = tensor_to_image(image_data_to_plot_dict["untransformed_image"], f"Randomly sampled untransformed image")  # noqa: E501, E241
            # writer.add_image(tag="Advanced Stats/sampled untransformed image", img_tensor=img, global_step=total_iter)  # noqa: E501, E241

            # title = image_data_to_plot_dict["title"]

            # img = tensor_to_image(image_data_to_plot_dict["img0"], f"Randomly sampled transformed image 1\n {title}")  # noqa: E501, E241
            # writer.add_image(tag="Advanced Stats/sampled transformed image 1", img_tensor=img, global_step=total_iter)  # noqa: E501, E241

            # img = tensor_to_image(image_data_to_plot_dict["img1"], f"Randomly sampled transformed image 2\n {title}")  # noqa: E501, E241
            # writer.add_image(tag="Advanced Stats/sampled transformed image 2", img_tensor=img, global_step=total_iter)  # noqa: E501, E241

            # ft_img = tensor_to_image(image_data_to_plot_dict["ft_img"], f"Randomly sampled finetuning image\n with label {image_data_to_plot_dict['ft_label']}")  # noqa: E501, E241
            # writer.add_image(tag="Advanced Stats/sampled finetuning image", img_tensor=ft_img, global_step=total_iter)  # noqa: E501, E241
            # fmt: on
    return total_iter


def pretrain(
    model,
    images_pt,
    criterion_pt,
    optimizer_pt,
    losses_pt,
    data_time,
    end,
    alternating_mode=False,
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

    # if config.expt.image_wise_gradients:
    #     with backpack(BatchGrad()):
    #         loss_pt.backward()
    # else:

    loss_pt.backward()
    # step does not change .grad field of the parameters.
    optimizer_pt.step()

    if alternating_mode:
        for key, param in model.module.backbone.named_parameters():
            grad_tensor = param.grad.detach_().clone().flatten()
            backbone_grads_lw[key] = torch.tensor(grad_tensor)
            backbone_grads_global = torch.cat([backbone_grads_global, grad_tensor], dim=0)

    return loss_pt, backbone_grads_lw, backbone_grads_global, z1, z2


def finetune(
    model,
    images_ft,
    target_ft,
    criterion_ft,
    optimizer_ft,
    losses_ft_meter,
    top1_meter,
    top5_meter,
    config,
    alternating_mode=False,
):
    backbone_grads_lw = OrderedDict()
    backbone_grads_global = torch.Tensor().cuda()

    # fine-tuning
    model.eval()

    optimizer_ft.zero_grad()
    # in finetuning mode, we only optimize the classifier head's parameters
    # -> turn on backbone params grad computation before forward is called
    if alternating_mode:
        model.module.backbone.requires_grad_(True)
    else:
        model.module.backbone.requires_grad_(False)

    model.module.classifier_head.requires_grad_(True)

    # compute outputs
    output_ft = model(images_ft, finetuning=True)
    loss_ft = criterion_ft(output_ft, target_ft)

    # if config.expt.image_wise_gradients:
    #     with backpack(BatchGrad()):
    #         loss_ft.backward()
    # else:

    loss_ft.backward()

    if alternating_mode:
        for key, param in model.module.backbone.named_parameters():
            grad_tensor = param.grad.detach_().clone().flatten()
            backbone_grads_lw[key] = torch.tensor(grad_tensor)
            backbone_grads_global = torch.cat([backbone_grads_global, grad_tensor], dim=0)

    # compute losses and measure accuracy
    acc1, acc5 = accuracy(output_ft, target_ft, topk=(1, 5))
    losses_ft_meter.update(loss_ft.item(), images_ft.size(0))
    top1_meter.update(acc1[0], images_ft.size(0))
    top5_meter.update(acc5[0], images_ft.size(0))

    # only optimizes classifier head parameters
    optimizer_ft.step()

    # just to make sure to prevent grad leakage
    for param in model.module.parameters():
        param.grad = None

    return loss_ft, backbone_grads_lw, backbone_grads_global


if __name__ == "__main__":
    user = os.environ.get("USER")

    config = get_parsed_config()
    print("\n\n\n\nConfig:\n", config, "\n\n\n\n")

    # Check whether it is a BOHB run or not + organize expt_dir accordingly
    is_bohb_run = True if config.expt.expt_mode.endswith("BOHB") else False
    expt_dir = organize_experiment_saving(user=user, config=config, is_bohb_run=is_bohb_run)

    # Error check
    if (
        config.finetuning.data_augmentation != "none"
        and config.expt.data_augmentation_mode != "probability_augment"
    ):
        raise ValueError(
            "If you use data augmentation for finetuning, 'probability_augment' is required as "
            "the data_augmentation_mode!"
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

    # Run BOHB / main
    if is_bohb_run:
        from metassl.hyperparameter_optimization.master import start_bohb_master

        start_bohb_master(yaml_config=config, expt_dir=expt_dir)

    else:
        main(config=config, expt_dir=expt_dir)
