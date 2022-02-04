import argparse
import os
import time
import math
import random
import warnings
from os import path, makedirs
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
import torch.backends.cudnn as cudnn

from metassl.baselines.simsiam.augment_datasets import Cifar10AugmentPT
from metassl.baselines.simsiam.get_sampler import get_train_valid_sampler
from metassl.baselines.simsiam.loader import TwoCropsTransform
from metassl.baselines.simsiam.model_factory import SimSiam
from metassl.baselines.simsiam.criterion import SimSiamLoss
from metassl.baselines.simsiam.validation import KNNValidation
from metassl.utils.albumentation_datasets import Cifar10AlbumentationsPT
# from metassl.utils.probability_augment import probability_augment


def main(args, trial_dir=None, bohb_infos=None):
    # adding seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        # cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        # torch.use_deterministic_algorithms(True)
        # + ADD DATALOADER: https://pytorch.org/docs/stable/notes/randomness.html
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )
    # BOHB only --------------------------------------------------------------------------------------------------------
    if bohb_infos is not None:
        # Integrate budget based on budget_mode
        if args.budget_mode == "epochs":
            args.pt_epochs = int(bohb_infos['bohb_budget'])
        else:
            raise ValueError(f"Budget mode '{args.budget_mode}' not implemented yet!")

        # Add --bohb.configspace_mode to bohb_infos
        bohb_infos['bohb_configspace'] = args.configspace_mode

        # Create subfoler for each config_id (directory where tensorboard and checkpoints are being saved)
        exp_dir_id = get_exp_dir_with_bohb_config_id(trial_dir, bohb_infos['bohb_config_id'])
        args.exp_dir = exp_dir_id

        print(f"\n\n\n\n\n\n{bohb_infos=}\n\n\n\n\n\n")
    # ------------------------------------------------------------------------------------------------------------------

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)

    trial_dir = path.join(args.exp_dir, args.trial)
    logger = SummaryWriter(trial_dir)
    print(f"Tensorboard logs kept in {logger.log_dir}")
    print(vars(args))

    # ------------------------------------------------------------------------------------------------------------------
    # Specify data augmentation hyperparameters for the pretraining part
    # ------------------------------------------------------------------------------------------------------------------
    # TODO: @Diane - put that into a separate function
    # TODO: @Diane - Add gaussian blur for ImageNet
    # Defaults
    p_colorjitter = 0.8
    p_grayscale = 0.2
    # p_gaussianblur = 0.5 if dataset_name == 'ImageNet' else 0
    brightness_strength = 0.4
    contrast_strength = 0.4
    saturation_strength = 0.4
    hue_strength = 0.1
    if args.use_fix_aug_params:
        # You can overwrite parameters here if you want to try out a specific setting.
        # Due to the flag, default experiments won't be affected by this.
        p_colorjitter = args.p_colorjitter
        p_grayscale = args.p_grayscale
        # p_gaussianblur = 0.5 if dataset_name == 'ImageNet' else 0
        brightness_strength = args.brightness_strength
        contrast_strength = args.contrast_strength
        saturation_strength = args.saturation_strength
        hue_strength = args.hue_strength

    # BOHB - SimSiam Cifar10 augment configspace
    if bohb_infos is not None and bohb_infos['bohb_configspace'] == 'cifar10_simsiam_augment':
        p_colorjitter = bohb_infos['bohb_config']['p_colorjitter']
        p_grayscale = bohb_infos['bohb_config']['p_grayscale']
        # p_gaussianblur = bohb_infos['bohb_config']['p_gaussianblur'] if dataset_name == 'ImageNet' else 0
        brightness_strength = bohb_infos['bohb_config']['brightness_strength']
        contrast_strength = bohb_infos['bohb_config']['contrast_strength']
        saturation_strength = bohb_infos['bohb_config']['saturation_strength']
        hue_strength = bohb_infos['bohb_config']['hue_strength']

    # BOHB - color jitter strengths configspace
    elif bohb_infos is not None and bohb_infos['bohb_configspace'].endswith('color_jitter_strengths'):
        brightness_strength = bohb_infos['bohb_config']['brightness_strength']
        contrast_strength = bohb_infos['bohb_config']['contrast_strength']
        saturation_strength = bohb_infos['bohb_config']['saturation_strength']
        hue_strength = bohb_infos['bohb_config']['hue_strength']

    elif bohb_infos is not None and bohb_infos['bohb_configspace'] == "weight_decay_annealing":
        print("\nWEIGHT DECAY ANNEALING\n")

    elif bohb_infos is not None:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    # Specify pretraining learning rate
    # ------------------------------------------------------------------------------------------------------------------
    if bohb_infos is not None and bohb_infos['bohb_configspace'] == 'lr_color_jitter_strengths':
        args.pt_learning_rate = bohb_infos['bohb_config']['pt_learning_rate']

    # ------------------------------------------------------------------------------------------------------------------
    # For testing augs
    print(f"\nPRETRAINING PARAMS")
    print(f"{p_colorjitter=}")
    print(f"{p_grayscale=}")
    # print(f"{p_gaussianblur=}")
    print(f"{brightness_strength=}")
    print(f"{contrast_strength=}")
    print(f"{saturation_strength=}")
    print(f"{hue_strength=}")

    # For testing pt_learning_rate
    print(f"{args.pt_learning_rate=}")
    # ------------------------------------------------------------------------------------------------------------------
    if args.is_probabilityaugment:
        dataset_name = "CIFAR10"
        get_fine_tuning_loaders = False
        use_fix_aug_params = False
        finetuning_data_augmentation = 'none'
        train_transform, valid_transform = probability_augment(dataset_name, get_fine_tuning_loaders, bohb_infos,
                                                               use_fix_aug_params, finetuning_data_augmentation)
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=brightness_strength, contrast=contrast_strength, saturation=saturation_strength, hue=hue_strength)  # not strengthened
            ], p=p_colorjitter),
            transforms.RandomGrayscale(p=p_grayscale),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    if args.is_trivialaugment:
        print("\n\n\n TRIVIAL AUGMENT \n\n\n")
        train_set = Cifar10AugmentPT(root=args.data_root,
                                     train=True,
                                     download=True,
                                     transform=None,
                                     augmentation_mode="trivialaugment",
                                     augmentation_ops_mode=args.trivialaugment_ops_mode)
    elif args.is_smartsamplingaugment:
        print("\n\n\n SMARTSAMPLING AUGMENT \n\n\n")
        train_set = Cifar10AugmentPT(root=args.data_root,
                                     train=True,
                                     download=True,
                                     transform=None,
                                     augmentation_mode="smartsamplingaugment",
                                     augmentation_ops_mode=None)
    elif args.is_probabilityaugment:
        print("\n\n\n PROBABILITY AUGMENT \n\n\n")
        train_set = Cifar10AlbumentationsPT(root=args.data_root,
                                     train=True,
                                     download=True,
                                     transform=train_transform,)
    else:
        train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=TwoCropsTransform(train_transforms))
    train_sampler, _ = get_train_valid_sampler(args, train_set)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.pt_batch_size,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    model = SimSiam(args)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.pt_learning_rate,
                          momentum=args.pt_momentum,
                          weight_decay=args.pt_weight_decay)

    criterion = SimSiamLoss(args.loss_version)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True

    start_epoch = 1
    if args.pt_resume is not None:
        if path.isfile(args.pt_resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.pt_resume)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.pt_resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.pt_resume))

    # routine
    best_acc = 0.0
    validation = KNNValidation(args, model.encoder)
    for epoch in range(start_epoch, args.pt_epochs + 1):

        adjust_learning_rate(optimizer, epoch, args)
        print("Training...")

        # train and log for one epoch
        train_loss, current_weight_decay, image_1, image_2 = train(train_loader, model, criterion, optimizer, epoch, args, bohb_infos)
        logger.add_scalar('Loss/train', train_loss, epoch)
        # logger.add_images('Train/Image1', image_1, epoch)
        # logger.add_images('Train/Image2', image_2, epoch)

        if epoch % args.eval_freq == 0:
            print("Validating...")
            val_top1_acc = validation.eval()
            print('Top1: {}'.format(val_top1_acc))

            # save the best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc

                save_checkpoint(args, epoch, model, optimizer, best_acc,
                                path.join(trial_dir, '{}_best.pth'.format(args.trial)),
                                'Saving the best model!')
            logger.add_scalar('Acc/val_top1', val_top1_acc, epoch)
            # Log weight decay in Tensorboard
            logger.add_scalar('Weight_decay/PT', current_weight_decay, epoch)

        # save the model
        if epoch % args.save_freq == 0:
            save_checkpoint(args, epoch, model, optimizer, val_top1_acc,
                            path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
                            'Saving...')

    print('Best accuracy:', best_acc)

    # save final model
    save_checkpoint(args, epoch, model, optimizer, best_acc,
                    path.join(trial_dir, '{}_last.pth'.format(args.trial)),
                    'Saving the model at the last epoch.')
    if bohb_infos is not None:
        return trial_dir


def train(train_loader, model, criterion, optimizer, epoch, args, bohb_infos):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    print(f"\n\n\n")
    if args.do_weight_decay_annealing:
        # Select weight decay parameters (based on whether it is a bohb run or not)
        current_epoch = epoch
        max_epochs = args.pt_epochs
        if args.is_bohb_run and args.configspace_mode == "weight_decay_annealing":
            start_weight_decay = bohb_infos['bohb_config']['start_weight_decay_pt']
            end_weight_decay = bohb_infos['bohb_config']['end_weight_decay_pt']
        else:
            # Set weight decay parameters manually
            start_weight_decay = 0.001
            end_weight_decay = 0.000001
        # Do annealing
        if epoch == 1:
            for group in optimizer.param_groups:
                group['weight_decay'] = start_weight_decay
                current_weight_decay = group['weight_decay']
        else:
            for group in optimizer.param_groups:
                group['weight_decay'] = end_weight_decay + 1/2 * (start_weight_decay - end_weight_decay) * (1 + math.cos(current_epoch/max_epochs * math.pi))
                current_weight_decay = group['weight_decay']
    else:
        # Non-annealing case
        current_weight_decay = args.pt_weight_decay
    print(f"PT: {current_weight_decay=}")

    for i, (images, _) in enumerate(train_loader):

        if args.is_trivialaugment or args.is_smartsamplingaugment:
        # To prevent the following error:
        # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
            images[0] = images[0].float()
            images[1] = images[1].float()

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        outs = model(im_aug1=images[0], im_aug2=images[1])
        loss = criterion(outs['z1'], outs['z2'], outs['p1'], outs['p2'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, current_weight_decay, images[0], images[1]


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.pt_learning_rate
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.pt_epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(args, epoch, model, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer


def get_exp_dir_with_bohb_config_id(expt_dir, bohb_config_id):
    config_id_path = "-".join(str(sub_id) for sub_id in bohb_config_id)
    expt_dir_id = os.path.join(expt_dir, config_id_path)
    return expt_dir_id


if __name__ == '__main__':
    main()
