import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets  # do not remove this  # noqa: F401
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import InterpolationMode

from metassl.utils.imagenet import ImageNet

from .simsiam import GaussianBlur, TwoCropsTransform

# fmt on


normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

normalize_cifar10 = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)

normalize_cifar100 = transforms.Normalize(
    mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
)


def get_train_valid_loader(
    config,
    neps_hyperparameters,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=1,
    pin_memory=True,
    download=False,
    dataset_name="ImageNet",
    distributed=False,
    drop_last=True,
    get_fine_tuning_loaders=False,
    parameterize_augmentation=False,
    bohb_infos=None,
    dataset_percentage_usage=100,
    use_fix_aug_params=False,
    data_augmentation_mode="default",
    finetuning_data_augmentation="none",
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over a dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - dataset_name: the dataset name as a string, supported: "CIFAR10", "CIFAR100", "ImageNet"
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    # error_msg = "[!] valid_size should be in the range [0, 1]."
    # assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    allowed_datasets = ["CIFAR10", "CIFAR100", "ImageNet"]
    if dataset_name not in allowed_datasets:
        print(f"dataset name should be in {allowed_datasets}")
        exit()

    dataset = eval("datasets." + dataset_name)

    if get_fine_tuning_loaders:
        print(f"using finetuning dataset: {dataset}")
    else:
        print(f"using pretraining dataset: {dataset}")

    if get_fine_tuning_loaders or data_augmentation_mode == "default":
        # default SimSiam Stuff + Fabio Stuff
        # TODO @Fabio/Diane - generate specific mode for Fabio stuff
        train_transform, valid_transform = get_train_valid_transforms(
            config,
            dataset_name,
            use_fix_aug_params,
            bohb_infos,
            get_fine_tuning_loaders,
            parameterize_augmentation,
            neps_hyperparameters,
        )
    else:
        from .probability_augment import probability_augment

        if data_augmentation_mode == "probability_augment":
            train_transform = probability_augment(
                config,
                dataset_name,
                use_fix_aug_params,
                neps_hyperparameters,
            )
        else:
            # For TrivialAugment, RandAugment, SmartAugment, and SmartSamplingAugment
            train_transform = None  # Done in Cifar10AugmentationPT

        valid_transform = TwoCropsTransform(
            transforms.Compose(
                [
                    transforms.Resize(int(32 * (8 / 7)), interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    normalize_cifar10 if dataset_name == "CIFAR10" else normalize_cifar100,
                ]
            )
        )

    if dataset_name == "ImageNet":
        # hardcoded for now
        root = "/data/datasets/ImageNet/imagenet-pytorch"
        # root = "/data/datasets/ILSVRC2012"

        # load the dataset
        train_dataset = ImageNet(
            root=root,
            split="train",
            transform=train_transform,
            ignore_archive=True,
        )

        valid_dataset = ImageNet(
            root=root,
            split="train",
            transform=valid_transform,
            ignore_archive=True,
        )
    elif dataset_name == "CIFAR10":
        # train_dataset
        # ------------------------------------------------------------------------------------------
        if data_augmentation_mode == "probability_augment" and not get_fine_tuning_loaders:
            from .albumentation_datasets import Cifar10AlbumentationsPT

            train_dataset = Cifar10AlbumentationsPT(
                root="datasets/CIFAR10",
                train=True,
                download=True,
                transform=train_transform,
            )

        elif data_augmentation_mode != "default" and not get_fine_tuning_loaders:
            from .albumentation_datasets import Cifar10AugmentationPT

            train_dataset = Cifar10AugmentationPT(
                root="datasets/CIFAR10",
                train=True,
                download=True,
                transform=train_transform,
                data_augmentation_mode=data_augmentation_mode,
                neps_hyperparameters=neps_hyperparameters,
            )

        else:
            train_dataset = torchvision.datasets.CIFAR10(
                root="datasets/CIFAR10",
                train=True,
                download=True,
                transform=train_transform,
            )
        # valid_dataset
        # ------------------------------------------------------------------------------------------
        valid_dataset = torchvision.datasets.CIFAR10(
            root="datasets/CIFAR10",
            train=True,
            download=True,
            transform=valid_transform,
        )
    elif dataset_name == "CIFAR100":
        # train_dataset
        # ------------------------------------------------------------------------------------------
        if data_augmentation_mode == "probability_augment":
            pass

        else:
            train_dataset = torchvision.datasets.CIFAR100(
                root="datasets/CIFAR100",
                train=True,
                download=True,
                transform=train_transform,
            )
        # valid_dataset
        # ------------------------------------------------------------------------------------------
        valid_dataset = torchvision.datasets.CIFAR100(
            root="datasets/CIFAR100",
            train=True,
            download=True,
            transform=valid_transform,
        )
    else:
        # not supported
        raise ValueError("invalid dataset name=%s" % dataset)

    num_train = int(len(train_dataset) / 100 * dataset_percentage_usage)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if np.isclose(valid_size, 0.0):
        train_idx, valid_idx = indices, indices
    else:
        train_idx, valid_idx = indices[split:], indices[:split]

    valid_sampler = SubsetRandomSampler(valid_idx)

    if distributed:
        train_sampler = DistributedSampler(torch.tensor(train_idx))
        # TODO: use distributed valid_sampler and average accuracies for more efficient validation
    else:
        train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # visualize some images
    # if show_sample:
    #     sample_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=9,
    #         shuffle=shuffle,
    #         num_workers=num_workers,
    #         pin_memory=pin_memory,
    #     )
    #     data_iter = iter(sample_loader)
    #     images, labels = data_iter.next()
    #     X = images.numpy().transpose([0, 2, 3, 1])
    #     plot_images(X, labels)

    if np.isclose(valid_size, 0.0):
        return train_loader, None, train_sampler, None
    else:
        return train_loader, valid_loader, train_sampler, valid_sampler


def get_test_loader(
    batch_size,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    download=False,
    dataset_name="ImageNet",
    drop_last=False,
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-100 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - dataset_name: the dataset name as a string, supported: "CIFAR10", "CIFAR100", "ImageNet"
    Returns
    -------
    - data_loader: test set iterator.
    """
    allowed_datasets = ["CIFAR10", "CIFAR100", "ImageNet"]
    if dataset_name not in allowed_datasets:
        print(f"dataset name should be in {allowed_datasets}")
        exit()

    dataset = eval("datasets." + dataset_name)

    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        transform = transforms.Compose(
            [
                transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                normalize_cifar10 if dataset_name == "CIFAR10" else normalize_cifar100,
            ]
        )

    elif dataset_name == "ImageNet":

        transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_imagenet,
            ]
        )

    else:
        # not supported
        raise ValueError("invalid dataset name=%s" % dataset)

    if dataset_name == "ImageNet":
        # hardcoded for now
        # TODO: move to imagenet.py
        root = "/data/datasets/ImageNet/imagenet-pytorch"
        # load the dataset
        dataset = ImageNet(
            root=root,
            split="val",
            transform=transform,
            ignore_archive=True,
        )
    elif dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root="datasets/CIFAR10", train=False, download=True, transform=transform
        )
    elif dataset_name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(
            root="datasets/CIFAR100", train=False, download=True, transform=transform
        )
    else:
        raise NotImplementedError("Dataset not supported.")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


def get_train_valid_transforms(
    config,
    dataset_name,
    use_fix_aug_params,
    bohb_infos,
    get_fine_tuning_loaders,
    parameterize_augmentation,
    neps_hyperparameters,
):
    # ----------------------------------------------------------------------------------------------
    # Specify data augmentation hyperparameters for the pretraining part
    # ----------------------------------------------------------------------------------------------
    # TODO: @Diane - put that into a separate function
    # Defaults
    p_colorjitter = 0.8
    p_grayscale = 0.2
    p_gaussianblur = 0.5 if dataset_name == "ImageNet" else 0
    p_horizontal_flip = 0.5
    p_solarize = 0
    brightness_strength = 0.4
    contrast_strength = 0.4
    saturation_strength = 0.4
    hue_strength = 0.1
    solarize_threshold = 255

    if use_fix_aug_params:
        # You can overwrite parameters here if you want to try out a specific setting.
        # Due to the flag, default experiments won't be affected by this.
        p_colorjitter = 0.32011207176587564
        p_grayscale = 0.13152910682197913
        p_solarize = 0.38645378707287636
        solarize_threshold = 121
        p_gaussianblur = 0.5 if dataset_name == "ImageNet" else 0
        brightness_strength = 0.5918737491981877
        contrast_strength = 1.1513307570530626
        saturation_strength = 0.01767797917203415
        hue_strength = 0.08749582439198282

    # BOHB - probability augment configspace
    if bohb_infos is not None and bohb_infos["bohb_configspace"].endswith(
        "probability_simsiam_augment"
    ):
        p_colorjitter = bohb_infos["bohb_config"]["p_colorjitter"]
        p_grayscale = bohb_infos["bohb_config"]["p_grayscale"]
        p_gaussianblur = (
            bohb_infos["bohb_config"]["p_gaussianblur"] if dataset_name == "ImageNet" else 0
        )

    # BOHB - color jitter strengths configspace
    if bohb_infos is not None and bohb_infos["bohb_configspace"] == "color_jitter_strengths":
        brightness_strength = bohb_infos["bohb_config"]["brightness_strength"]
        contrast_strength = bohb_infos["bohb_config"]["contrast_strength"]
        saturation_strength = bohb_infos["bohb_config"]["saturation_strength"]
        hue_strength = bohb_infos["bohb_config"]["hue_strength"]

    # NEPS only ------------------------------------------------------------------------------------
    if (
        config.neps.is_neps_run
        and neps_hyperparameters is not None
        and (
            config.neps.config_space == "data_augmentation"
            or config.neps.config_space == "combined"
        )
    ):
        # Probabilities
        p_colorjitter = neps_hyperparameters["p_colorjitter"]
        p_grayscale = neps_hyperparameters["p_grayscale"]
        p_horizontal_flip = neps_hyperparameters["p_horizontal_flip"]
        p_solarize = neps_hyperparameters["p_solarize"]

        # Strengths and Thresholds
        brightness_strength = neps_hyperparameters["brightness_strength"]
        contrast_strength = neps_hyperparameters["contrast_strength"]
        saturation_strength = neps_hyperparameters["saturation_strength"]
        hue_strength = neps_hyperparameters["hue_strength"]
        solarize_threshold = neps_hyperparameters["solarize_threshold"]
    # ----------------------------------------------------------------------------------------------

    # For testing
    print(f"p_colorjitter: {p_colorjitter}")
    print(f"p_grayscale: {p_grayscale}")
    print(f"p_gaussianblur: {p_gaussianblur}")
    print(f"p_horizontal_flip: {p_horizontal_flip}")
    print(f"p_solarize: {p_solarize}")
    print(f"brightness_strength: {brightness_strength}")
    print(f"contrast_strength: {contrast_strength}")
    print(f"saturation_strength: {saturation_strength}")
    print(f"hue_strength: {hue_strength}")
    print(f"solarize_threshold: {solarize_threshold}")
    # ----------------------------------------------------------------------------------------------

    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        # No blur augmentation for CIFAR10!
        if not get_fine_tuning_loaders:
            if parameterize_augmentation:
                # rest is done outside
                train_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                )
            else:
                train_transform = TwoCropsTransform(
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(
                                size=32, scale=(0.2, 1.0), interpolation=Image.BICUBIC
                            ),
                            transforms.RandomApply(
                                [
                                    transforms.ColorJitter(
                                        brightness=brightness_strength,
                                        contrast=contrast_strength,
                                        saturation=saturation_strength,
                                        hue=hue_strength,
                                    )
                                ],
                                p=p_colorjitter,
                            ),
                            transforms.RandomGrayscale(p=p_grayscale),
                            transforms.RandomHorizontalFlip(p_horizontal_flip),
                            transforms.RandomSolarize(threshold=solarize_threshold, p=p_solarize),
                            transforms.ToTensor(),
                            normalize_cifar10 if dataset_name == "CIFAR10" else normalize_cifar100,
                        ]
                    )
                )

            valid_transform = TwoCropsTransform(
                transforms.Compose(
                    [
                        transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
                        transforms.CenterCrop(32),
                        transforms.ToTensor(),
                        normalize_cifar10 if dataset_name == "CIFAR10" else normalize_cifar100,
                    ]
                )
            )
        else:  # TODO: Check out which data augmentations are being used here!
            train_transform = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(32, scale=(0.8, 1.0),
                    #                              ratio=(3.0 / 4.0, 4.0 / 3.0),
                    #                              interpolation=Image.BICUBIC),
                    transforms.RandomResizedCrop(
                        size=32, scale=(0.2, 1.0), interpolation=Image.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_cifar10 if dataset_name == "CIFAR10" else normalize_cifar100,
                ]
            )

            valid_transform = transforms.Compose(
                [
                    transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    normalize_cifar10 if dataset_name == "CIFAR10" else normalize_cifar100,
                ]
            )

    # TODO: Why padding=4?
    # elif dataset_name == "CIFAR100":
    #     train_transform = transforms.Compose(
    #         [
    #             transforms.RandomCrop(32, padding=4),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             normalize_cifar100,
    #         ]
    #     )
    #
    #     valid_transform = transforms.Compose(
    #         [
    #             transforms.ToTensor(),
    #             normalize_cifar100,
    #         ]
    #     )

    elif dataset_name == "ImageNet":
        if not get_fine_tuning_loaders:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            if parameterize_augmentation:
                # rest is done outside
                train_transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            224, scale=(0.2, 1.0), interpolation=Image.BICUBIC
                        ),
                        transforms.ToTensor(),
                    ]
                )
            else:
                train_transform = TwoCropsTransform(
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(
                                224, scale=(0.2, 1.0), interpolation=Image.BICUBIC
                            ),
                            transforms.RandomApply(
                                [
                                    transforms.ColorJitter(
                                        brightness=brightness_strength,
                                        contrast=contrast_strength,
                                        saturation=saturation_strength,
                                        hue=hue_strength,
                                    )
                                ],
                                p=p_colorjitter,
                            ),
                            transforms.RandomGrayscale(p=p_grayscale),
                            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=p_gaussianblur),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomSolarize(threshold=solarize_threshold, p=p_solarize),
                            transforms.ToTensor(),
                            normalize_imagenet,
                        ]
                    )
                )

            valid_transform = TwoCropsTransform(
                transforms.Compose(
                    [
                        transforms.Resize(256, interpolation=Image.BICUBIC),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        normalize_imagenet,
                    ]
                )
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_imagenet,
                ]
            )
            # same as above without two crop transform
            valid_transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=Image.BICUBIC),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    normalize_imagenet,
                ]
            )
    else:
        # not supported
        raise ValueError("invalid dataset name=%s" % dataset_name)

    return train_transform, valid_transform


def get_loaders(
    config,
    parameterize_augmentation=False,
    bohb_infos=None,
    download=False,
    neps_hyperparameters=None,
    mode="alternating",
):
    if mode == "pretraining":
        train_loader_pt, _, train_sampler_pt, _ = get_train_valid_loader(
            config=config,
            neps_hyperparameters=neps_hyperparameters,
            batch_size=config.train.batch_size,
            random_seed=config.expt.seed,
            valid_size=config.finetuning.valid_size,
            dataset_name=config.data.dataset,
            shuffle=True,
            num_workers=config.expt.workers,
            pin_memory=True,
            download=download,
            distributed=config.expt.distributed,
            drop_last=True,
            get_fine_tuning_loaders=False,
            parameterize_augmentation=parameterize_augmentation,
            bohb_infos=bohb_infos,
            dataset_percentage_usage=config.data.dataset_percentage_usage,
            use_fix_aug_params=config.expt.use_fix_aug_params,
            data_augmentation_mode=config.expt.data_augmentation_mode,
            finetuning_data_augmentation=config.finetuning.data_augmentation,
        )
        train_loader_ft = None
        train_sampler_ft = None
        valid_loader_ft = None
        test_loader_ft = None

    elif mode == "finetuning":
        train_loader_pt = None
        train_sampler_pt = None
        train_loader_ft, valid_loader_ft, train_sampler_ft, _ = get_train_valid_loader(
            config=config,
            neps_hyperparameters=neps_hyperparameters,
            batch_size=config.finetuning.batch_size,
            random_seed=config.expt.seed,
            valid_size=config.finetuning.valid_size,
            dataset_name=config.data.dataset,
            shuffle=True,
            num_workers=config.expt.workers,
            pin_memory=True,
            download=download,
            distributed=config.expt.distributed,
            drop_last=True,
            get_fine_tuning_loaders=True,
            parameterize_augmentation=False,  # we never parameterize augmentations in the FT case
            bohb_infos=bohb_infos,
            dataset_percentage_usage=config.data.dataset_percentage_usage,
            use_fix_aug_params=config.expt.use_fix_aug_params,
            data_augmentation_mode=config.expt.data_augmentation_mode,
            finetuning_data_augmentation=config.finetuning.data_augmentation,
        )

        test_loader_ft = get_test_loader(
            batch_size=config.finetuning.batch_size,
            dataset_name=config.data.dataset,
            shuffle=False,
            num_workers=config.expt.workers,
            pin_memory=True,
            download=download,
            drop_last=False,
        )
    elif mode == "alternating":
        train_loader_pt, _, train_sampler_pt, _ = get_train_valid_loader(
            config=config,
            neps_hyperparameters=neps_hyperparameters,
            batch_size=config.train.batch_size,
            random_seed=config.expt.seed,
            valid_size=config.finetuning.valid_size,
            dataset_name=config.data.dataset,
            shuffle=True,
            num_workers=config.expt.workers,
            pin_memory=True,
            download=download,
            distributed=config.expt.distributed,
            drop_last=True,
            get_fine_tuning_loaders=False,
            parameterize_augmentation=parameterize_augmentation,
            bohb_infos=bohb_infos,
            dataset_percentage_usage=config.data.dataset_percentage_usage,
            use_fix_aug_params=config.expt.use_fix_aug_params,
            data_augmentation_mode=config.expt.data_augmentation_mode,
            finetuning_data_augmentation=config.finetuning.data_augmentation,
        )

        train_loader_ft, valid_loader_ft, train_sampler_ft, _ = get_train_valid_loader(
            config=config,
            neps_hyperparameters=neps_hyperparameters,
            batch_size=config.finetuning.batch_size,
            random_seed=config.expt.seed,
            valid_size=config.finetuning.valid_size,
            dataset_name=config.data.dataset,
            shuffle=True,
            num_workers=config.expt.workers,
            pin_memory=True,
            download=download,
            distributed=config.expt.distributed,
            drop_last=True,
            get_fine_tuning_loaders=True,
            parameterize_augmentation=False,  # we never parameterize augmentations in the FT case
            bohb_infos=bohb_infos,
            dataset_percentage_usage=config.data.dataset_percentage_usage,
            use_fix_aug_params=config.expt.use_fix_aug_params,
            data_augmentation_mode=config.expt.data_augmentation_mode,
            finetuning_data_augmentation=config.finetuning.data_augmentation,
        )

        test_loader_ft = get_test_loader(
            batch_size=config.finetuning.batch_size,
            dataset_name=config.data.dataset,
            shuffle=False,
            num_workers=config.expt.workers,
            pin_memory=True,
            download=download,
            drop_last=False,
        )

    else:
        raise NotImplementedError("Not implemented data loader mode!")

    if config.finetuning.valid_size > 0:
        return (
            train_loader_pt,
            train_sampler_pt,
            train_loader_ft,
            train_sampler_ft,
            valid_loader_ft,
            test_loader_ft,
        )
    else:  # TODO: @Diane - Checkout and test on *parameterized_aug*
        return (
            train_loader_pt,
            train_sampler_pt,
            train_loader_ft,
            train_sampler_ft,
            test_loader_ft,
        )
