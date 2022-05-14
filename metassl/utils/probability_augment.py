import cv2
import torchvision.transforms as transforms
from albumentations import (
    ChannelShuffle,
    ColorJitter,
    Compose,
    Cutout,
    ElasticTransform,
    Equalize,
    GaussianBlur,
    GaussNoise,
    GridDistortion,
    HorizontalFlip,
    Normalize,
    OpticalDistortion,
    RandomGridShuffle,
    RandomResizedCrop,
    Solarize,
    SomeOf,
    ToGray,
)
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
from albumentations.pytorch.transforms import ToTensorV2

normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

normalize_cifar10 = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)

normalize_cifar100 = transforms.Normalize(
    mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
)


def probability_augment(
    config,
    dataset_name,
    use_fix_aug_params,
    neps_hyperparameters=None,
):
    # --------------------------------------------------------------------------------------------
    # Specify data augmentation hyperparameters
    # --------------------------------------------------------------------------------------------
    p_color_transformations = 0.5
    p_geometric_transformations = 0.5
    p_non_rigid_transformations = 0
    p_quality_transformations = 0
    p_exotic_transformations = 0
    n_color_transformations = 1
    n_geometric_transformations = 1
    n_non_rigid_transformations = 1
    n_quality_transformations = 1
    n_exotic_transformations = 1
    n_total = 1

    if use_fix_aug_params:
        # You can overwrite parameters here if you want to try out a specific setting.
        # Due to the flag, default experiments won't be affected by this.
        p_color_transformations = 0.5
        p_geometric_transformations = 0.5
        p_non_rigid_transformations = 0
        p_quality_transformations = 0
        p_exotic_transformations = 0
        n_geometric_transformations = 1
        n_non_rigid_transformations = 1
        n_quality_transformations = 1
        n_exotic_transformations = 1
        n_total = 1

    # NEPS - probability_augment configspace
    if config.neps.is_neps_run and config.neps.config_space == "probability_augment":
        p_color_transformations = neps_hyperparameters["p_color_transformations"]
        p_geometric_transformations = neps_hyperparameters["p_geometric_transformations"]
        p_non_rigid_transformations = neps_hyperparameters["p_non_rigid_transformations"]
        p_quality_transformations = neps_hyperparameters["p_quality_transformations"]
        p_exotic_transformations = neps_hyperparameters["p_exotic_transformations"]
        n_color_transformations = neps_hyperparameters["n_color_transformations"]
        n_geometric_transformations = neps_hyperparameters["n_geometric_transformations"]
        n_non_rigid_transformations = neps_hyperparameters["n_non_rigid_transformations"]
        n_quality_transformations = neps_hyperparameters["n_quality_transformations"]
        n_exotic_transformations = neps_hyperparameters["n_exotic_transformations"]
        n_total = neps_hyperparameters["n_total"]

    # For testing
    print(f"p_color_transformations: {p_color_transformations}")
    print(f"p_geometric_transformations: {p_geometric_transformations}")
    print(f"p_non_rigid_transformations: {p_non_rigid_transformations}")
    print(f"p_quality_transformations: {p_quality_transformations}")
    print(f"p_exotic_transformations: {p_exotic_transformations}")
    print(f"n_color_transformations: {n_color_transformations}")
    print(f"n_geometric_transformations: {n_geometric_transformations}")
    print(f"n_non_rigid_transformations: {n_non_rigid_transformations}")
    print(f"n_quality_transformations: {n_quality_transformations}")
    print(f"n_exotic_transformations: {n_exotic_transformations}")
    print(f"n_total: {n_total}")
    # --------------------------------------------------------------------------------------------

    if dataset_name == "CIFAR10":
        train_transform = Compose(
            [
                # basic SimSiam transformation
                RandomResizedCrop(
                    height=32, width=32, scale=(0.2, 1.0), interpolation=cv2.INTER_CUBIC
                ),  # Very important for SimSiam!
                SomeOf(
                    [
                        # color transformations
                        SomeOf(
                            [
                                ColorJitter(
                                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1
                                ),
                                ToGray(p=1),
                                Solarize(p=1),
                                Equalize(p=1),
                                ChannelShuffle(p=1),
                                # InvertImg(p=1),
                            ],
                            n=n_color_transformations,
                            replace=False,
                            p=p_color_transformations,
                        ),
                        # geometric transformations
                        SomeOf(
                            [
                                ShiftScaleRotate(interpolation=cv2.INTER_CUBIC, p=1),
                                HorizontalFlip(p=1),
                            ],
                            n=n_geometric_transformations,
                            replace=False,
                            p=p_geometric_transformations,
                        ),
                        # non-rigid transformations
                        SomeOf(
                            [
                                ElasticTransform(
                                    alpha=0.5,
                                    sigma=10,
                                    alpha_affine=5,
                                    interpolation=cv2.INTER_CUBIC,
                                    p=1,
                                ),
                                GridDistortion(interpolation=cv2.INTER_CUBIC, p=1),
                                OpticalDistortion(
                                    distort_limit=0.5,
                                    shift_limit=0.5,
                                    interpolation=cv2.INTER_CUBIC,
                                    p=1,
                                ),
                            ],
                            n=n_non_rigid_transformations,
                            replace=False,
                            p=p_non_rigid_transformations,
                        ),
                        # quality transformations
                        SomeOf(
                            [
                                GaussianBlur(p=1),
                                GaussNoise(p=1),
                                # Downscale(p=1),
                                # Blur(p=1),
                                # GlassBlur(p=1),
                                # ImageCompression(p=1),
                                # ISONoise(p=1),
                                # JpegCompression(p=1),
                                # MultiplicativeNoise(p=1)
                            ],
                            n=n_quality_transformations,
                            replace=False,
                            p=p_quality_transformations,
                        ),
                        # exotic transformations
                        SomeOf(
                            [
                                RandomGridShuffle(p=1),
                                Cutout(num_holes=4, p=1),
                            ],
                            n=n_exotic_transformations,
                            replace=False,
                            p=p_exotic_transformations,
                        ),
                    ],
                    n=n_total,
                    replace=False,
                    p=1,
                ),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ToTensorV2(),
            ],
            p=1,
        )

        return train_transform

    else:
        raise NotImplementedError(
            "ProbabilityAugment needs to be implemented for CIFAF100 and ImageNet"
        )
