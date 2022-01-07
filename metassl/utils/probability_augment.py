
import torchvision.transforms as transforms
from albumentations.pytorch.transforms import ToTensorV2

from .simsiam import TwoCropsTransform

from albumentations import GaussNoise,ElasticTransform, \
    GridDistortion, OpticalDistortion, Normalize, RandomGridShuffle, RandomShadow, Blur, ColorJitter, Downscale, \
    Equalize, ChannelShuffle, GaussianBlur, GlassBlur, ImageCompression, InvertImg, ISONoise, JpegCompression, \
    MultiplicativeNoise
from albumentations.augmentations.geometric.transforms import Affine
from albumentations import Compose, OneOf


def probability_augment(
    dataset_name,
    get_fine_tuning_loaders,
    bohb_infos,
    use_fix_aug_params,
    finetuning_data_augmentation,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Specify data augmentation hyperparameters
    # ------------------------------------------------------------------------------------------------------------------
    p_color_transformations = 0.25
    p_geometric_transformations = 0.25
    p_non_rigid_transformations = 0.25
    p_quality_transformations = 0.25
    p_exotic_transformations = 0

    if use_fix_aug_params:
        # You can overwrite parameters here if you want to try out a specific setting.
        # Due to the flag, default experiments won't be affected by this.
        p_color_transformations = 0.25
        p_geometric_transformations = 0.25
        p_non_rigid_transformations = 0.25
        p_quality_transformations = 0.25
        p_exotic_transformations = 0

    # BOHB - probability_augment configspace
    if bohb_infos is not None:
        if bohb_infos['bohb_configspace'] == 'probability_augment' or bohb_infos['bohb_configspace'] == 'double_probability_augment':
            p_color_transformations = bohb_infos['bohb_config']['p_color_transformations']
            p_geometric_transformations = bohb_infos['bohb_config']['p_geometric_transformations']
            p_non_rigid_transformations = bohb_infos['bohb_config']['p_non_rigid_transformations']
            p_quality_transformations = bohb_infos['bohb_config']['p_quality_transformations']
            p_exotic_transformations = bohb_infos['bohb_config']['p_exotic_transformations']
        else:
            raise ValueError("Select 'probability_augment' configspace or 'double_probability_augment 'configspace if this is a BOHB run with 'probability_augment' data_augmentation_mode!")

    # For testing
    print(f"{p_color_transformations=}")
    print(f"{p_geometric_transformations=}")
    print(f"{p_non_rigid_transformations=}")
    print(f"{p_quality_transformations=}")
    print(f"{p_exotic_transformations=}")
    # ------------------------------------------------------------------------------------------------------------------

    # TODO: @Diane - Think about the seleted ops for: color, exotic, quality
    if dataset_name == "CIFAR10":
        if not get_fine_tuning_loaders:
            train_transform = Compose([
                    # color transformations
                    OneOf([
                        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1),
                        Equalize(p=1),
                        ChannelShuffle(p=1),
                        InvertImg(p=1)
                    ], p=p_color_transformations),
                    # geometric transformations
                    # TODO: @Diane - Checkout Affine
                    Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False, always_apply=False, p=p_geometric_transformations),
                    # non-rigid transformations
                    OneOf([
                        ElasticTransform(p=1),
                        GridDistortion(p=1),
                        OpticalDistortion(p=1),
                    ], p=p_non_rigid_transformations),
                    # quality transformations
                    OneOf([
                        Blur(p=1),
                        Downscale(p=1),
                        GaussianBlur(p=1),
                        GaussNoise(p=1),
                        GlassBlur(p=1),
                        ImageCompression(p=1),
                        ISONoise(p=1),
                        JpegCompression(p=1),
                        MultiplicativeNoise(p=1)
                    ], p=p_quality_transformations),
                    # exotic transformations
                    OneOf([
                        RandomGridShuffle(p=1),
                        # RandomShadow(p=1),
                    ], p=p_exotic_transformations),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ToTensorV2(),
                ], p=1)

            valid_transform = TwoCropsTransform(
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ]
                )
            )
        else:
            print(f"\n\n\nFINETUNING")
            print(f"{finetuning_data_augmentation=}")
            print(f"{bohb_infos['bohb_configspace']=}")
            # TODO @Diane - Use better variable names
            a = (finetuning_data_augmentation == 'p_probability_augment_pt' or finetuning_data_augmentation == 'p_probability_augment_1-pt') and bohb_infos['bohb_configspace'] == 'probability_augment'
            b = finetuning_data_augmentation == 'p_probability_augment_ft' and bohb_infos['bohb_configspace'] == 'double_probability_augment'
            if finetuning_data_augmentation == 'none':
                print("CASE: DEFAULT")
                train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                    ]
                )
            elif a or b:
                # [None, 'p_probability_augment_pt', 'p_probability_augment_ft', 'p_probability_augment_1-pt']
                # TODO: @Diane - Think for a better implementation design
                if finetuning_data_augmentation == 'p_probability_augment_pt':
                    print("CASE: PT")
                    p_color_transformations = bohb_infos['bohb_config']['p_color_transformations']
                    p_geometric_transformations = bohb_infos['bohb_config']['p_geometric_transformations']
                    p_non_rigid_transformations = bohb_infos['bohb_config']['p_non_rigid_transformations']
                    p_quality_transformations = bohb_infos['bohb_config']['p_quality_transformations']
                    p_exotic_transformations = bohb_infos['bohb_config']['p_exotic_transformations']
                elif finetuning_data_augmentation == 'p_probability_augment_ft':
                    print("CASE: FT")
                    p_color_transformations = bohb_infos['bohb_config']['p_color_transformations_ft']
                    p_geometric_transformations = bohb_infos['bohb_config']['p_geometric_transformations_ft']
                    p_non_rigid_transformations = bohb_infos['bohb_config']['p_non_rigid_transformations_ft']
                    p_quality_transformations = bohb_infos['bohb_config']['p_quality_transformations_ft']
                    p_exotic_transformations = bohb_infos['bohb_config']['p_exotic_transformations_ft']
                elif finetuning_data_augmentation == 'p_probability_augment_1-pt':
                    print("CASE: 1-PT")
                    p_color_transformations = 1 - bohb_infos['bohb_config']['p_color_transformations']
                    p_geometric_transformations = 1 - bohb_infos['bohb_config']['p_geometric_transformations']
                    p_non_rigid_transformations = 1 - bohb_infos['bohb_config']['p_non_rigid_transformations']
                    p_quality_transformations = 1 - bohb_infos['bohb_config']['p_quality_transformations']
                    p_exotic_transformations = 1 - bohb_infos['bohb_config']['p_exotic_transformations']
                print(f"{p_color_transformations=}")
                print(f"{p_geometric_transformations=}")
                print(f"{p_non_rigid_transformations=}")
                print(f"{p_quality_transformations=}")
                print(f"{p_exotic_transformations=}")
                # ------------------------------------------------------------------------------------------------------
                train_transform = Compose([
                    # color transformations
                    OneOf([
                        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=1),
                        Equalize(p=1),
                        ChannelShuffle(p=1),
                        InvertImg(p=1)
                    ], p=p_color_transformations),
                    # geometric transformations
                    # TODO: @Diane - Checkout Affine
                    Affine(scale=None, translate_percent=None, translate_px=None, rotate=None, shear=None,
                           interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                           always_apply=False, p=p_geometric_transformations),
                    # non-rigid transformations
                    OneOf([
                        ElasticTransform(p=1),
                        GridDistortion(p=1),
                        OpticalDistortion(p=1),
                    ], p=p_non_rigid_transformations),
                    # quality transformations
                    OneOf([
                        Blur(p=1),
                        Downscale(p=1),
                        GaussianBlur(p=1),
                        GaussNoise(p=1),
                        GlassBlur(p=1),
                        ImageCompression(p=1),
                        ISONoise(p=1),
                        JpegCompression(p=1),
                        MultiplicativeNoise(p=1)
                    ], p=p_quality_transformations),
                    # exotic transformations
                    OneOf([
                        RandomGridShuffle(p=1),
                        # RandomShadow(p=1),
                    ], p=p_exotic_transformations),
                    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                    ToTensorV2(),
                ], p=1)
            else:
                # TODO: @Diane - Implement case: no BOHB run but finetuning uses data augmentation
                raise ValueError("Not implemented yet.")
            valid_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ]
            )
    return train_transform, valid_transform