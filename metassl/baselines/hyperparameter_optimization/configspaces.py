import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

def get_cifar10_probability_simsiam_augment_configspace():
    cs = CS.ConfigurationSpace()
    p_colorjitter = CSH.UniformFloatHyperparameter(
        "p_colorjitter", lower=0, upper=1, log=False, default_value=0.8,
    )
    p_grayscale = CSH.UniformFloatHyperparameter(
        "p_grayscale", lower=0, upper=1, log=False, default_value=0.2,
    )
    cs.add_hyperparameters([p_colorjitter, p_grayscale])
    return cs

def get_color_jitter_strengths_configspace():
    cs = CS.ConfigurationSpace()
    brightness_strength = CSH.UniformFloatHyperparameter(
        "brightness_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    contrast_strength = CSH.UniformFloatHyperparameter(
        "contrast_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    saturation_strength = CSH.UniformFloatHyperparameter(
        "saturation_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    hue_strength = CSH.UniformFloatHyperparameter(
        "hue_strength", lower=0, upper=0.4, log=False, default_value=0.1,
    )
    cs.add_hyperparameters([brightness_strength, contrast_strength, saturation_strength, hue_strength])
    return cs

def get_lr_color_jitter_strengths_configspace():
    cs = CS.ConfigurationSpace()
    brightness_strength = CSH.UniformFloatHyperparameter(
        "brightness_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    contrast_strength = CSH.UniformFloatHyperparameter(
        "contrast_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    saturation_strength = CSH.UniformFloatHyperparameter(
        "saturation_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    hue_strength = CSH.UniformFloatHyperparameter(
        "hue_strength", lower=0, upper=0.4, log=False, default_value=0.1,
    )
    pt_learning_rate = CSH.UniformFloatHyperparameter(
        "pt_learning_rate", lower=0.01, upper=0.1, log=True, default_value=0.06
    )
    ft_learning_rate = CSH.UniformFloatHyperparameter(
        "ft_learning_rate", lower=10, upper=50, log=True, default_value=30
    )
    cs.add_hyperparameters([brightness_strength, contrast_strength, saturation_strength, hue_strength, pt_learning_rate, ft_learning_rate])
    return cs

def get_double_color_jitter_strengths_configspace():
    cs = CS.ConfigurationSpace()
    brightness_strength = CSH.UniformFloatHyperparameter(
        "brightness_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    contrast_strength = CSH.UniformFloatHyperparameter(
        "contrast_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    saturation_strength = CSH.UniformFloatHyperparameter(
        "saturation_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    hue_strength = CSH.UniformFloatHyperparameter(
        "hue_strength", lower=0, upper=0.4, log=False, default_value=0.1,
    )
    ft_brightness_strength = CSH.UniformFloatHyperparameter(
        "ft_brightness_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    ft_contrast_strength = CSH.UniformFloatHyperparameter(
        "ft_contrast_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    ft_saturation_strength = CSH.UniformFloatHyperparameter(
        "ft_saturation_strength", lower=0, upper=1.2, log=False, default_value=0.4,
    )
    ft_hue_strength = CSH.UniformFloatHyperparameter(
        "ft_hue_strength", lower=0, upper=0.4, log=False, default_value=0.1,
    )
    cs.add_hyperparameters([brightness_strength, contrast_strength, saturation_strength, hue_strength, ft_brightness_strength, ft_contrast_strength, ft_saturation_strength, ft_hue_strength])
    return cs

def get_rand_augment_configspace():
    cs = CS.ConfigurationSpace()
    num_ops = CSH.UniformIntegerHyperparameter(
        "num_ops", lower=1, upper=3, log=False, default_value=2,
    )
    magnitude = CSH.UniformIntegerHyperparameter(
        "magnitude", lower=0, upper=30, log=False, default_value=15,
    )
    cs.add_hyperparameters([num_ops, magnitude])
    return cs


def get_probability_augment_configspace():
    cs = CS.ConfigurationSpace()
    p_color_transformations = CSH.UniformFloatHyperparameter(
        "p_color_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_geometric_transformations = CSH.UniformFloatHyperparameter(
        "p_geometric_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_non_rigid_transformations = CSH.UniformFloatHyperparameter(
        "p_non_rigid_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_quality_transformations = CSH.UniformFloatHyperparameter(
        "p_quality_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_exotic_transformations = CSH.UniformFloatHyperparameter(
        "p_exotic_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    cs.add_hyperparameters(
        [p_color_transformations, p_geometric_transformations, p_non_rigid_transformations, p_quality_transformations, p_exotic_transformations]

    )
    return cs

def get_double_probability_augment_configspace():
    cs = CS.ConfigurationSpace()
    p_color_transformations = CSH.UniformFloatHyperparameter(
        "p_color_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_geometric_transformations = CSH.UniformFloatHyperparameter(
        "p_geometric_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_non_rigid_transformations = CSH.UniformFloatHyperparameter(
        "p_non_rigid_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_quality_transformations = CSH.UniformFloatHyperparameter(
        "p_quality_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_exotic_transformations = CSH.UniformFloatHyperparameter(
        "p_exotic_transformations", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_color_transformations_ft = CSH.UniformFloatHyperparameter(
        "p_color_transformations_ft", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_geometric_transformations_ft = CSH.UniformFloatHyperparameter(
        "p_geometric_transformations_ft", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_non_rigid_transformations_ft = CSH.UniformFloatHyperparameter(
        "p_non_rigid_transformations_ft", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_quality_transformations_ft = CSH.UniformFloatHyperparameter(
        "p_quality_transformations_ft", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    p_exotic_transformations_ft = CSH.UniformFloatHyperparameter(
        "p_exotic_transformations_ft", lower=0.0, upper=1.0, log=False, default_value=0.5,
    )
    cs.add_hyperparameters(
        [p_color_transformations, p_geometric_transformations, p_non_rigid_transformations, p_quality_transformations, p_exotic_transformations, p_color_transformations_ft, p_geometric_transformations_ft, p_non_rigid_transformations_ft, p_quality_transformations_ft, p_exotic_transformations_ft]

    )
    return cs


def get_weight_decay_annealing_configspace():
    cs = CS.ConfigurationSpace()
    weight_decay_value_1 = CSH.UniformFloatHyperparameter(
        "weight_decay_value_1", lower=1e-8, upper=10, log=True, default_value=0.5,
    )
    weight_decay_value_2 = CSH.UniformFloatHyperparameter(
        "weight_decay_value_2", lower=1e-8, upper=10, log=True, default_value=0.5,
    )
    direction = CSH.CategoricalHyperparameter(
        "direction", choices=["low-to-high", "high-to-low"],
    )

    cs.add_hyperparameters(
        [weight_decay_value_1, weight_decay_value_2, direction]

    )
    return cs