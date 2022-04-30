import neps

from metassl.hyperparameter_optimization.hierarchical_configspaces import (
    get_hierarchical_backbone,
    get_hierarchical_predictor,
    get_hierarchical_projector,
)


def get_neps_pipeline_space(config_space, user_prior):
    if config_space == "data_augmentation":
        if user_prior:
            pipeline_space = dict(
                # Probabilities
                p_colorjitter=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.8, default_confidence="medium"
                ),
                p_grayscale=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.2, default_confidence="medium"
                ),
                p_horizontal_flip=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
                ),
                p_solarize=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0, default_confidence="low"
                ),
                # Strengths and Thresholds
                brightness_strength=neps.FloatParameter(
                    lower=0, upper=1.5, log=False, default=0.4, default_confidence="medium"
                ),
                contrast_strength=neps.FloatParameter(
                    lower=0, upper=1.5, log=False, default=0.4, default_confidence="medium"
                ),
                saturation_strength=neps.FloatParameter(
                    lower=0, upper=1.5, log=False, default=0.4, default_confidence="medium"
                ),
                hue_strength=neps.FloatParameter(
                    lower=0, upper=0.5, log=False, default=0.1, default_confidence="medium"
                ),
                solarize_threshold=neps.IntegerParameter(
                    lower=0, upper=255, log=False, default=255, default_confidence="low"
                ),
            )
        else:
            pipeline_space = dict(
                # Probabilities
                p_colorjitter=neps.FloatParameter(lower=0, upper=1, log=False),
                p_grayscale=neps.FloatParameter(lower=0, upper=1, log=False),
                p_horizontal_flip=neps.FloatParameter(lower=0, upper=1, log=False),
                p_solarize=neps.FloatParameter(lower=0, upper=1, log=False),
                # Strengths and Thresholds
                brightness_strength=neps.FloatParameter(lower=0, upper=1.5, log=False),
                contrast_strength=neps.FloatParameter(lower=0, upper=1.5, log=False),
                saturation_strength=neps.FloatParameter(lower=0, upper=1.5, log=False),
                hue_strength=neps.FloatParameter(lower=0, upper=0.5, log=False),
                solarize_threshold=neps.IntegerParameter(lower=0, upper=255, log=False),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    elif config_space == "hierarchical_nas":
        if user_prior:
            pipeline_space = dict(
                hierarchical_backbone=get_hierarchical_backbone(user_prior=user_prior),
                hierarchical_projector=get_hierarchical_projector(
                    prev_dim=512, user_prior=user_prior
                ),
                hierarchical_predictor=get_hierarchical_predictor(
                    prev_dim=512, user_prior=user_prior
                ),
            )
        else:
            pipeline_space = dict(
                hierarchical_backbone=get_hierarchical_backbone(),
                hierarchical_projector=get_hierarchical_projector(prev_dim=512),
                hierarchical_predictor=get_hierarchical_predictor(prev_dim=512),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    elif config_space == "training":
        if user_prior:
            pipeline_space = dict(
                pt_learning_rate=neps.FloatParameter(
                    lower=0.01, upper=10, log=True, default=0.03, default_confidence="medium"
                ),
                warmup_epochs=neps.IntegerParameter(
                    lower=5, upper=50, log=False, default=0, default_confidence="medium"
                ),
                warmup_multiplier=neps.IntegerParameter(
                    lower=0, upper=5, log=False, default=0, default_confidence="medium"
                ),
                pt_optimizer=neps.CategoricalParameter(
                    choices=["adamw", "sgd", "lars"], default="sgd", default_confidence="medium"
                ),
                pt_weight_decay_start=neps.FloatParameter(
                    lower=5.0e-6,
                    upper=5.0e-2,
                    log=True,
                    default=5.0e-4,
                    default_confidence="medium",
                ),
                pt_weight_decay_end=neps.FloatParameter(
                    lower=5.0e-6,
                    upper=5.0e-2,
                    log=True,
                    default=5.0e-4,
                    default_confidence="medium",
                ),
            )
        else:
            pipeline_space = dict(
                pt_learning_rate=neps.FloatParameter(lower=0.01, upper=10, log=True),
                warmup_epochs=neps.IntegerParameter(lower=5, upper=50, log=False),
                warmup_multiplier=neps.IntegerParameter(lower=0, upper=5, log=False),
                pt_optimizer=neps.CategoricalParameter(choices=["adamw", "sgd", "lars"]),
                pt_weight_decay_start=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
                pt_weight_decay_end=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    elif config_space == "combined":
        if user_prior:
            pipeline_space = dict(
                # DATA AUGMENTATION
                # Probabilities
                p_colorjitter=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.8, default_confidence="medium"
                ),
                p_grayscale=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.2, default_confidence="medium"
                ),
                p_horizontal_flip=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
                ),
                p_solarize=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0, default_confidence="low"
                ),
                # Strengths and Thresholds
                brightness_strength=neps.FloatParameter(
                    lower=0, upper=1.5, log=False, default=0.4, default_confidence="medium"
                ),
                contrast_strength=neps.FloatParameter(
                    lower=0, upper=1.5, log=False, default=0.4, default_confidence="medium"
                ),
                saturation_strength=neps.FloatParameter(
                    lower=0, upper=1.5, log=False, default=0.4, default_confidence="medium"
                ),
                hue_strength=neps.FloatParameter(
                    lower=0, upper=0.5, log=False, default=0.1, default_confidence="medium"
                ),
                solarize_threshold=neps.IntegerParameter(
                    lower=0, upper=255, log=False, default=255, default_confidence="low"
                ),
                # HIERARCHICAL NAS
                hierarchical_backbone=get_hierarchical_backbone(user_prior=user_prior),
                hierarchical_projector=get_hierarchical_projector(
                    prev_dim=512, user_prior=user_prior
                ),
                hierarchical_predictor=get_hierarchical_predictor(
                    prev_dim=512, user_prior=user_prior
                ),
                # TRAINING
                pt_learning_rate=neps.FloatParameter(
                    lower=0.01, upper=10, log=True, default=0.03, default_confidence="medium"
                ),
                warmup_epochs=neps.IntegerParameter(
                    lower=5, upper=50, log=False, default=0, default_confidence="medium"
                ),
                warmup_multiplier=neps.IntegerParameter(
                    lower=0, upper=5, log=False, default=0, default_confidence="medium"
                ),
                pt_optimizer=neps.CategoricalParameter(
                    choices=["adamw", "sgd", "lars"], default="sgd", default_confidence="medium"
                ),
                pt_weight_decay_start=neps.FloatParameter(
                    lower=5.0e-6,
                    upper=5.0e-2,
                    log=True,
                    default=5.0e-4,
                    default_confidence="medium",
                ),
                pt_weight_decay_end=neps.FloatParameter(
                    lower=5.0e-6,
                    upper=5.0e-2,
                    log=True,
                    default=5.0e-4,
                    default_confidence="medium",
                ),
            )
        else:
            pipeline_space = dict(
                # DATA AUGMENTATION
                # Probabilities
                p_colorjitter=neps.FloatParameter(lower=0, upper=1, log=False),
                p_grayscale=neps.FloatParameter(lower=0, upper=1, log=False),
                p_horizontal_flip=neps.FloatParameter(lower=0, upper=1, log=False),
                p_solarize=neps.FloatParameter(lower=0, upper=1, log=False),
                # Strengths and Thresholds
                brightness_strength=neps.FloatParameter(lower=0, upper=1.5, log=False),
                contrast_strength=neps.FloatParameter(lower=0, upper=1.5, log=False),
                saturation_strength=neps.FloatParameter(lower=0, upper=1.5, log=False),
                hue_strength=neps.FloatParameter(lower=0, upper=0.5, log=False),
                solarize_threshold=neps.IntegerParameter(lower=0, upper=255, log=False),
                # HIERARCHICAL NAS
                hierarchical_backbone=get_hierarchical_backbone(),
                hierarchical_projector=get_hierarchical_projector(prev_dim=512),
                hierarchical_predictor=get_hierarchical_predictor(prev_dim=512),
                # TRAINING
                pt_learning_rate=neps.FloatParameter(lower=0.01, upper=10, log=True),
                warmup_epochs=neps.IntegerParameter(lower=5, upper=50, log=False),
                warmup_multiplier=neps.IntegerParameter(lower=0, upper=5, log=False),
                pt_optimizer=neps.CategoricalParameter(choices=["adamw", "sgd", "lars"]),
                pt_weight_decay_start=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
                pt_weight_decay_end=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError
