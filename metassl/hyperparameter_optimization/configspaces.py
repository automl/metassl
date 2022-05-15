import neps

from metassl.hyperparameter_optimization.hierarchical_configspaces import (
    get_hierarchical_backbone,
    get_hierarchical_predictor,
    get_hierarchical_projector,
)


def get_neps_pipeline_space(config):
    config_space = config.neps.config_space
    user_prior = config.neps.is_user_prior

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
                    lower=0,
                    upper=1,
                    log=False,
                    default=0.2,
                    default_confidence="medium",  # default as in BYOL paper
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
                    lower=0,
                    upper=255,
                    log=False,
                    default=127,
                    default_confidence="medium",  # default as in BYOL paper
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
    elif config_space == "probability_augment":
        if user_prior:
            pipeline_space = dict(
                # Probabilities
                p_color_transformations=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
                ),
                p_geometric_transformations=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0.5, default_confidence="medium"
                ),
                p_non_rigid_transformations=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0, default_confidence="medium"
                ),
                p_quality_transformations=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0, default_confidence="medium"
                ),
                p_exotic_transformations=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=0, default_confidence="medium"
                ),
                # Number of transformations per group
                n_color_transformations=neps.IntegerParameter(
                    lower=1, upper=5, log=False, default=1, default_confidence="medium"
                ),
                n_geometric_transformations=neps.IntegerParameter(
                    lower=1, upper=2, log=False, default=1, default_confidence="medium"
                ),
                n_non_rigid_transformations=neps.IntegerParameter(
                    lower=1, upper=3, log=False, default=1, default_confidence="medium"
                ),
                n_quality_transformations=neps.IntegerParameter(
                    lower=1, upper=2, log=False, default=1, default_confidence="medium"
                ),
                n_exotic_transformations=neps.IntegerParameter(
                    lower=1, upper=2, log=False, default=1, default_confidence="medium"
                ),
                n_total=neps.IntegerParameter(
                    lower=1, upper=5, log=False, default=1, default_confidence="medium"
                ),
            )
        else:
            pipeline_space = dict(
                # Probabilities
                p_color_transformations=neps.FloatParameter(lower=0, upper=1, log=False),
                p_geometric_transformations=neps.FloatParameter(lower=0, upper=1, log=False),
                p_non_rigid_transformations=neps.FloatParameter(lower=0, upper=1, log=False),
                p_quality_transformations=neps.FloatParameter(
                    lower=0,
                    upper=1,
                    log=False,
                ),
                p_exotic_transformations=neps.FloatParameter(
                    lower=0,
                    upper=1,
                    log=False,
                ),
                # Number of transformations per group
                n_color_transformations=neps.IntegerParameter(lower=1, upper=5, log=False),
                n_geometric_transformations=neps.IntegerParameter(lower=1, upper=2, log=False),
                n_non_rigid_transformations=neps.IntegerParameter(lower=1, upper=3, log=False),
                n_quality_transformations=neps.IntegerParameter(
                    lower=1,
                    upper=2,
                    log=False,
                ),
                n_exotic_transformations=neps.IntegerParameter(lower=1, upper=2, log=False),
                n_total=neps.IntegerParameter(lower=1, upper=5, log=False),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    elif config_space == "rand_augment":
        if user_prior:
            pipeline_space = dict(
                num_ops=neps.IntegerParameter(
                    lower=1, upper=15, log=False, default=3, default_confidence="medium"
                ),
                magnitude=neps.IntegerParameter(
                    lower=0, upper=30, log=False, default=15, default_confidence="medium"
                ),
            )
        else:
            pipeline_space = dict(
                num_ops=neps.IntegerParameter(lower=1, upper=15, log=False),
                magnitude=neps.IntegerParameter(lower=0, upper=30, log=False),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    elif config_space == "smart_augment":
        if user_prior:
            pipeline_space = dict(
                num_col_ops=neps.IntegerParameter(
                    lower=1, upper=9, log=False, default=2, default_confidence="medium"
                ),
                num_geo_ops=neps.IntegerParameter(
                    lower=1, upper=5, log=False, default=1, default_confidence="medium"
                ),
                col_magnitude=neps.IntegerParameter(
                    lower=0, upper=30, log=False, default=4, default_confidence="medium"
                ),
                geo_magnitude=neps.IntegerParameter(
                    lower=0, upper=30, log=False, default=4, default_confidence="medium"
                ),
                apply_ops_prob=neps.FloatParameter(
                    lower=0, upper=1, log=False, default=1, default_confidence="medium"
                ),
            )
        else:
            pipeline_space = dict(
                num_col_ops=neps.IntegerParameter(
                    lower=1,
                    upper=9,
                    log=False,
                ),
                num_geo_ops=neps.IntegerParameter(
                    lower=1,
                    upper=5,
                    log=False,
                ),
                col_magnitude=neps.IntegerParameter(
                    lower=0,
                    upper=30,
                    log=False,
                ),
                geo_magnitude=neps.IntegerParameter(
                    lower=0,
                    upper=30,
                    log=False,
                ),
                apply_ops_prob=neps.FloatParameter(
                    lower=0,
                    upper=1,
                    log=False,
                ),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    elif config_space == "hierarchical_nas":
        if config.neps.optimize_backbone_only:
            if user_prior:
                pipeline_space = dict(
                    hierarchical_backbone=get_hierarchical_backbone(user_prior=user_prior),
                )
            else:
                pipeline_space = dict(
                    hierarchical_backbone=get_hierarchical_backbone(),
                )
        else:
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
                    lower=0.003, upper=0.3, log=True, default=0.03, default_confidence="medium"
                ),
                warmup_epochs=neps.IntegerParameter(
                    lower=0, upper=80, log=False, default=0, default_confidence="medium"
                ),
                warmup_multiplier=neps.FloatParameter(
                    lower=1.0, upper=3.0, log=False, default=1.0, default_confidence="medium"
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
                pt_learning_rate=neps.FloatParameter(lower=0.003, upper=0.3, log=True),
                warmup_epochs=neps.IntegerParameter(lower=0, upper=80, log=False),
                warmup_multiplier=neps.FloatParameter(lower=1.0, upper=3.0, log=False),
                pt_optimizer=neps.CategoricalParameter(choices=["adamw", "sgd", "lars"]),
                pt_weight_decay_start=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
                pt_weight_decay_end=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    # TODO: optimize backbone-only or backbone + projector + predictor?
    elif config_space == "combined" and config.neps.optimize_backbone_only:
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
                    lower=0,
                    upper=1,
                    log=False,
                    default=0.2,
                    default_confidence="medium"
                    # default as in BYOL paper
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
                    lower=0,
                    upper=255,
                    log=False,
                    default=127,
                    default_confidence="medium",  # default as in BYOL paper
                ),
                # HIERARCHICAL NAS
                hierarchical_backbone=get_hierarchical_backbone(user_prior=user_prior),
                # TRAINING
                pt_learning_rate=neps.FloatParameter(
                    lower=0.003, upper=0.3, log=True, default=0.03, default_confidence="medium"
                ),
                warmup_epochs=neps.IntegerParameter(
                    lower=0, upper=80, log=False, default=0, default_confidence="medium"
                ),
                warmup_multiplier=neps.FloatParameter(
                    lower=1.0, upper=3.0, log=False, default=1.0, default_confidence="medium"
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
                # TRAINING
                pt_learning_rate=neps.FloatParameter(lower=0.003, upper=0.3, log=True),
                warmup_epochs=neps.IntegerParameter(lower=0, upper=80, log=False),
                warmup_multiplier=neps.FloatParameter(lower=1.0, upper=3.0, log=False),
                pt_optimizer=neps.CategoricalParameter(choices=["adamw", "sgd", "lars"]),
                pt_weight_decay_start=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
                pt_weight_decay_end=neps.FloatParameter(lower=5.0e-6, upper=5.0e-2, log=True),
            )
        return pipeline_space
    # ----------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError(
            "Config space 'combined' needs to be implemented for the 'not backbone-only' case"
        )
