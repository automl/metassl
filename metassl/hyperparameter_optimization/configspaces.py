import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_test_configspace():
    cs = CS.ConfigurationSpace()
    val_metric = CSH.UniformIntegerHyperparameter(
        "val_metric", lower=1, upper=30, log=False, default_value=15,
    )
    cs.add_hyperparameters([val_metric])
    return cs


def get_data_augmentation_configspace():
    # TODO: Think for an interesting configspace for optimizing data augmentations
    cs = CS.ConfigurationSpace()
    num_ops = CSH.UniformIntegerHyperparameter(
        "num_ops", lower=1, upper=16, log=False, default_value=5,
    )
    cs.add_hyperparameters([num_ops])

    return cs
