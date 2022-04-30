import argparse

import jsonargparse
from jsonargparse import ArgumentParser

from metassl.utils.config import AttrDict, _parse_args


def get_parsed_config():
    config_parser = argparse.ArgumentParser(
        description="Only used as a first parser for the config file path."
    )
    config_parser.add_argument(
        "--config",
        default="metassl/default_metassl_config.yaml",
        help="Select which yaml file to use depending on the selected experiment mode",
    )
    parser = ArgumentParser()

    # ----------------------------------------------------------------------------------------------
    # EXPT
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--expt", default="expt", type=str, metavar="N")
    parser.add_argument(
        "--expt.expt_name",
        default="pre-training-fix-lr-100-256",
        type=str,
        help="experiment name",
    )
    parser.add_argument(
        "--expt.expt_mode",
        default="ImageNet",
        choices=["ImageNet", "CIFAR10"],
        help="Define which dataset to use to select the correct yaml file.",
    )
    parser.add_argument(
        "--expt.save_model",
        action="store_false",
        help="save the model to disc or not (default: True)",
    )
    parser.add_argument(
        "--expt.save_model_frequency",
        default=10,
        type=int,
        metavar="N",
        help="save model frequency in # of epochs",
    )
    parser.add_argument(
        "--expt.alternating_finetune_frequency",
        default=1,
        type=int,
        metavar="N",
        help="determines how many number of steps should be skipped before the next finetuning and "
        "aug optimizer step is invoked",
    )
    parser.add_argument(
        "--expt.ssl_model_checkpoint_path",
        type=str,
        help="path to the pre-trained model, resumes training if model with same config exists",
    )
    parser.add_argument(
        "--expt.target_model_checkpoint_path",
        type=str,
        help="path to the downstream task model, resumes training if model with same config exists",
    )
    parser.add_argument("--expt.print_freq", default=10, type=int, metavar="N")
    parser.add_argument(
        "--expt.gpu",
        default=None,
        type=int,
        metavar="N",
        help="GPU ID to train on (if not distributed)",
    )
    parser.add_argument(
        "--expt.multiprocessing_distributed",
        action="store_false",
        help="Use multi-processing distributed training to launch N processes per node, which has "
        "N GPUs. This is the fastest way to use PyTorch for either single node or multi node "
        "data parallel training (default: True)",
    )
    parser.add_argument("--expt.dist_backend", type=str, default="nccl", help="distributed backend")
    parser.add_argument(
        "--expt.dist_url",
        type=str,
        default="tcp://localhost:10005",
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--expt.workers",
        default=32,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--expt.rank",
        default=0,
        type=int,
        metavar="N",
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--expt.world_size",
        default=1,
        type=int,
        metavar="N",
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--expt.eval_freq",
        default=10,
        type=int,
        metavar="N",
        help="every eval_freq epoch will the model be evaluated",
    )
    parser.add_argument(
        "--expt.seed",
        default=123,
        type=int,
        metavar="N",
        help="random seed of numpy and torch",
    )
    parser.add_argument(
        "--expt.evaluate",
        action="store_true",
        help="evaluate model on validation set once and terminate (default: False)",
    )
    parser.add_argument(
        "--expt.is_non_grad_based",
        action="store_true",
        help="Set this flag to run default SimSiam or BOHB runs",
    )
    parser.add_argument(
        "--expt.warmup_epochs",
        default=10,
        type=int,
        metavar="N",
        help="denotes the number of epochs that we only pre-train without finetuning afterwards; "
        "warmup is turned off when set to 0; we use a linear incremental schedule during "
        "warmup",
    )
    parser.add_argument(
        "--expt.warmup_multiplier",
        default=1.0,
        type=float,
        metavar="N",
        help="A factor that is multiplied with the pretraining lr used in the linear incremental "
        "learning rate scheduler during warmup. The final lr is multiplier * pre-training lr",
    )
    parser.add_argument(
        "--expt.use_fix_aug_params",
        action="store_true",
        help="Use this flag if you want to try out specific aug params (e.g., from a best BOHB "
        "config). Default values will be overwritten then without crashing other experiments.",
    )
    parser.add_argument(
        "--expt.data_augmentation_mode",
        default="default",
        choices=["default", "probability_augment", "rand_augment"],
        help="Select which data augmentation to use. Default is for the standard SimSiam setting "
        "and for parameterize aug setting.",
    )
    parser.add_argument(
        "--expt.write_summary_frequency",
        default=10,
        type=int,
        metavar="N",
        help="Specifies, after how many batches the TensorBoard summary writer should flush new "
        "data to the summary object.",
    )
    parser.add_argument(
        "--expt.wd_decay_pt",
        action="store_true",
        help="use weight decay decay (annealing) during pre-training? (default: False)",
    )
    parser.add_argument(
        "--expt.wd_decay_ft",
        action="store_true",
        help="use weight decay decay (annealing) during fine-tuning? (default: False)",
    )
    parser.add_argument(
        "--expt.run_knn_val",
        action="store_true",
        help="activate knn evaluation during training (default: False)",
    )
    parser.add_argument(
        "--expt.is_testing_mode",
        action="store_true",
        help="Set this flag to enter the test mode to test the code quickly (default: False)",
    )

    # ----------------------------------------------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--train", default="train", type=str, metavar="N")
    parser.add_argument(
        "--train.batch_size",
        default=256,
        type=int,
        metavar="N",
        help="in distributed setting this is the total batch size, i.e. batch size = individual bs "
        "* number of GPUs",
    )
    parser.add_argument(
        "--train.epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of pre-training epochs",
    )
    parser.add_argument(
        "--train.start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start training at epoch n",
    )
    parser.add_argument(
        "--train.optimizer",
        type=str,
        default="sgd",
        help="optimizer type, options: sgd",
    )
    parser.add_argument(
        "--train.schedule",
        type=str,
        default="cosine",
        help="learning rate schedule, not implemented",
    )
    parser.add_argument("--train.weight_decay", default=0.0001, type=float, metavar="N")
    parser.add_argument(
        "--train.momentum", default=0.9, type=float, metavar="N", help="SGD momentum"
    )
    parser.add_argument(
        "--train.lr",
        default=0.05,
        type=float,
        metavar="N",
        help="pre-training learning rate",
    )
    parser.add_argument(
        "--train.wd_start",
        default=1e-3,
        type=float,
        help="Upper value of WD Decay. Only used when wd_decay is True.",
    )
    parser.add_argument(
        "--train.wd_end",
        default=1e-6,
        type=float,
        help="Lower value of WD Decay. Only used when wd_decay is True.",
    )

    # ----------------------------------------------------------------------------------------------
    # FINETUNING
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--finetuning", default="finetuning", type=str, metavar="N")
    parser.add_argument(
        "--finetuning.batch_size",
        default=256,
        type=int,
        metavar="N",
        help="in distributed setting this is the total batch size, i.e. batch size = individual bs "
        "* number of GPUs",
    )
    parser.add_argument(
        "--finetuning.epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of pre-training epochs",
    )
    parser.add_argument(
        "--finetuning.start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="start training at epoch n",
    )
    parser.add_argument(
        "--finetuning.optimizer",
        type=str,
        default="sgd",
        help="optimizer type, options: sgd",
    )
    parser.add_argument(
        "--finetuning.schedule",
        type=str,
        default="cosine",
        help="learning rate schedule, not implemented",
    )
    parser.add_argument("--finetuning.weight_decay", default=0.0, type=float, metavar="N")
    parser.add_argument(
        "--finetuning.momentum",
        default=0.9,
        type=float,
        metavar="N",
        help="SGD momentum",
    )
    parser.add_argument(
        "--finetuning.lr",
        default=100,
        type=float,
        metavar="N",
        help="finetuning learning rate",
    )
    parser.add_argument(
        "--finetuning.valid_size",
        default=0.0,
        type=float,
        help="If valid_size > 0, pick some images from the trainset to do evaluation on. "
        "If valid_size=0 evaluation is done on the testset.",
    )
    parser.add_argument(
        "--finetuning.data_augmentation",
        default="none",
        choices=[
            "none",
            "p_probability_augment_pt",
            "p_probability_augment_ft",
            "p_probability_augment_1-pt",
        ],
        help="Select if and how finetuning gets augmented.",
    )
    parser.add_argument(
        "--finetuning.wd_start",
        default=1e-3,
        type=float,
        help="Upper value of WD Decay. Only used when wd_decay is True.",
    )
    parser.add_argument(
        "--finetuning.wd_end",
        default=1e-6,
        type=float,
        help="Lower value of WD Decay. Only used when wd_decay is True.",
    )
    parser.add_argument(
        "--finetuning.use_alternative_scheduler",
        action="store_true",
        help="Use the learning rate scheduler from the baseline codebase",
    )

    # ----------------------------------------------------------------------------------------------
    # MODEL
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--model", default="model", type=str, metavar="N")
    parser.add_argument(
        "--model.model_type",
        type=str,
        default="resnet50",
        help="all torchvision ResNets",
    )
    parser.add_argument("--model.seed", type=int, default=123, help="the seed")
    parser.add_argument(
        "--model.turn_off_bn",
        action="store_true",
        help="turns off all batch norm instances in the model",
    )

    # ----------------------------------------------------------------------------------------------
    # DATA
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--data", default="data", type=str, metavar="N")
    parser.add_argument("--data.seed", type=int, default=123, help="the seed")
    parser.add_argument(
        "--data.dataset",
        type=str,
        default="ImageNet",
        help="supported datasets: CIFAR10, CIFAR100, ImageNet",
    )
    parser.add_argument(
        "--data.dataset_percentage_usage",
        type=float,
        default=100.0,
        help="Indicates what percentage of the data is used for the experiments.",
    )

    # ----------------------------------------------------------------------------------------------
    # SIMSIAM
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--simsiam", default="simsiam", type=str, metavar="N")
    parser.add_argument("--simsiam.dim", type=int, default=2048, help="the feature dimension")
    parser.add_argument(
        "--simsiam.pred_dim",
        type=int,
        default=512,
        help="the hidden dimension of the predictor",
    )
    parser.add_argument(
        "--simsiam.fix_pred_lr",
        action="store_false",
        help="fix learning rate for the predictor (default: True",
    )

    # ----------------------------------------------------------------------------------------------
    # LEARN AUG
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--learnaug", default="learnaug", type=str, metavar="N")
    parser.add_argument(
        "--learnaug.type",
        default="default",
        choices=["colorjitter", "default", "full_net"],
        help="Define which type of learned augmentation to use.",
    )

    # ----------------------------------------------------------------------------------------------
    # BOHB
    # ----------------------------------------------------------------------------------------------
    # parser.add_argument('--bohb', default="bohb", type=str, metavar='N')
    # parser.add_argument("--bohb.run_id", default="default_BOHB")
    # parser.add_argument("--bohb.seed", type=int, default=123, help="random seed")
    # parser.add_argument("--bohb.n_iterations", type=int, default=10,
    #                     help="How many BOHB iterations")
    # parser.add_argument("--bohb.min_budget", type=int, default=2)
    # parser.add_argument("--bohb.max_budget", type=int, default=4)
    # parser.add_argument("--bohb.budget_mode", type=str, default="epochs",
    #                     choices=["epochs", "data"], help="Choose your desired fidelity")
    # parser.add_argument("--bohb.eta", type=int, default=2)
    # parser.add_argument("--bohb.configspace_mode", type=str, default='color_jitter_strengths',
    #                     choices=["imagenet_probability_simsiam_augment",
    #                              "cifar10_probability_simsiam_augment", "color_jitter_strengths",
    #                              "rand_augment", "probability_augment",
    #                              "double_probability_augment"],
    #                     help='Define which configspace to use.')
    # parser.add_argument("--bohb.nic_name", default="lo",
    #                     help="The network interface to use")  # local: "lo", cluster: "eth0"
    # parser.add_argument("--bohb.port", type=int, default=0)
    # parser.add_argument("--bohb.worker", action="store_true",
    #                     help="Make this execution a worker server")
    # parser.add_argument("--bohb.warmstarting", type=bool, default=False)
    # parser.add_argument("--bohb.warmstarting_dir", type=str, default=None)
    # parser.add_argument("--bohb.test_env", action='store_true',
    #                     help='If using this flag, the master runs a worker in the background and '
    #                          'workers are not being shutdown after registering results.')

    # ----------------------------------------------------------------------------------------------
    # NEPS
    # ----------------------------------------------------------------------------------------------
    parser.add_argument("--neps", default="neps", type=str, metavar="NEPS")
    parser.add_argument(
        "--neps.is_neps_run",
        action="store_true",
        help="Set this flag to run a NEPS experiment.",
    )
    parser.add_argument(
        "--neps.config_space",
        type=str,
        default="hierarchical_nas",
        choices=[
            "data_augmentation",
            "hierarchical_nas",
            "training",
            "combined",
        ],
        help="Define which configspace to use.",
    )
    parser.add_argument(
        "--neps.is_user_prior",
        action="store_true",
        help="Set this flag to run a NEPS experiment with user prior.",
    )

    # ----------------------------------------------------------------------------------------------
    # USE FIXED ARGS
    # ----------------------------------------------------------------------------------------------
    parser.add_argument(
        "--use_fixed_args",
        action="store_true",
        help="To control whether to take arguments from yaml file as default or from arg parse",
    )

    config = _parse_args(config_parser, parser)
    config = AttrDict(jsonargparse.namespace_to_dict(config))
    return config
