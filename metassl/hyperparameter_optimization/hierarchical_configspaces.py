import neps
from neps.search_spaces.graph_grammar import topologies as topos

from metassl.hyperparameter_optimization.hierarchical_classes import (
    GELU,
    BatchNorm,
    FullyConnected,
    Identity,
    LayerNorm,
    LeakyReLU,
    ReLU,
    ResNetBasicBlockStride1,
    ResNetBasicBlockStride2,
    Sequential,
    Sequential3Edge,
    Sequential4Edge,
)


def get_hierarchical_backbone(user_prior=None):  # ResNet18
    primitives = {
        "Identity": {"op": Identity},
        "ResNetBB_BN_GELU_1": {
            "op": ResNetBasicBlockStride1,
            "norm": "BatchNorm",
            "activation": "GELU",
            "stride": 1,
        },
        "ResNetBB_LN_GELU_1": {
            "op": ResNetBasicBlockStride1,
            "norm": "LayerNorm",
            "activation": "GELU",
            "stride": 1,
        },
        "ResNetBB_BN_ReLU_1": {
            "op": ResNetBasicBlockStride1,
            "norm": "BatchNorm",
            "activation": "ReLU",
            "stride": 1,
        },
        "ResNetBB_LN_ReLU_1": {
            "op": ResNetBasicBlockStride1,
            "norm": "LayerNorm",
            "activation": "ReLU",
            "stride": 1,
        },
        "ResNetBB_BN_GELU_2": {
            "op": ResNetBasicBlockStride2,
            "norm": "BatchNorm",
            "activation": "GELU",
            "stride": 2,
        },
        "ResNetBB_LN_GELU_2": {
            "op": ResNetBasicBlockStride2,
            "norm": "LayerNorm",
            "activation": "GELU",
            "stride": 2,
        },
        "ResNetBB_BN_ReLU_2": {
            "op": ResNetBasicBlockStride2,
            "norm": "BatchNorm",
            "activation": "ReLU",
            "stride": 2,
        },
        "ResNetBB_LN_ReLU_2": {
            "op": ResNetBasicBlockStride2,
            "norm": "LayerNorm",
            "activation": "ReLU",
            "stride": 2,
        },
        "Sequential": Sequential,
        "Sequential3": Sequential3Edge,
        "Sequential4": Sequential4Edge,
    }

    structure = {
        "S": [
            "Sequential4 block-stride1 block2 block2 block2",  # baseline
            # "Sequential4 block-stride1 block1 block4 block2",  # TODO: RuntimeError: CUDA out of memory. > Check on the cluster  # noqa: E501
            "Sequential4 block-stride1 block2 block4 block1",
            # Not used for the moment (maybe later for rebuttal?)
            # "Sequential4 block2 block2 block2 block2",
            # "Sequential4 block-stride1 block-stride1 block4 block2",
            # "Sequential4 block-stride1 block2 block4 block2",
            # "Sequential4 block2 block2 block4 block2",
        ],
        "block-stride1": [
            "Sequential ResNetBB_stride1 ResNetBB_stride1",
        ],
        "block1": [
            "Sequential ResNetBB_stride1 Identity",
        ],
        "block2": [
            "Sequential ResNetBB_stride2 ResNetBB_stride1",
        ],
        "block4": [
            "Sequential4 ResNetBB_stride2 ResNetBB_stride1 ResNetBB_stride2 ResNetBB_stride1",
        ],
        "ResNetBB_stride1": [
            # "ResNetBB_BN_GELU_1",    # TODO: RuntimeError: CUDA out of memory; BUT: works sometimes > Check on the cluster  # noqa: E501
            # "ResNetBB_LN_GELU_1",  # TODO: RuntimeError: Given normalized_shape=[128], expected input with shape [*, 128], but got input of size[512, 128, 16, 16]  # noqa: E501
            "ResNetBB_BN_ReLU_1",  # baseline
            # "ResNetBB_LN_ReLU_1",  # TODO: RuntimeError: Given normalized_shape=[64], expected input with shape [*, 64], but got input of size[512, 64, 32, 32]  # noqa: E501
        ],
        "ResNetBB_stride2": [
            # "ResNetBB_BN_GELU_2",  # TODO: RuntimeError: CUDA out of memory; BUT: works sometimes > Check on the cluster  # noqa: E501
            # "ResNetBB_LN_GELU_2",  # TODO: RuntimeError: Given normalized_shape=[128], expected input with shape [*, 128], but got input of size[512, 128, 16, 16]  # noqa: E501
            "ResNetBB_BN_ReLU_2",  # baseline
            # "ResNetBB_LN_ReLU_2",  # TODO: RuntimeError: Given normalized_shape=[128], expected input with shape [*, 128], but got input of size[512, 128, 16, 16]  # noqa: E501
        ],
    }

    # TODO: integrate user prior
    # prior_distr = {
    #     "S": [0.5, 0.25, 0.25],  # TODO: give probability for each element in S
    #     "ResNetBB_stride1": [1 / 4 for _ in range(4)],
    #     # TODO: Do for all groups
    # }

    def set_recursive_attribute(op_name, predecessor_values):
        in_channels = 64 if predecessor_values is None else predecessor_values["C_out"]
        out_channels = in_channels * 2 if op_name == "ResNetBasicBlockStride2" else in_channels
        return dict(C_in=in_channels, C_out=out_channels)

    # Generated hierarchical_predictor
    hierarchical_backbone = neps.FunctionParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        # TODO: prior=prior_distr if config.neps.is_user_prior else None,
        name="hierarchical_backbone",
    )

    return hierarchical_backbone


def get_hierarchical_projector(prev_dim, user_prior=None):  # encoder head
    primitives = {
        "Identity": {"op": Identity},
        "FullyConnected": {"op": FullyConnected, "prev_dim": prev_dim},
        "ReLU": {"op": ReLU},
        "LeakyReLU": {"op": LeakyReLU},
        "GELU": {"op": GELU},
        "BatchNorm": {"op": BatchNorm, "prev_dim": prev_dim},
        "LayerNorm": {"op": LayerNorm, "prev_dim": prev_dim},
        "residual": topos.Residual,
        "diamond": topos.Diamond,
        "linear": topos.Linear,
        "diamond_mid": topos.DiamondMid,
        "linear3": Sequential3Edge,
        "linear4": Sequential4Edge,
    }

    structure = {
        "S": [
            "linear block block",
            "linear3 block block block",
            "linear4 block block block block",
            "diamond block block block block",
        ],
        "block": [
            "linear3 transform activation norm",
            "linear3 transform norm activation",
            "linear transform norm",
            "linear transform activation",
            "transform",
            "neutral",
        ],
        "transform": ["FullyConnected"],
        "activation": ["ReLU", "LeakyReLU", "GELU"],
        "norm": ["BatchNorm", "LayerNorm"],
        "neutral": ["Identity"],
    }

    # Default projector
    # Sequential(
    # (0): Linear(in_features=512, out_features=512, bias=False)
    # (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (2): ReLU(inplace=True)
    # (3): Linear(in_features=512, out_features=512, bias=False)
    # (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (5): ReLU(inplace=True)
    # (6): Linear(in_features=512, out_features=2048, bias=True)                               > FIX
    # (7): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)  > FIX
    # )

    # TODO: integrate user prior

    # Generated hierarchical_projector
    hierarchical_projector = neps.FunctionParameter(
        set_recursive_attribute=None,
        structure=structure,
        primitives=primitives,
        # TODO: prior=prior_distr if config.neps.is_user_prior else None,
        name="hierarchical_projector",
    )

    return hierarchical_projector


def get_hierarchical_predictor(prev_dim, user_prior=None):
    primitives = {
        "Identity": {"op": Identity},
        "FullyConnected": {"op": FullyConnected, "prev_dim": prev_dim},
        "ReLU": {"op": ReLU},
        "LeakyReLU": {"op": LeakyReLU},
        "GELU": {"op": GELU},
        "BatchNorm": {"op": BatchNorm, "prev_dim": prev_dim},
        "LayerNorm": {"op": LayerNorm, "prev_dim": prev_dim},
        "residual": topos.Residual,
        "diamond": topos.Diamond,
        "linear": topos.Linear,
        "diamond_mid": topos.DiamondMid,
        "linear3": Sequential3Edge,
        "linear4": Sequential4Edge,
    }

    structure = {
        "S": ["linear finish-block S2"],
        "finish-block": [
            "linear norm activation",
            "linear activation norm",
            "linear norm neutral",
            "linear activation neutral",
            "linear neutral neutral",
        ],
        "S2": [
            "linear block block",
            "linear3 block block block",
            "linear4 block block block block",
            "diamond block block block block",
        ],
        "block": [
            "linear3 transform activation norm",
            "linear3 transform norm activation",
            "linear transform norm",
            "linear transform activation",
            "transform",
            "neutral",
        ],
        "transform": ["FullyConnected"],
        "activation": ["ReLU", "LeakyReLU", "GELU"],
        "norm": ["BatchNorm", "LayerNorm"],
        "neutral": ["Identity"],
    }

    # Default predictor:
    # Sequential(
    # (0): Linear(in_features=2048, out_features=512, bias=False)  > FIX
    # (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (2): ReLU(inplace=True)
    # (3): Linear(in_features=512, out_features=2048, bias=True)   > FIX
    # )

    # TODO: integrate user prior

    # Generated hierarchical_predictor
    hierarchical_predictor = neps.FunctionParameter(
        set_recursive_attribute=None,
        structure=structure,
        primitives=primitives,
        # TODO: prior=prior_distr if config.neps.is_user_prior else None,
        name="hierarchical_predictor",
    )

    return hierarchical_predictor
