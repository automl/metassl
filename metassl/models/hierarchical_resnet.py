from __future__ import annotations

import logging

import neps
from neps.search_spaces.graph_grammar import primitives as ops
from torch import nn

from metassl.hyperparameter_optimization.hierarchical_classes import (
    ResNetBasicBlockStride1,
    ResNetBasicBlockStride2,
    Sequential,
    Sequential3Edge,
    Sequential4Edge,
)

primitives = {
    "Identity": ops.Identity(),
    # "Conv3x3BNReLU": {"op": ConvBNReLU, "kernel_size": 3, "stride": 1, "padding": 1},
    # "Conv1x1BNReLU": {"op": ConvBNReLU, "kernel_size": 1},
    # "Conv3x3BN": {"op": ConvBN, "kernel_size": 3, "stride": 1, "padding": 1},
    # "Conv1x1BN": {"op": ConvBN, "kernel_size": 1},
    # "ReLU": {"op": ReLU},
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
    # "ResNetBB_stride1": {"op": ResNetBasicBlockStride1, "stride": 1},
    # "ResNetBB_stride2": {"op": ResNetBasicBlockStride2, "stride": 2},
    "Sequential": Sequential,
    "Sequential3": Sequential3Edge,
    "Sequential4": Sequential4Edge,
    # "avg_pool": {"op": ops.AvgPool1x1, "kernel_size": 3, "stride": 1},
    "downsample": {"op": ops.ResNetBasicblock, "stride": 2},
    # "residual": topos.Residual,
    # "diamond": topos.Diamond,
    # "linear": topos.Sequential,
    # "diamond_mid": topos.DiamondMid,
    # "down1": topos.Sequential,
    # # "down1": topos.DownsampleBlock,
}

# Baseline only
# -------------
# structure = {
#     "S": ["Sequential conv2 3-conv"],
#     "conv2": ["Sequential ResNetBB_stride1 ResNetBB_stride1"],
#     "3-conv": ["Sequential3 convx convx convx"],
#     "convx": ["Sequential ResNetBB_stride2 ResNetBB_stride1"],
# }

# Baseline : [2, 2, 2, 2]

# [2, 2-3, 2-3, 2-3] > [2, 3, 3, 3], [2, 3, 2, 2], ...
# [2, 2-3, 2-5, 2-3] -> 2, 3, 5, 3 (ResNet50)
# Search Space: [2, 2-3, 4-5, 2] || [2, 2-3, 2-3, 2] || [2-3, 2, 2, 2-3] || [2-3, 2, 2, 1]
# || [1, 2-3, 2-3, 2]
# (3, 4, 6, 3) in ResNet-50 to (3, 3, 9, 3),
#
structure = {
    "S": [
        "Sequential4 block-stride1 block2 block2 block2",  # baseline
        # "Sequential4 block2 block2 block2 block2",
        # "Sequential4 block-stride1 block-stride1 block4 block2",
        # "Sequential4 block-stride1 block2 block4 block2",
        # "Sequential4 block2 block2 block4 block2",
        "Sequential4 block-stride1 block1 block4 block2",
        "Sequential4 block-stride1 block2 block4 block1",
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
        "ResNetBB_BN_GELU_1",
        "ResNetBB_LN_GELU_1",
        "ResNetBB_BN_ReLU_1",
        "ResNetBB_LN_ReLU_1",
    ],
    "ResNetBB_stride2": [
        "ResNetBB_BN_GELU_2",
        "ResNetBB_LN_GELU_2",
        "ResNetBB_BN_ReLU_2",
        "ResNetBB_LN_ReLU_2",
    ],
}


def set_recursive_attribute(op_name, predecessor_values):
    in_channels = 64 if predecessor_values is None else predecessor_values["C_out"]
    out_channels = in_channels * 2 if op_name == "ResNetBasicBlockStride2" else in_channels
    return dict(C_in=in_channels, C_out=out_channels)


def run_pipeline(working_directory, architecture):
    in_channels = 3
    n_classes = 10  # For CIFAR10, TODO: generalize
    base_channels = 64
    out_channels = 512

    model = architecture.to_pytorch()
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ),
        model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(out_channels, n_classes),
    )

    print(model)
    # print(len(model.parameters()))
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    return 0


pipeline_space = dict(
    architecture=neps.FunctionParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        name="makrograph",
    )
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="experiments/hierarchical_resnet",
    max_evaluations_total=50,
    overwrite_working_directory=True,
)

previous_results, pending_configs = neps.status("experiments/hierarchical_resnet")
