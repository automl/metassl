from __future__ import annotations

import logging

import neps
from neps.search_spaces.graph_grammar import primitives as ops
from neps.search_spaces.graph_grammar import topologies as topos
from torch import nn

from metassl.hyperparameter_optimization.hierarchical_classes import (
    ConvBN,
    ConvBNReLU,
    Linear3Edge,
    ReLU,
    ResNetBasicBlockStride1,
    ResNetBasicBlockStride2,
)

primitives = {
    # "Identity": ops.Identity(),
    "Conv3x3BNReLU": {"op": ConvBNReLU, "kernel_size": 3, "stride": 1, "padding": 1},
    "Conv1x1BNReLU": {"op": ConvBNReLU, "kernel_size": 1},
    "Conv3x3BN": {"op": ConvBN, "kernel_size": 3, "stride": 1, "padding": 1},
    "Conv1x1BN": {"op": ConvBN, "kernel_size": 1},
    "ReLU": {"op": ReLU},
    "ResNetBasicBlock_stride1": {"op": ResNetBasicBlockStride1, "stride": 1},
    "ResNetBasicBlock_stride2": {"op": ResNetBasicBlockStride2, "stride": 2},
    "Linear": topos.Linear,
    "Linear3": Linear3Edge,
    # "avg_pool": {"op": ops.AvgPool1x1, "kernel_size": 3, "stride": 1},
    "downsample": {"op": ops.ResNetBasicblock, "stride": 2},
    # "residual": topos.Residual,
    # "diamond": topos.Diamond,
    # "linear": topos.Linear,
    # "diamond_mid": topos.DiamondMid,
    # "down1": topos.Linear,
    # # "down1": topos.DownsampleBlock,
}

structure = {
    "S": ["Linear conv2 3-conv"],
    "conv2": ["Linear ResNetBasicBlock_stride1 ResNetBasicBlock_stride1"],
    "3-conv": ["Linear3 convx convx convx"],
    "convx": ["Linear ResNetBasicBlock_stride2 ResNetBasicBlock_stride1"],
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
    max_evaluations_total=1,
    overwrite_working_directory=True,
)

previous_results, pending_configs = neps.status("experiments/hierarchical_resnet")
