import neps
import numpy as np
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
        "Identity":           {
            "op": Identity
            },
        "ResNetBB_BN_GELU_1": {
            "op":         ResNetBasicBlockStride1,
            "norm":       "BatchNorm",
            "activation": "GELU",
            "stride":     1,
            },
        "ResNetBB_LN_GELU_1": {
            "op":         ResNetBasicBlockStride1,
            "norm":       "LayerNorm",
            "activation": "GELU",
            "stride":     1,
            },
        "ResNetBB_BN_ReLU_1": {
            "op":         ResNetBasicBlockStride1,
            "norm":       "BatchNorm",
            "activation": "ReLU",
            "stride":     1,
            },
        "ResNetBB_LN_ReLU_1": {
            "op":         ResNetBasicBlockStride1,
            "norm":       "LayerNorm",
            "activation": "ReLU",
            "stride":     1,
            },
        "ResNetBB_BN_GELU_2": {
            "op":         ResNetBasicBlockStride2,
            "norm":       "BatchNorm",
            "activation": "GELU",
            "stride":     2,
            },
        "ResNetBB_LN_GELU_2": {
            "op":         ResNetBasicBlockStride2,
            "norm":       "LayerNorm",
            "activation": "GELU",
            "stride":     2,
            },
        "ResNetBB_BN_ReLU_2": {
            "op":         ResNetBasicBlockStride2,
            "norm":       "BatchNorm",
            "activation": "ReLU",
            "stride":     2,
            },
        "ResNetBB_LN_ReLU_2": {
            "op":         ResNetBasicBlockStride2,
            "norm":       "LayerNorm",
            "activation": "ReLU",
            "stride":     2,
            },
        "Sequential":         Sequential,
        "Sequential3":        Sequential3Edge,
        "Sequential4":        Sequential4Edge,
        }

    structure = {
        "S":                  [
            # LayerNorm
            "Sequential4 block-stride1LN block2LN block2LN block2LN",
            "Sequential4 block-stride1LN block1LN block4LN block2LN",
            "Sequential4 block-stride1LN block2LN block4LN block1LN",
            # BatchNorm
            "Sequential4 block-stride1BN block2BN block2BN block2BN",  # baseline
            "Sequential4 block-stride1BN block1BN block4BN block2BN",
            "Sequential4 block-stride1BN block2BN block4BN block1BN",
            # Not used for the moment (maybe later for rebuttal?)
            # "Sequential4 block2 block2 block2 block2",
            # "Sequential4 block-stride1 block-stride1 block4 block2",
            # "Sequential4 block-stride1 block2 block4 block2",
            # "Sequential4 block2 block2 block4 block2",
            ],
        "block-stride1BN":    [
            "Sequential ResNetBB_stride1BN ResNetBB_stride1BN",
            ],
        "block1BN":           [
            "Sequential ResNetBB_stride1BN Identity",
            ],
        "block2BN":           [
            "Sequential ResNetBB_stride2BN ResNetBB_stride1BN",
            ],
        "block4BN":           [
            "Sequential4 ResNetBB_stride2BN ResNetBB_stride1BN ResNetBB_stride2BN "
            "ResNetBB_stride1BN",
            ],

        "block-stride1LN":    [
            "Sequential ResNetBB_stride1LN ResNetBB_stride1LN",
            ],
        "block1LN":           [
            "Sequential ResNetBB_stride1LN Identity",
            ],
        "block2LN":           [
            "Sequential ResNetBB_stride2LN ResNetBB_stride1LN",
            ],
        "block4LN":           [
            "Sequential4 ResNetBB_stride2LN ResNetBB_stride1LN ResNetBB_stride2LN "
            "ResNetBB_stride1LN",
            ],

        "ResNetBB_stride1LN": [
            "ResNetBB_LN_GELU_1",
            "ResNetBB_LN_ReLU_1",
            ],
        "ResNetBB_stride2LN": [
            "ResNetBB_LN_GELU_2",
            "ResNetBB_LN_ReLU_2",
            ],

        "ResNetBB_stride1BN": [
            "ResNetBB_BN_GELU_1",
            "ResNetBB_BN_ReLU_1",  # baseline
            ],
        "ResNetBB_stride2BN": [
            "ResNetBB_BN_GELU_2",
            "ResNetBB_BN_ReLU_2",  # baseline
            ],
        }

    # TODO: @Diane please double check :)
    prior_distr = {
        # "S":                  [0.5, 0.25, 0.25],
        "S":                  [0.3, 0.1, 0.1, 0.3, 0.1, 0.1],
        "block-stride1BN":    [1.0],
        "block-stride1LN":    [1.0],
        "block1BN":           [1.0],
        "block1LN":           [1.0],
        "block2BN":           [1.0],
        "block2LN":           [1.0],
        "block4BN":           [1.0],
        "block4LN":           [1.0],
        "ResNetBB_stride1BN": [0.25, 0.25, 0.25, 0.25],
        "ResNetBB_stride1LN": [0.25, 0.25, 0.25, 0.25],
        "ResNetBB_stride2BN": [0.25, 0.25, 0.25, 0.25],
        "ResNetBB_stride2LN": [0.25, 0.25, 0.25, 0.25],
        }

    assert all(np.isclose(sum(v), 1.) for v in prior_distr.values()), "propabilities should sum to 1"

    def set_recursive_attribute(op_name, predecessor_values):
        in_channels = 64 if predecessor_values is None else predecessor_values["C_out"]
        out_channels = in_channels * 2 if op_name == "ResNetBasicBlockStride2" else in_channels
        return dict(C_in=in_channels, C_out=out_channels)

    # Generated hierarchical_predictor
    hierarchical_backbone = neps.FunctionParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        prior=prior_distr if user_prior else None,
        name="hierarchical_backbone",
        )

    return hierarchical_backbone


def get_hierarchical_projector(prev_dim, user_prior=None):  # encoder head
    primitives = {
        "Identity":       {
            "op": Identity
            },
        "FullyConnected": {
            "op":       FullyConnected,
            "prev_dim": prev_dim
            },
        "ReLU":           {
            "op": ReLU
            },
        "LeakyReLU":      {
            "op": LeakyReLU
            },
        "GELU":           {
            "op": GELU
            },
        "BatchNorm":      {
            "op":       BatchNorm,
            "prev_dim": prev_dim
            },
        "LayerNorm":      {
            "op":           LayerNorm,
            "num_features": prev_dim
            },
        "residual":       topos.Residual,
        "diamond":        topos.Diamond,
        "linear":         topos.Linear,
        "diamond_mid":    topos.DiamondMid,
        "linear3":        Sequential3Edge,
        "linear4":        Sequential4Edge,
        }

    structure = {
        "S":          [
            # BatchNorm
            "linear blockBN blockBN",  # baseline
            "linear3 blockBN blockBN blockBN",
            "linear4 blockBN blockBN blockBN blockBN",
            "diamond blockBN blockBN blockBN blockBN",
            # LayerNorm
            "linear blockLN blockLN",
            "linear3 blockLN blockLN blockLN",
            "linear4 blockLN blockLN blockLN blockLN",
            "diamond blockLN blockLN blockLN blockLN",
            ],
        "blockBN":    [
            "linear3 transform normBN activation",  # baseline
            "linear transform normBN",
            "linear transform activation",
            "transform",
            "neutral",
            ],
        "blockLN":    [
            "linear3 transform normLN activation",
            "linear transform normLN",
            "linear transform activation",
            "transform",
            "neutral",
            ],
        "transform":  ["FullyConnected"],
        "activation": ["ReLU", "GELU"],
        "normBN":     ["BatchNorm"],
        "normLN":     ["LayerNorm"],
        "neutral":    ["Identity"],
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

    # TODO: @Diane please double check
    prior_distr = {
        "S":          [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1],  # baseline is weighted with 0.4
        "blockBN":    [0.4, 0.15, 0.15, 0.15, 0.15],  # baseline is weighted with 0.4
        "blockLN":    [0.4, 0.15, 0.15, 0.15, 0.15],
        "transform":  [1.0],
        "activation": [0.5, 0.5],
        "normBN":     [1.0],
        "normLN":     [1.0],
        "neutral":    [1.0],
        }

    assert all(np.isclose(sum(v), 1.) for v in prior_distr.values()), "propabilities should sum to 1"

    # Generated hierarchical_projector
    hierarchical_projector = neps.FunctionParameter(
        set_recursive_attribute=None,
        structure=structure,
        primitives=primitives,
        prior=prior_distr if user_prior else None,
        name="hierarchical_projector",
        )

    return hierarchical_projector


def get_hierarchical_predictor(prev_dim, user_prior=None):
    primitives = {
        "Identity":       {
            "op": Identity
            },
        "FullyConnected": {
            "op":       FullyConnected,
            "prev_dim": prev_dim
            },
        "ReLU":           {
            "op": ReLU
            },
        "LeakyReLU":      {
            "op": LeakyReLU
            },
        "GELU":           {
            "op": GELU
            },
        "BatchNorm":      {
            "op":       BatchNorm,
            "prev_dim": prev_dim
            },
        "LayerNorm":      {
            "op":           LayerNorm,
            "num_features": prev_dim
            },
        "residual":       topos.Residual,
        "diamond":        topos.Diamond,
        "linear":         topos.Linear,
        "diamond_mid":    topos.DiamondMid,
        "linear3":        Sequential3Edge,
        "linear4":        Sequential4Edge,
        }

    structure = {
        "S":              [
            "linear finish-blockBN S2BN",
            "linear finish-blockLN S2LN"
            ],
        "finish-blockBN": [
            "linear normBN activation",
            "linear normBN neutral",
            "linear activation neutral",
            "linear neutral neutral",
            ],
        "finish-blockLN": [
            "linear normLN activation",
            "linear normLN neutral",
            "linear activation neutral",
            "linear neutral neutral",
            ],
        "S2BN":           [
            "linear blockBN blockBN",
            "linear3 blockBN blockBN blockBN",
            "linear4 blockBN blockBN blockBN blockBN",
            "diamond blockBN blockBN blockBN blockBN",
            ],
        "S2LN":           [
            "linear blockLN blockLN",
            "linear3 blockLN blockLN blockLN",
            "linear4 blockLN blockLN blockLN blockLN",
            "diamond blockLN blockLN blockLN blockLN",
            ],
        "blockBN":        [
            "linear3 transform normBN activation",
            "linear transform normBN",
            "linear transform activation",
            "transform",
            "neutral",
            ],
        "blockLN":        [
            "linear3 transform normLN activation",
            "linear transform normLN",
            "linear transform activation",
            "transform",
            "neutral",
            ],
        "transform":      ["FullyConnected"],
        "activation":     ["ReLU", "GELU"],
        "normBN":         ["BatchNorm"],
        "normLN":         ["LayerNorm"],
        "neutral":        ["Identity"],
        }

    # Default predictor:
    # Sequential(
    # (0): Linear(in_features=2048, out_features=512, bias=False)  > FIX
    # (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (2): ReLU(inplace=True)
    # (3): Linear(in_features=512, out_features=2048, bias=True)   > FIX
    # )

    # TODO: @Diane please double check
    prior_distr = {
        "S":              [0.5, 0.5],
        "finish-blockBN": [0.4, 0.2, 0.2, 0.2],  # baseline is weighted with 0.4
        "finish-blockLN": [0.4, 0.2, 0.2, 0.2],
        "S2BN":           [0.4, 0.2, 0.2, 0.2],  # baseline is weighted with 0.4
        "S2LN":           [0.4, 0.2, 0.2, 0.2],
        "blockBN":        [0.15, 0.15, 0.15, 0.15, 0.4],  # baseline is weighted with 0.4
        "blockLN":        [0.15, 0.15, 0.15, 0.15, 0.4],
        "transform":      [1.0],
        "activation":     [0.5, 0.5],
        "normBN":         [1.0],
        "normLN":         [1.0],
        "neutral":        [1.0],
        }

    assert all(np.isclose(sum(v), 1.) for v in prior_distr.values()), "propabilities should sum to 1"

    # Generated hierarchical_predictor
    hierarchical_predictor = neps.FunctionParameter(
        set_recursive_attribute=None,
        structure=structure,
        primitives=primitives,
        prior=prior_distr if user_prior else None,
        name="hierarchical_predictor",
        )

    return hierarchical_predictor
