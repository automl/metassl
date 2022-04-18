import neps
from neps.search_spaces.graph_grammar import topologies as topos

from metassl.hyperparameter_optimization.hierarchical_classes import (
    GELU,
    BatchNorm,
    FullyConnected,
    Identity,
    LayerNorm,
    LeakyReLU,
    Linear3Edge,
    Linear4Edge,
    ReLU,
)


def get_hierarchical_projector(prev_dim):  # encoder head
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
        "linear3": Linear3Edge,
        "linear4": Linear4Edge,
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

    # Generated hierarchical_projector
    hierarchical_projector = neps.FunctionParameter(
        set_recursive_attribute=None,
        structure=structure,
        primitives=primitives,
        name="hierarchical_projector",
    )

    return hierarchical_projector


def get_hierarchical_predictor(prev_dim):
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
        "linear3": Linear3Edge,
        "linear4": Linear4Edge,
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

    # Generated hierarchical_predictor
    hierarchical_predictor = neps.FunctionParameter(
        set_recursive_attribute=None,
        structure=structure,
        primitives=primitives,
        name="hierarchical_predictor",
    )

    return hierarchical_predictor
