import torch
import torch.nn.functional as F
from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive
from neps.search_spaces.graph_grammar.topologies import AbstractTopology
from torch import nn


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x, edge_data=None):
        return x

    def get_embedded_ops(self):
        return None


class FullyConnected(AbstractPrimitive):
    """
    Implementation of 2d convolution, followed by 2d batch normalization and ReLU activation.
    """

    def __init__(self, prev_dim, **kwargs):
        super().__init__(locals())
        self.op = nn.Linear(prev_dim, prev_dim, bias=False)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "FullyConnected"
        return op_name


class ConvNormActivation(AbstractPrimitive):
    def __init__(self, C_in, C_out, kernel_size, norm, activation, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        kernel_size = int(kernel_size)
        stride = int(stride)

        allowed_norms = ["BatchNorm", "LayerNorm", "None"]
        allowed_activations = ["ReLU", "GELU", "None"]

        # print("\n\n\n", norm)

        if norm == "BatchNorm":
            norm = nn.BatchNorm2d(C_out, affine=affine)
        elif norm == "LayerNorm":
            norm = LayerNorm2D(C_out, data_format="channels_first")
        elif norm == "None":
            norm = nn.Identity()
        else:
            raise ValueError(f"norm must be in {allowed_norms}")

        if activation == "ReLU":
            activation = nn.ReLU(inplace=True)
        elif activation == "GELU":
            activation = nn.GELU()
        elif activation == "None":
            activation = nn.Identity()
        else:
            raise ValueError(f"activation must be in {allowed_activations}")

        self.kernel_size = kernel_size
        pad = 0 if int(stride) == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            norm,
            activation,
        )

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"Conv{self.kernel_size}x{self.kernel_size}BNReLU"
        return op_name


class ConvNorm(AbstractPrimitive):
    def __init__(self, C_in, C_out, kernel_size, norm, stride=1, affine=True, **kwargs):
        super().__init__(locals())
        kernel_size = int(kernel_size)
        stride = int(stride)

        allowed_norms = ["BatchNorm", "LayerNorm", "None"]

        if norm == "BatchNorm":
            norm = nn.BatchNorm2d(C_out, affine=affine)
        elif norm == "LayerNorm":
            norm = LayerNorm2D(C_out, data_format="channels_first")
        elif norm == "None":
            norm = nn.Identity()
        else:
            raise ValueError(f"norm must be in {allowed_norms}")

        self.kernel_size = kernel_size
        pad = 0 if int(stride) == 1 and kernel_size == 1 else 1
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=pad, bias=False),
            norm,
        )

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = super().get_op_name
        op_name += f"Conv{self.kernel_size}x{self.kernel_size}BNReLU"
        return op_name


# class ResNetBasicBlockConvBNReLU(AbstractPrimitive):
#     def __init__(self, C_in, C_out, stride, affine=True, **kwargs):  # pylint:disable=W0613
#         super().__init__(locals())
#         assert stride == 1 or stride == 2, f"invalid stride {stride}"
#         self.conv_a = ConvBNReLU(C_in, C_out, 3, stride)
#         self.conv_b = ConvBN(C_out, C_out, 3)
#         if stride == 2:
#             self.downsample = nn.Sequential(
#                 # nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
#                 nn.Conv2d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False),
#                 nn.BatchNorm2d(C_out),
#             )
#         else:
#             self.downsample = None
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, edge_data=None):  # pylint: disable=W0613
#         basicblock = self.conv_a(x, None)
#         basicblock = self.conv_b(basicblock, None)
#         residual = self.downsample(x) if self.downsample is not None else x
#         return self.relu(residual + basicblock)
#
#     @staticmethod
#     def get_embedded_ops():
#         return None


class ResNetBasicBlockStride1(AbstractPrimitive):
    def __init__(
        self, C_in, C_out, norm="LayerNorm", activation="GELU", stride=2, affine=True, **kwargs
    ):  # pylint:disable=W0613
        super().__init__(locals())
        assert stride == 1 or stride == 2, f"invalid stride {stride}"
        # if stride == 2:
        #     C_out *= 2

        self.conv_a = ConvNormActivation(C_in, C_out, 3, norm, activation, stride)
        self.conv_b = ConvNorm(C_out, C_out, 3, norm)

        allowed_norms = ["BatchNorm", "LayerNorm", "None"]

        if norm == "BatchNorm":
            norm = nn.BatchNorm2d(C_out, affine=affine)
        elif norm == "LayerNorm":
            norm = LayerNorm2D(C_out, data_format="channels_first")
        elif norm == "None":
            norm = nn.Identity()
        else:
            raise ValueError(f"norm must be in {allowed_norms}")

        if stride == 2:
            self.downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False),
                norm,
            )
        else:
            self.downsample = None

    def forward(self, x, edge_data=None):  # pylint: disable=W0613
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock

    @staticmethod
    def get_embedded_ops():
        return None


class ResNetBasicBlockStride2(AbstractPrimitive):
    def __init__(
        self, C_in, C_out, norm="LayerNorm", activation="GELU", stride=2, affine=True, **kwargs
    ):  # pylint:disable=W0613
        super().__init__(locals())
        assert stride == 1 or stride == 2, f"invalid stride {stride}"
        # if stride == 2:
        #     C_out *= 2

        self.conv_a = ConvNormActivation(C_in, C_out, 3, norm, activation, stride)
        self.conv_b = ConvNorm(C_out, C_out, 3, norm)

        allowed_norms = ["BatchNorm", "LayerNorm", "None"]

        if norm == "BatchNorm":
            norm = nn.BatchNorm2d(C_out, affine=affine)
        elif norm == "LayerNorm":
            norm = LayerNorm2D(C_out, data_format="channels_first")
        elif norm == "None":
            norm = nn.Identity()
        else:
            raise ValueError(f"norm must be in {allowed_norms}")

        if stride == 2:
            self.downsample = nn.Sequential(
                # nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=2, padding=0, bias=False),
                norm,
            )
        else:
            self.downsample = None

    def forward(self, x, edge_data=None):  # pylint: disable=W0613
        basicblock = self.conv_a(x, None)
        basicblock = self.conv_b(basicblock, None)
        residual = self.downsample(x) if self.downsample is not None else x
        return residual + basicblock

    @staticmethod
    def get_embedded_ops():
        return None


class ReLU(AbstractPrimitive):
    """
    Implementation of a ReLU activation.
    """

    def __init__(self, **kwargs):  # TODO: check wheter **kwargs are needed or not
        super().__init__(locals())
        self.op = nn.ReLU(inplace=True)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "ReLU"
        return op_name


class LeakyReLU(AbstractPrimitive):
    """
    Implementation of a ReLU activation.
    """

    def __init__(self, **kwargs):  # TODO: check wheter **kwargs are needed or not
        super().__init__(locals())
        self.op = nn.LeakyReLU(inplace=True)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "LeakyReLU"
        return op_name


class GELU(AbstractPrimitive):
    """
    Implementation of a ReLU activation.
    """

    def __init__(self, **kwargs):  # TODO: check wheter **kwargs are needed or not
        super().__init__(locals())
        self.op = nn.GELU()

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "GELU"
        return op_name


class BatchNorm(AbstractPrimitive):
    """
    Implementation of a 2d batch normalization.
    """

    def __init__(self, prev_dim, affine=True, **kwargs):
        super().__init__(locals())
        self.op = nn.BatchNorm1d(prev_dim, affine=affine)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "BatchNorm"
        return op_name


class BatchNorm2D(AbstractPrimitive):
    """
    Implementation of a 2d batch normalization.
    """

    def __init__(self, prev_dim, affine=True, **kwargs):
        super().__init__(locals())
        self.op = nn.BatchNorm2d(prev_dim, affine=affine)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "BatchNorm2D"
        return op_name


# class LayerNorm(AbstractPrimitive):
#     """
#     Implementation of a 2d batch normalization.
#     """
#
#     def __init__(self, prev_dim, affine=True, **kwargs):
#         super().__init__(locals())
#         self.op = nn.LayerNorm(prev_dim)
#
#     def forward(self, x, edge_data=None):
#         return self.op(x)
#
#     def get_embedded_ops(self):
#         return None
#
#     @property
#     def get_op_name(self):
#         op_name = "LayerNorm"
#         return op_name


class LayerNorm(AbstractPrimitive):
    r"""LayerNorm1d"""

    def __init__(self, prev_dim):
        # super().__init__()
        super().__init__(locals())
        self.op = nn.LayerNorm(prev_dim)

    def forward(self, x, edge_data=None):
        return self.op(x)

    def get_embedded_ops(self):
        return None

    @property
    def get_op_name(self):
        op_name = "LayerNorm"
        return op_name


class LayerNorm2D(AbstractPrimitive):
    r"""LayerNorm that supports input from Conv2D, i.e. two data formats: channels_last (default)
    or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to
     inputs with shape (batch_size, height, width, channels) while channels_first corresponds to
     inputs with shape (batch_size, channels, height, width).
    Taken from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_features, eps=1e-6, data_format="channels_last", **kwargs):
        # super().__init__()
        super().__init__(locals())
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (num_features,)

    def forward(self, x, edge_data=None):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise NotImplementedError

    @property
    def get_op_name(self):
        op_name = "LayerNorm2D"
        return op_name

    def get_embedded_ops(self):
        return None


class Sequential(AbstractTopology):
    edge_list = [
        (1, 2),
        (2, 3),
    ]

    def __init__(self, *edge_vals):
        super().__init__()

        self.name = "Sequential"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Sequential3Edge(AbstractTopology):
    edge_list: list = [(1, 2), (2, 3), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()
        number_of_edges = 3
        self.name = f"Sequential_{number_of_edges}_Edges"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Sequential4Edge(AbstractTopology):
    edge_list: list = [(1, 2), (2, 3), (3, 4), (4, 5)]

    def __init__(self, *edge_vals):
        super().__init__()
        number_of_edges = 4
        self.name = f"Sequential_{number_of_edges}_Edges"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Sequential5Edge(AbstractTopology):
    edge_list: list = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    def __init__(self, *edge_vals):
        super().__init__()
        number_of_edges = 5
        self.name = f"Sequential_{number_of_edges}_Edges"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)
