from neps.search_spaces.graph_grammar.primitives import AbstractPrimitive
from neps.search_spaces.graph_grammar.topologies import AbstractTopology
from torch import nn


class Identity(AbstractPrimitive):
    """
    An implementation of the Identity operation.
    """

    def __init__(self, **kwargs):
        super().__init__(locals())

    def forward(self, x, edge_data):
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


class Linear3Edge(AbstractTopology):
    edge_list: list = [(1, 2), (2, 3), (3, 4)]

    def __init__(self, *edge_vals):
        super().__init__()
        number_of_edges = 3
        self.name = f"Linear_{number_of_edges}_Edges"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)


class Linear4Edge(AbstractTopology):
    edge_list: list = [(1, 2), (2, 3), (3, 4), (4, 5)]

    def __init__(self, *edge_vals):
        super().__init__()
        number_of_edges = 4
        self.name = f"Linear_{number_of_edges}_Edges"
        self.create_graph(dict(zip(self.edge_list, edge_vals)))
        self.set_scope(self.name)
