from ..modeling import InfiniTensorModel, DTYPE
import numpy as np


class Linear(InfiniTensorModel):
    """
    Linear layer follows the formula Y = XW^T + b, where W is weight of shape (out_features, in_features), b is
    optional bias of shape (out_features,). Input X should have shape (..., in_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DTYPE = DTYPE.F32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        shape = (out_features, in_features)
        self.weight = self.parameter(shape, dtype, "weight")
        self.use_bias = bias
        if self.use_bias:
            self.bias = self.parameter((out_features,), dtype, "bias")

    def forward(self, input):
        output = self.matmul(input, self.weight, transB=1)
        if self.use_bias:
            output = self.add(output, self.bias)
        return output
