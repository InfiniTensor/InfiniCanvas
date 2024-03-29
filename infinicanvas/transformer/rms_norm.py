from ..modeling import InfiniTensorModel, DTYPE
import numpy as np


class RMSNorm(InfiniTensorModel):
    def __init__(self, hidden_size, eps: float = 1e-6, dtype=DTYPE.F32, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = self.parameter((self.hidden_size,), DTYPE.F32, "weight")

    def forward(self, hidden_states):
        hidden_states = self.cast(hidden_states, DTYPE.F32)
        hidden_states = self.rms_norm(hidden_states, self.weight, self.eps)
        hidden_states = self.cast(hidden_states, self.dtype)
        return hidden_states
