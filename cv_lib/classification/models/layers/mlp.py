import torch.nn as nn
from torch import Tensor

from . import get_activation_fn, get_dropout


class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self,
        embed_dim: int,
        dim_feedforward: int,
        dropout: float = None,
        activation: str = "relu"
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, embed_dim)
        self.drop1 = get_dropout(dropout)
        self.drop2 = get_dropout(dropout)
        self.activation = get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, 1e-6)
        nn.init.normal_(self.fc2.bias, 1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.activation(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x
