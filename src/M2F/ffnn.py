#    ___________________________________
#  /                                     \
# |   Currently unfinished interfaces     |
#  \ ___________________________________ /

from torch.nn import Dropout, Linear, Module
from torch.nn.functional import relu, sigmoid
import torch
from .pyg_data_interfaces import ProteinDataset


class FFNN(Module):

    def __init__(self,
                 in_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 out_dim: int,
                 dropout_p: float = 0.5):
        self.l1 = Linear(in_dim, hidden_dim1)
        self.l2 = Linear(hidden_dim1, hidden_dim2)
        self.dropout = Dropout(p=dropout_p)
        self.l_out = Linear(hidden_dim2, out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = relu(self.l1(X))
        h = self.dropout(relu(self.l2(X)))
        y_hat = self.l_out(h)
        if self.training:
            return y_hat
        return sigmoid(y_hat)

    def fit():
        pass

    def test():
        pass


__all__ = [
    "FFNN"
]


if __name__ == "__main__":
    pass
