import torch
from torch.nn import Dropout, Linear, Module
from torch.nn.functional import relu
from torch_geometric.nn import MessagePassing
from typing import Literal


class GraphConv(MessagePassing):

    def __init__(self,
                 in_dim: int,
                 edge_dim: int,
                 msg_dim: int,
                 state_dim: int,
                 aggr: str = "max",
                 *,
                 aggr_kwargs=None,
                 flow: str = "source_to_target",
                 edge_features_used_as: Literal["scaling", "catting"] = "scaling",
                 node_dim: int = -2,
                 decomposed_layers: int = 1):
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
            flow=flow,
            node_dim=node_dim,
            decomposed_layers=decomposed_layers,
        )

        if edge_features_used_as not in {"scaling", "catting"}:
            raise ValueError(
                "`edge_features_used_as` must be either 'scaling' or 'catting'"
            )
        if edge_dim < 0:
            raise ValueError("`edge_dim` must be >= 0")
        if edge_features_used_as == "scaling" and edge_dim == 0:
            raise ValueError(
                "`edge_dim` must be > 0 when edge features are used as scaling gates"
            )

        msg_in_dim = in_dim if edge_features_used_as == "scaling" else in_dim + edge_dim
        self.msg_lin_transform = Linear(msg_in_dim, msg_dim, bias=False)
        self.upd_lin_transform = Linear(in_dim + msg_dim, state_dim, bias=True)
        self.edge_features_used_as = edge_features_used_as
        self.edge_weight_lin_transform = (
            Linear(edge_dim, 1, bias=True)
            if edge_features_used_as == "scaling"
            else None
        )
    
    def message(self, h_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self.edge_features_used_as == "scaling":
            gate = torch.sigmoid(self.edge_weight_lin_transform(edge_attr))
            return self.msg_lin_transform(gate * h_j)
        return self.msg_lin_transform(torch.cat([h_j, edge_attr], dim=-1))
    
    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        pre_h = torch.cat([h, aggr_out], dim=-1)
        return relu(self.upd_lin_transform(pre_h))

    def forward(self, h, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, h=h, edge_attr=edge_attr)


class GraphConvNodeClassifier(Module):
    def __init__(self,
                 in_dim: int,
                 edge_dim: int,
                 msg_dim: int,
                 state_dim: int,
                 out_dim: int,
                 *,
                 edge_features_used_as: Literal["scaling", "catting"] = "scaling",
                 dropout_p: float = 0.5):
        super().__init__()
        self.conv1 = GraphConv(
            in_dim=in_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            state_dim=state_dim,
            edge_features_used_as=edge_features_used_as,
        )
        self.conv2 = GraphConv(
            in_dim=state_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            state_dim=state_dim,
            edge_features_used_as=edge_features_used_as,
        )
        self.lin = Linear(state_dim, out_dim)
        self.dropout = Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        out = self.lin(h)
        return out
