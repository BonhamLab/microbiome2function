#    ___________________________________
#  /                                     \
# |   Currently unfinished interfaces     |
#  \ ___________________________________ /

# third party
from torch_geometric.nn import MessagePassing
from torch.nn import Dropout, Linear, Module
from torch.nn.functional import relu, sigmoid
from torch.optim.lr_scheduler import ExponentialLR
import torch
# built in 
from typing import Literal
from pathlib import Path
import logging
import os
# local
from .pyg_data_interfaces import ProteinGraphInMemoryDataset, ProteinGraphOnDisk
from .testing_utils import accuracy, recall
from .util import current_time

_logger = logging.getLogger(__name__)


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
        if self.training:
            return out
        return sigmoid(out)

    def fit(self,
            data: ProteinGraphInMemoryDataset | ProteinGraphOnDisk, # <-- need to swap for data loaders and adjust the code accordingly
            epochs: int,
            early_stopping: bool = True,
            save_model_to: Path | str | None = None,
            *,
            tolerance: int = 5,
            optimizer=None,
            optimizer_kwargs: dict = None,
            lr_sched=None,
            lr_sched_kwargs: dict = None,
            report_performance_every_kth_epoch: int = 10):
        k = report_performance_every_kth_epoch
        save_model_to = Path(save_model_to if save_model_to is not None else os.getcwd())
        # --------------------------------------------------------------------
        if optimizer is not None:
            if optimizer_kwargs is None:
                raise ValueError(...)
            optimizer = optimizer(**optimizer_kwargs)
        else:
            optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3, weight_decay=1e-4)
        
        if lr_sched is not None:
            if lr_sched_kwargs is None:
                raise ValueError(...)
            lr_sched_kwargs[optimizer]=optimizer
            lr_sched = lr_sched(**lr_sched_kwargs)
        else:
            lr_sched = ExponentialLR(optimizer=optimizer, gamma=0.99)

        criterion = torch.nn.BCEWithLogitsLoss()
        # --------------------------------------------------------------------

        no_generalization_after = 0
        val_loss = torch(float("inf"))
        mdl_name: str | None = None
        last_mdl_name: str | None = None
        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            logits = self(data.x, data.edge_index, data.edge_attr)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            lr_sched.step()

            self.eval()
            with torch.no_grad():
                logits = self(data.x, data.edge_index, data.edge_attr)
                current_val_loss = criterion(logits[data.val_mask], data.y[data.val_mask])

                if current_val_loss >= val_loss:
                    no_generalization_after += 1
                    if no_generalization_after > tolerance:
                        _logger.info(f"No generalization improvement after {no_generalization_after} epochs, stopping early.")
                        break
                else:
                    mdl_name = f"m2f_gnn_{current_time()}"
                    torch.save(self, save_model_to / mdl_name)
                    if last_mdl_name is not None:
                        os.remove(last_mdl_name)
                    last_mdl_name = mdl_name
                val_loss = current_val_loss

                if epoch == 1 or epoch % k == 0:
                    val_acc = accuracy(logits, data.y, data.val_mask)
                    val_recall = recall(logits, data.y, data.val_mask)
                    _logger.info(f"Validation accuracy and recall at epoch {epoch} are {val_acc}, {val_recall}, respectively")


    def test():
        pass


__all__ = [

]


if __name__ == "__main__":
    pass
