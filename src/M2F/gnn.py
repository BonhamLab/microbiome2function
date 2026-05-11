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
from torch_geometric.loader import NeighborLoader
from .testing_utils import accuracy, recall
from .util import current_time

_logger = logging.getLogger(__name__)


class GraphConv(MessagePassing):

    """
    Represent the `GraphConv` type.
    """
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
        """
        Initialize a `GraphConv` instance.

        Args:
            in_dim: Input value for `in_dim`.
            edge_dim: Input value for `edge_dim`.
            msg_dim: Input value for `msg_dim`.
            state_dim: Input value for `state_dim`.
            aggr: Input value for `aggr`.
            aggr_kwargs: Input value for `aggr_kwargs`.
            flow: Input value for `flow`.
            edge_features_used_as: Input value for `edge_features_used_as`.
            node_dim: Input value for `node_dim`.
            decomposed_layers: Input value for `decomposed_layers`.
        """
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
        """
        Execute `message`.

        Args:
            h_j: Input value for `h_j`.
            edge_attr: Input value for `edge_attr`.
        """
        if self.edge_features_used_as == "scaling":
            gate = torch.sigmoid(self.edge_weight_lin_transform(edge_attr))
            return self.msg_lin_transform(gate * h_j)
        return self.msg_lin_transform(torch.cat([h_j, edge_attr], dim=-1))
    
    def update(self, aggr_out: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Execute `update`.

        Args:
            aggr_out: Input value for `aggr_out`.
            h: Input value for `h`.
        """
        pre_h = torch.cat([h, aggr_out], dim=-1)
        return relu(self.upd_lin_transform(pre_h))

    def forward(self, h, edge_index, edge_attr):
        """
        Run forward propagation for `GraphConv`.

        Args:
            h: Input value for `h`.
            edge_index: Input value for `edge_index`.
            edge_attr: Input value for `edge_attr`.
        """
        return self.propagate(edge_index=edge_index, h=h, edge_attr=edge_attr)


class GraphConvNodeClassifier(Module):
    """
    Represent the `GraphConvNodeClassifier` type.
    """
    def __init__(self,
                 in_dim: int,
                 edge_dim: int,
                 msg_dim: int,
                 state_dim: int,
                 out_dim: int,
                 *,
                 edge_features_used_as: Literal["scaling", "catting"] = "scaling",
                 dropout_p: float = 0.5):
        """
        Initialize a `GraphConvNodeClassifier` instance.

        Args:
            in_dim: Input value for `in_dim`.
            edge_dim: Input value for `edge_dim`.
            msg_dim: Input value for `msg_dim`.
            state_dim: Input value for `state_dim`.
            out_dim: Input value for `out_dim`.
            edge_features_used_as: Input value for `edge_features_used_as`.
            dropout_p: Input value for `dropout_p`.
        """
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

    def _forward_logits(self, x, edge_index, edge_attr):
        """
        Execute `forward logits`.

        Args:
            x: Input value for `x`.
            edge_index: Input value for `edge_index`.
            edge_attr: Input value for `edge_attr`.
        """
        h = self.conv1(x, edge_index, edge_attr)
        h = relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_attr)
        return self.lin(h)

    def forward(self, x, edge_index, edge_attr):
        """
        Run forward propagation for `GraphConvNodeClassifier`.

        Args:
            x: Input value for `x`.
            edge_index: Input value for `edge_index`.
            edge_attr: Input value for `edge_attr`.
        """
        out = self._forward_logits(x, edge_index, edge_attr)
        if self.training:
            return out
        return sigmoid(out)

    def fit(self,
            train: NeighborLoader,
            val: NeighborLoader,
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
        """
        Fit the current object.

        Args:
            train: Input value for `train`.
            val: Input value for `val`.
            epochs: Input value for `epochs`.
            early_stopping: Input value for `early_stopping`.
            save_model_to: Input value for `save_model_to`.
            tolerance: Input value for `tolerance`.
            optimizer: Input value for `optimizer`.
            optimizer_kwargs: Input value for `optimizer_kwargs`.
            lr_sched: Input value for `lr_sched`.
            lr_sched_kwargs: Input value for `lr_sched_kwargs`.
            report_performance_every_kth_epoch: Input value for `report_performance_every_kth_epoch`.
        """
        if epochs < 1:
            raise ValueError("`epochs` must be >= 1")

        k = report_performance_every_kth_epoch
        save_model_to = Path(save_model_to if save_model_to is not None else os.getcwd())
        save_model_to.mkdir(parents=True, exist_ok=True)

        device = next(self.parameters()).device # note, need to take `next` of `self.parameters()` because it is an iterator
        criterion = torch.nn.BCEWithLogitsLoss()
        _logger.info(
            "Starting GNN fit (epochs=%d, early_stopping=%s, tolerance=%d, device=%s, save_dir=%s)",
            epochs,
            early_stopping,
            tolerance,
            device,
            save_model_to,
        )

        # ------------------------------- optimizer -------------------------------
        if optimizer is None:
            optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-3, weight_decay=1e-4)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            kwargs = dict(optimizer_kwargs or {})
            optimizer = optimizer(params=self.parameters(), **kwargs)
        # -------------------------------------------------------------------------

        # ------------------------------- scheduler -------------------------------
        if lr_sched is None:
            lr_sched = ExponentialLR(optimizer=optimizer, gamma=0.99)
        elif isinstance(lr_sched, type):
            kwargs = dict(lr_sched_kwargs or {})
            kwargs.setdefault("optimizer", optimizer)
            lr_sched = lr_sched(**kwargs)
        elif not hasattr(lr_sched, "step"):
            raise TypeError("`lr_sched` must be a scheduler instance or scheduler class.")
        # -------------------------------------------------------------------------

        no_generalization_after = 0
        best_val_loss = float("inf")
        best_model_path: Path | None = None
        history: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            # ------------------------------- train -------------------------------
            self.train()
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            train_recall_sum = 0.0
            train_examples = 0

            for batch in train:
                batch = batch.to(device)
                batch_size = int(getattr(batch, "batch_size", batch.y.size(0)))
                if batch_size == 0:
                    continue

                mask = torch.zeros(batch.y.size(0), dtype=torch.bool, device=device)
                mask[:batch_size] = True

                optimizer.zero_grad()
                logits = self._forward_logits(batch.x, batch.edge_index, batch.edge_attr)
                y = batch.y.float()
                loss = criterion(logits[mask], y[mask])
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_loss_sum += float(loss.item()) * batch_size
                    train_acc_sum += accuracy(logits, y, mask) * batch_size
                    train_recall_sum += recall(logits, y, mask) * batch_size
                    train_examples += batch_size

            if train_examples == 0:
                raise RuntimeError("Train loader produced no batches with seed nodes.")

            train_loss = train_loss_sum / train_examples
            train_acc = train_acc_sum / train_examples
            train_recall = train_recall_sum / train_examples
            # -------------------------------------------------------------------

            # -------------------------------- val ------------------------------
            self.eval()
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_recall_sum = 0.0
            val_examples = 0
            with torch.no_grad():
                for batch in val:
                    batch = batch.to(device)
                    batch_size = int(getattr(batch, "batch_size", batch.y.size(0)))
                    if batch_size == 0:
                        continue

                    mask = torch.zeros(batch.y.size(0), dtype=torch.bool, device=device)
                    mask[:batch_size] = True

                    logits = self._forward_logits(batch.x, batch.edge_index, batch.edge_attr)
                    y = batch.y.float()
                    loss = criterion(logits[mask], y[mask])

                    val_loss_sum += float(loss.item()) * batch_size
                    val_acc_sum += accuracy(logits, y, mask) * batch_size
                    val_recall_sum += recall(logits, y, mask) * batch_size
                    val_examples += batch_size

            if val_examples == 0:
                raise RuntimeError("Validation loader produced no batches with seed nodes.")

            current_val_loss = val_loss_sum / val_examples
            val_acc = val_acc_sum / val_examples
            val_recall = val_recall_sum / val_examples
            # -------------------------------------------------------------------

            # -------------------------- scheduler + early stop ------------------
            try:
                lr_sched.step(current_val_loss)
            except TypeError:
                lr_sched.step()

            improved = current_val_loss < best_val_loss
            if improved:
                best_val_loss = current_val_loss
                no_generalization_after = 0
                best_model_path = save_model_to / f"m2f_gnn_{current_time()}.pt"
                torch.save(self.state_dict(), best_model_path)
                _logger.debug(
                    "New best validation loss %.6f at epoch %d; saved checkpoint to %s",
                    best_val_loss,
                    epoch,
                    best_model_path,
                )
            else:
                no_generalization_after += 1

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_recall": train_recall,
                "val_loss": current_val_loss,
                "val_acc": val_acc,
                "val_recall": val_recall,
            })

            if epoch == 1 or epoch % k == 0:
                _logger.info(
                    "Epoch %d | train_loss=%.6f train_acc=%.4f train_recall=%.4f | "
                    "val_loss=%.6f val_acc=%.4f val_recall=%.4f",
                    epoch, train_loss, train_acc, train_recall, current_val_loss, val_acc, val_recall
                )

            if early_stopping and no_generalization_after > tolerance:
                _logger.info(
                    "No validation improvement for %d epoch(s). Stopping early.",
                    no_generalization_after
                )
                break
            # -------------------------------------------------------------------

        out = {
            "best_val_loss": best_val_loss,
            "best_model_path": str(best_model_path) if best_model_path is not None else None,
            "history": history,
        }
        _logger.info(
            "Finished GNN fit (epochs_ran=%d, best_val_loss=%.6f, best_model_path=%s)",
            len(history),
            best_val_loss,
            out["best_model_path"],
        )
        return out

    def test(self, test: NeighborLoader, *, threshold: float = 0.5) -> dict[str, float]:
        """
        Test the current object.

        Args:
            test: Input value for `test`.
            threshold: Input value for `threshold`.
        """
        device = next(self.parameters()).device
        criterion = torch.nn.BCEWithLogitsLoss()
        _logger.info("Starting GNN test (threshold=%.3f, device=%s)", threshold, device)

        self.eval()
        test_loss_sum = 0.0
        test_acc_sum = 0.0
        test_recall_sum = 0.0
        test_examples = 0

        with torch.no_grad():
            for batch in test:
                batch = batch.to(device)
                batch_size = int(getattr(batch, "batch_size", batch.y.size(0)))
                if batch_size == 0:
                    continue

                mask = torch.zeros(batch.y.size(0), dtype=torch.bool, device=device)
                mask[:batch_size] = True

                logits = self._forward_logits(batch.x, batch.edge_index, batch.edge_attr)
                y = batch.y.float()
                loss = criterion(logits[mask], y[mask])

                test_loss_sum += float(loss.item()) * batch_size
                test_acc_sum += accuracy(logits, y, mask, threshold=threshold) * batch_size
                test_recall_sum += recall(logits, y, mask, threshold=threshold) * batch_size
                test_examples += batch_size

        if test_examples == 0:
            raise RuntimeError("Test loader produced no batches with seed nodes.")

        metrics = {
            "test_loss": test_loss_sum / test_examples,
            "test_acc": test_acc_sum / test_examples,
            "test_recall": test_recall_sum / test_examples,
        }
        _logger.info(
            "Test metrics | loss=%.6f acc=%.4f recall=%.4f",
            metrics["test_loss"], metrics["test_acc"], metrics["test_recall"]
        )
        return metrics


__all__ = [

]


if __name__ == "__main__":
    pass
