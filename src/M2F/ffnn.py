# third party
from torch.nn import Dropout, Linear, Module
from torch.nn.functional import relu, sigmoid
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader as pt_DataLoader
import torch

# built in
from pathlib import Path
import logging
import os

# local
from .testing_utils import accuracy, recall
from .util import current_time

_logger = logging.getLogger(__name__)


class FFNN(Module):

    """
    Represent the `FFNN` type.
    """
    def __init__(self,
                 in_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 out_dim: int,
                 dropout_p: float = 0.5):
        """
        Initialize a `FFNN` instance.

        Args:
            in_dim: Input value for `in_dim`.
            hidden_dim1: Input value for `hidden_dim1`.
            hidden_dim2: Input value for `hidden_dim2`.
            out_dim: Input value for `out_dim`.
            dropout_p: Input value for `dropout_p`.
        """
        super().__init__()
        self.l1 = Linear(in_dim, hidden_dim1)
        self.l2 = Linear(hidden_dim1, hidden_dim2)
        self.dropout = Dropout(p=dropout_p)
        self.l_out = Linear(hidden_dim2, out_dim)

    def _forward_logits(self, X: torch.Tensor) -> torch.Tensor:
        """
        Execute `forward logits`.

        Args:
            X: Input value for `X`.
        """
        h = relu(self.l1(X))
        h = self.dropout(relu(self.l2(h)))
        return self.l_out(h)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Run forward propagation for `FFNN`.

        Args:
            X: Input value for `X`.
        """
        y_hat = self._forward_logits(X)
        if self.training:
            return y_hat
        return sigmoid(y_hat)

    def fit(self,
            train: pt_DataLoader,
            val: pt_DataLoader,
            epochs: int,
            early_stopping: bool = True,
            save_model_to: Path | str | None = None,
            *,
            tolerance: int = 5,
            optimizer=None,
            optimizer_kwargs: dict | None = None,
            lr_sched=None,
            lr_sched_kwargs: dict | None = None,
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
        if report_performance_every_kth_epoch < 1:
            raise ValueError("`report_performance_every_kth_epoch` must be >= 1")
        if tolerance < 0:
            raise ValueError("`tolerance` must be >= 0")

        k = report_performance_every_kth_epoch
        save_model_to = Path(save_model_to if save_model_to is not None else os.getcwd())
        save_model_to.mkdir(parents=True, exist_ok=True)

        device = next(self.parameters()).device
        criterion = torch.nn.BCEWithLogitsLoss()
        _logger.info(
            "Starting FFNN fit (epochs=%d, early_stopping=%s, tolerance=%d, device=%s, save_dir=%s)",
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

            for X, y in train:
                X = X.to(device)
                y = y.to(device).float()
                batch_size = int(X.size(0))
                if batch_size == 0:
                    continue

                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                optimizer.zero_grad()
                logits = self._forward_logits(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                with torch.no_grad(): # each step's loss is weighted by num_examples_in_step / total_num_examples
                    train_loss_sum += float(loss.item()) * batch_size
                    train_acc_sum += accuracy(logits, y, mask) * batch_size
                    train_recall_sum += recall(logits, y, mask) * batch_size
                    train_examples += batch_size

            if train_examples == 0:
                raise RuntimeError("Train loader produced no non-empty batches.")

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
                for X, y in val:
                    X = X.to(device)
                    y = y.to(device).float()
                    batch_size = int(X.size(0))
                    if batch_size == 0:
                        continue

                    mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                    logits = self._forward_logits(X)
                    loss = criterion(logits, y)

                    val_loss_sum += float(loss.item()) * batch_size
                    val_acc_sum += accuracy(logits, y, mask) * batch_size
                    val_recall_sum += recall(logits, y, mask) * batch_size
                    val_examples += batch_size

            if val_examples == 0:
                raise RuntimeError("Validation loader produced no non-empty batches.")

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
                best_model_path = save_model_to / f"m2f_ffnn_{current_time()}.pt"
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
            "Finished FFNN fit (epochs_ran=%d, best_val_loss=%.6f, best_model_path=%s)",
            len(history),
            best_val_loss,
            out["best_model_path"],
        )
        return out

    def test(self, test: pt_DataLoader, *, threshold: float = 0.5) -> dict[str, float]:
        """
        Test the current object.

        Args:
            test: Input value for `test`.
            threshold: Input value for `threshold`.
        """
        device = next(self.parameters()).device
        criterion = torch.nn.BCEWithLogitsLoss()
        _logger.info("Starting FFNN test (threshold=%.3f, device=%s)", threshold, device)

        self.eval()
        test_loss_sum = 0.0
        test_acc_sum = 0.0
        test_recall_sum = 0.0
        test_examples = 0

        with torch.no_grad():
            for X, y in test:
                X = X.to(device)
                y = y.to(device).float()
                batch_size = int(X.size(0))
                if batch_size == 0:
                    continue

                mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                logits = self._forward_logits(X)
                loss = criterion(logits, y)

                test_loss_sum += float(loss.item()) * batch_size
                test_acc_sum += accuracy(logits, y, mask, threshold=threshold) * batch_size
                test_recall_sum += recall(logits, y, mask, threshold=threshold) * batch_size
                test_examples += batch_size

        if test_examples == 0:
            raise RuntimeError("Test loader produced no non-empty batches.")

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
    "FFNN"
]


if __name__ == "__main__":
    pass
