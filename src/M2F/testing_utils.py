import logging
import torch

_logger = logging.getLogger(__name__)


def accuracy(logits, y_true, mask, threshold=0.5):
    """
    Execute `accuracy`.

    Args:
        logits: Input value for `logits`.
        y_true: Input value for `y_true`.
        mask: Input value for `mask`.
        threshold: Input value for `threshold`.
    """
    if mask.sum() == 0:
        _logger.debug("accuracy(): empty mask, returning 0.0")
        return 0.0
    probs = torch.sigmoid(logits[mask])
    preds = (probs >= threshold).float()
    true = y_true[mask]
    return (preds == true).float().mean().item()

def recall(logits, y_true, mask, threshold=0.5, eps=1e-8):
    """
    Execute `recall`.

    Args:
        logits: Input value for `logits`.
        y_true: Input value for `y_true`.
        mask: Input value for `mask`.
        threshold: Input value for `threshold`.
        eps: Input value for `eps`.
    """
    if mask.sum() == 0:
        _logger.debug("recall(): empty mask, returning 0.0")
        return 0.0
    probs = torch.sigmoid(logits[mask])
    preds = (probs >= threshold).float()
    true = y_true[mask]
    tp = ((preds == 1) & (true == 1)).sum().float()
    fn = ((preds == 0) & (true == 1)).sum().float()
    return (tp / (tp + fn + eps)).item()


def precision(logits, y_true, mask, threshold=0.5, eps=1e-8):
    """
    Execute `precision`.

    Args:
        logits: Input value for `logits`.
        y_true: Input value for `y_true`.
        mask: Input value for `mask`.
        threshold: Input value for `threshold`.
        eps: Input value for `eps`.
    """
    if mask.sum() == 0:
        _logger.debug("precision(): empty mask, returning 0.0")
        return 0.0
    probs = torch.sigmoid(logits[mask])
    preds = (probs >= threshold).float()
    true = y_true[mask]
    tp = ((preds == 1) & (true == 1)).sum().float()
    fp = ((preds == 1) & (true == 0)).sum().float()
    return (tp / (tp + fp + eps)).item()


def f1(logits, y_true, mask, threshold=0.5, eps=1e-8):
    """
    Execute `f1`.

    Args:
        logits: Input value for `logits`.
        y_true: Input value for `y_true`.
        mask: Input value for `mask`.
        threshold: Input value for `threshold`.
        eps: Input value for `eps`.
    """
    if mask.sum() == 0:
        _logger.debug("f1(): empty mask, returning 0.0")
        return 0.0
    probs = torch.sigmoid(logits[mask])
    preds = (probs >= threshold).float()
    true = y_true[mask]
    tp = ((preds == 1) & (true == 1)).sum().float()
    fp = ((preds == 1) & (true == 0)).sum().float()
    fn = ((preds == 0) & (true == 1)).sum().float()
    precision_ = tp / (tp + fp + eps)
    recall_ = tp / (tp + fn + eps)
    return (2.0 * precision_ * recall_ / (precision_ + recall_ + eps)).item()


__all__ = [
    "accuracy",
    "recall",
    "precision",
    "f1",
]

if __name__ == "__main__":
    pass
