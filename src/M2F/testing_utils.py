import torch


def accuracy(logits, y_true, mask, threshold=0.5):
    if mask.sum() == 0:
        return 0.0
    probs = torch.sigmoid(logits[mask])
    preds = (probs >= threshold).float()
    true = y_true[mask]
    return (preds == true).float().mean().item()

def recall(logits, y_true, mask, threshold=0.5, eps=1e-8):
    if mask.sum() == 0:
        return 0.0
    probs = torch.sigmoid(logits[mask])
    preds = (probs >= threshold).float()
    true = y_true[mask]
    tp = ((preds == 1) & (true == 1)).sum().float()
    fn = ((preds == 0) & (true == 1)).sum().float()
    return (tp / (tp + fn + eps)).item()


__all__ = [
    "accuracy",
    "recall"
]

if __name__ == "__main__":
    pass
