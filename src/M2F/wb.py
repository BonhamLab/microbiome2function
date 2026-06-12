import wandb
import torch


def is_active():
    return wandb.run is not None


def log_epoch(epoch,
            train_loss,
            train_accuracy,
            train_precision,
            train_recall,
            train_f1,
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            optimizer=None):

    metrics = {
        "epoch": epoch,

        "train/loss": train_loss,
        "train/accuracy": train_accuracy,
        "train/precision": train_precision,
        "train/recall": train_recall,
        "train/f1": train_f1,

        "val/loss": val_loss,
        "val/accuracy": val_accuracy,
        "val/precision": val_precision,
        "val/recall": val_recall,
        "val/f1": val_f1,
    }

    if optimizer is not None:
        metrics["lr"] = optimizer.param_groups[0]["lr"]

    if is_active():
        wandb.log(metrics)

    return metrics


def save_best_model(model, model_name, path):
    torch.save(model.state_dict(), path)
    if is_active():
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)

    return path
