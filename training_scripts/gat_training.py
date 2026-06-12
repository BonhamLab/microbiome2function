from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from M2F.cleaning_utils import clean_col
from M2F.embedding_utils import AAChainEmbedder
from M2F.feature_engineering_utils import embed_AAsequences, encode_go
from M2F.gnn import GATNodeClassifier
from M2F.logging_utils import configure_logging
from M2F.pyg_data_interfaces import DatasetInput, ProteinGraphInMemoryDataset
import M2F.wb as wb


_logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a GAT node classifier on an M2F protein co-occurrence graph."
    )

    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force-reload", action="store_true")

    parser.add_argument("--go-depth", type=int, default=5)
    parser.add_argument("--aa-model-key", default="esm2_t6_8M_UR50D")
    parser.add_argument("--aa-batch-size", type=int, default=16)
    parser.add_argument("--aa-device", default="auto")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-neighbors", default="15,10")
    parser.add_argument("--tolerance", type=int, default=5)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--msg-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-dropout-p", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--val-set-size", type=float, default=0.1)
    parser.add_argument("--test-set-size", type=float, default=0.1)
    parser.add_argument("--request-size", type=int, default=25)
    parser.add_argument("--rps", type=float, default=1.0)
    parser.add_argument("--max-retry", type=int, default=20)

    parser.add_argument("--sequence-col", default="Sequence")
    parser.add_argument("--go-col", default="Gene Ontology (molecular function)")
    parser.add_argument("--edge-dst-column", default="j")
    parser.add_argument("--edge-attr-columns", default="v")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))

    args = parser.parse_args()
    if args.sequence_col != "Sequence":
        parser.error("--sequence-col must be 'Sequence'; embed_AAsequences currently expects that column.")
    return args


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def parse_num_neighbors(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_edge_attr_columns(value: str) -> tuple[str, ...] | None:
    value = value.strip()
    if value.lower() in {"", "none", "null"}:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def as_go_multihot(idx_tuple, y_dim: int):
    if not isinstance(idx_tuple, tuple) or y_dim == 0:
        return np.nan
    vec = np.zeros(y_dim, dtype=np.float32)
    if idx_tuple:
        vec[np.asarray(idx_tuple, dtype=np.int64)] = 1.0
    return vec if vec.sum() > 0 else np.nan


def make_pre_transform(args: argparse.Namespace, aa_device: str, go_label_map: dict[str, int]):
    aa_encoder = AAChainEmbedder(model_key=args.aa_model_key, device=aa_device)

    def composed_pre_transform(node_df: pd.DataFrame) -> pd.DataFrame:
        df = node_df.copy()

        df = clean_col(
            df,
            args.sequence_col,
            apply_norm=False,
            apply_strip_pubmed=False,
            inplace=True,
        )
        df = clean_col(
            df,
            args.go_col,
            apply_norm=False,
            apply_strip_pubmed=True,
            inplace=True,
        )

        df, labels = encode_go(
            df,
            col_name=args.go_col,
            depth=args.go_depth,
            inplace=True,
        )
        go_label_map.clear()
        go_label_map.update(labels)
        y_dim = len(go_label_map)

        df.loc[:, args.go_col] = df[args.go_col].map(
            lambda idx_tuple: as_go_multihot(idx_tuple, y_dim)
        )

        return embed_AAsequences(
            df,
            embedder=aa_encoder,
            batch_size=args.aa_batch_size,
            inplace=True,
        )

    return composed_pre_transform


def make_pre_filter(args: argparse.Namespace):
    def pre_filter_mask(df: pd.DataFrame):
        x_ok = df[args.sequence_col].map(
            lambda x: isinstance(x, np.ndarray) and x.size > 0 and np.isfinite(x).all()
        )
        y_ok = df[args.go_col].map(
            lambda y: (
                isinstance(y, np.ndarray)
                and y.size > 0
                and np.isfinite(y).all()
                and y.sum() > 0
            )
        )
        return x_ok & y_ok

    return pre_filter_mask


def start_wandb_if_requested(args: argparse.Namespace, config: dict) -> None:
    if not args.wandb_project:
        return

    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "group": args.wandb_group,
        "job_type": "gat-training",
        "config": config,
    }
    if args.wandb_mode:
        init_kwargs["mode"] = args.wandb_mode

    init_kwargs["config"] = json_safe(init_kwargs["config"])
    wb.wandb.init(**init_kwargs)


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(str(args.run_dir / "logs"), console_level=logging.INFO)
    set_seed(args.seed)

    model_device = resolve_device(args.device)
    aa_device = resolve_device(args.aa_device)
    num_neighbors = parse_num_neighbors(args.num_neighbors)
    edge_attr_columns = parse_edge_attr_columns(args.edge_attr_columns)

    dataset_root = args.run_dir / "dataset"
    model_dir = args.run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    go_label_map: dict[str, int] = {}
    dataset_input = DatasetInput(
        path_to_accession_ids_csv_file=args.data_dir / "uniref_index_count.csv",
        path_to_edge_csv_dir=args.data_dir,
        X={"sequence": args.sequence_col},
        Y={"go_f": args.go_col},
        edge_dst_column=args.edge_dst_column,
        edge_attr_columns=edge_attr_columns,
        request_size=args.request_size,
        rps=args.rps,
        max_retry=args.max_retry,
    )

    pre_transform = make_pre_transform(args, aa_device, go_label_map)
    pre_filter = make_pre_filter(args)

    ds = ProteinGraphInMemoryDataset(
        root=dataset_root,
        dataset_input=dataset_input,
        pre_transform=pre_transform,
        pre_filter=pre_filter,
        force_reload=args.force_reload,
        val_set_size=args.val_set_size,
        test_set_size=args.test_set_size,
    )

    data = ds[0]
    in_dim = int(data.x.size(-1))
    out_dim = int(data.y.size(-1)) if data.y.ndim > 1 else 1
    edge_dim = int(data.edge_attr.size(-1)) if data.edge_attr is not None else 0

    model_config = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "in_dim": in_dim,
        "edge_dim": edge_dim,
        "out_dim": out_dim,
        "num_neighbors": num_neighbors,
        "model_device": model_device,
        "aa_device": aa_device,
    }
    _logger.info("Dataset/model config: %s", model_config)

    if go_label_map:
        with open(args.run_dir / "go_label_map.json", "w", encoding="utf-8") as handle:
            json.dump(go_label_map, handle, indent=2, sort_keys=True)

    start_wandb_if_requested(args, {**vars(args), **model_config})

    try:
        train_loader = ds.train_loader(
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = ds.val_loader(
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
        )
        test_loader = ds.test_loader(
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
        )

        model = GATNodeClassifier(
            in_dim=in_dim,
            edge_dim=edge_dim,
            msg_dim=args.msg_dim,
            state_dim=args.state_dim,
            out_dim=out_dim,
            heads=args.heads,
            attention_dropout_p=args.attention_dropout_p,
            dropout_p=args.dropout_p,
        ).to(model_device)

        fit_result = model.fit(
            train=train_loader,
            val=val_loader,
            epochs=args.epochs,
            save_model_to=model_dir,
            tolerance=args.tolerance,
            optimizer=torch.optim.Adam,
            optimizer_kwargs={"lr": args.lr, "weight_decay": args.weight_decay},
            report_performance_every_kth_epoch=args.report_every,
        )

        if fit_result["best_model_path"] is not None:
            state = torch.load(fit_result["best_model_path"], map_location=model_device)
            model.load_state_dict(state)

        test_metrics = model.test(test_loader, threshold=args.threshold)
        if wb.is_active():
            wb.wandb.log({key.replace("_", "/"): value for key, value in test_metrics.items()})

        output = {
            "args": json_safe(vars(args)),
            "model_config": json_safe(model_config),
            "fit_result": fit_result,
            "test_metrics": test_metrics,
        }
        with open(args.run_dir / "results.json", "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

        print(json.dumps({"best_model_path": fit_result["best_model_path"], **test_metrics}, indent=2))
    finally:
        if wb.is_active():
            wb.wandb.finish()


if __name__ == "__main__":
    main()
