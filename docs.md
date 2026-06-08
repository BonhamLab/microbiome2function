# M2F Documentation

This document is the implementation-accurate user guide for `microbiome2function` (M2F).
It is written against the current code under `src/M2F`.

## 1. What M2F Is For

M2F is a practical toolkit for turning protein identifiers and UniProt annotations into ML-ready inputs.

Primary use-cases:
- Mine UniProt features from UniRef IDs.
- Clean and normalize noisy annotation text.
- Convert biology fields into numeric tensors (embeddings + encodings).
- Build datasets for graph and non-graph modeling:
- Graph neural networks (PyTorch Geometric): `ProteinGraphInMemoryDataset`, `ProteinGraphOnDiskDataset`.
- Feed-forward neural networks (plain PyTorch): `ProteinDataset` (features + labels, no edges).

Design goals:
- Scalable processing for large accession sets (batched UniProt mining and batched feature shards).
- Reproducible schema validation at dataset boundaries.
- Explicit failure on ambiguous data (duplicate accessions, inconsistent tensor dimensions).

## 2. Install and Environment

## 2.1 Python / Packaging

Project metadata (`pyproject.toml`):
- Package name: `microbiome2function`
- Python: `>=3.11,<3.13`
- Source layout: `src/`

Install from repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Why editable install:
- Keeps imports stable (`import M2F`) while you iterate code.
- Ensures tests and notebooks run against your local working tree.

## 2.2 Heavy Dependencies to Plan For

`requirements.txt` includes large ML packages:
- `torch==2.8.0`
- `torch-geometric==2.7.0`
- `transformers==4.55.0`
- `zarr==3.1.1`
- `openai==1.99.3`

Operational implications:
- GNN workflows require PyG-compatible PyTorch environment.
- ESM embedding workflows require HuggingFace model downloads (first run cost).
- Free-text embedding workflows require a valid OpenAI API key.

## 2.3 Logging Setup (Recommended)

```python
import logging
from M2F import configure_logging

configure_logging(
    logs_dir="logs",
    file_level=logging.DEBUG,
    console_level=logging.INFO,
)
```

Why:
- M2F emits useful progress and validation messages during mining, processing, and training.
- Debug logs are especially useful for long batched dataset builds.

## 3. Public API Overview

Top-level import path:

```python
import M2F
```

Current exported API (`M2F.__all__`) includes:
- Logging: `configure_logging`.
- Mining: `extract_accessions_from_humann`, `extract_all_accessions_from_dir`, `fetch_uniprotkb_fields`, `fetch_save_uniprotkb_batches`.
- Cleaning: `clean_col`, `clean_cols`.
- Embedding / Encoding: `AAChainEmbedder`, `FreeTXTEmbedder`, `MultiHotEncoder`, `GOEncoder`, `ECEncoder`, `encode_multihot`, `get_GODag`.
- Feature engineering / persistence: `embed_ft_domains`, `embed_AAsequences`, `embed_freetxt_cols`, `encode_go`, `encode_ec`, `empty_tuples_to_NaNs`, `save_df`, `load_df`.
- Models: `FFNN`, `GraphConv`, `GraphConvNodeClassifier`, `GATNodeClassifier`.
- Metrics: `accuracy`, `recall`, `precision`, `f1`.
- Dataset interfaces: `DatasetInput`, `build_topology_from_DatasetInput`, `build_features_from_DatasetInput`, `ProteinGraphInMemoryDataset`, `ProteinGraphOnDiskDataset`, `ProteinDataset`.
- Utility namespace: `util`.

## 4. Data Contracts You Must Respect

M2F works well only if input schemas are strict. This is intentional.

## 4.1 Accession Index CSV

Expected columns exactly:
- `uniref`
- `i`

Constraints enforced by `DatasetInput.validate(...)`:
- `i` must be integer dtype.
- `i` must be 1-based positive IDs.
- `i` must not contain duplicates.
- `uniref` values must start with `UniRef90_`.

Why strict index requirements:
- All reindexing and topology construction depend on deterministic old node IDs (`i - 1`).
- Relaxed IDs would make edge mapping ambiguous and error-prone.

## 4.2 Edge CSV Files (Graph Datasets Only)

Required only when `require_graph=True` (graph interfaces).

Defaults:
- File pattern: `chunk_\d+\.csv`
- Destination column: `j`

Rules:
- Number of edge files must equal number of accession rows.
- Each edge file must contain destination column `edge_dst_column`.
- Optional `edge_attr_columns` must exist if explicitly provided.

Why one chunk per source node:
- Source node ID is inferred from filename (`chunk_<i>.csv`).
- This keeps topology build streaming and deterministic.

## 4.3 `DatasetInput` Query/Return Mapping

`DatasetInput` uses:
- `X: dict[str, str]` mapping UniProt query field -> return column name.
- `Y: dict[str, str]` singleton mapping UniProt query field -> return column name.

Important:
- `Y` must contain exactly one entry.
- `Y` cannot overlap with `X` keys or values.
- `Y` key cannot be `accession`.
- `accession` is always injected into `X` internally as `"Entry"`.

Why mapping instead of plain list:
- You control the semantic output names used by downstream feature builders.
- It decouples UniProt field identifiers from model-facing column names.

## 5. Quick Start: End-to-End Patterns

## 5.1 Mining Accessions from HUMAnN

```python
from M2F import extract_accessions_from_humann, extract_all_accessions_from_dir

unirefs, uniclusts = extract_accessions_from_humann("sample_gene_families.tsv")
all_unirefs, all_uniclusts = extract_all_accessions_from_dir("humann_outputs/")
```

Notes:
- UniRef IDs prefixed with `UNK`/`UPI` are excluded before UniProt mining because they are not queryable reliably.

## 5.2 Fetch UniProt Fields

```python
from M2F import fetch_uniprotkb_fields

df = fetch_uniprotkb_fields(
    uniref_ids=["A0A1B2C3D4", "Q9XYZ1"],
    fields=["accession", "sequence", "go_f", "ec"],
    request_size=50,
    rps=5,
    max_retry=20,
)
```

Field-name note:
- `fields` values must be valid UniProt API field identifiers.
- Returned DataFrame column names can differ from query names (for example, title-cased labels).
- Your later mapping/transforms must match the actual returned column names.

Recommended defaults for stability:
- Start with moderate `request_size` (25-100).
- Keep `rps` conservative if network is noisy.

Why this matters:
- On HTTP failures, the function recursively halves batch size. A too-large initial request size can increase retries and runtime variance.

## 5.3 Clean UniProt Text Columns

```python
from M2F import clean_cols

cleaned = clean_cols(
    df,
    col_names=[
        "Gene Ontology (molecular function)",
        "EC number",
        "Domain [FT]",
        "Function [CC]",
    ],
    inplace=False,
)
```

What you get:
- Each cleaned column becomes tuple-based tokenized values.
- Missing/empty values become `()` in cleaning stage.

Why tuple outputs:
- Deterministic multi-label representation that plugs directly into encoders.

## 5.4 Encode and Embed

```python
import os
from M2F import (
    AAChainEmbedder,
    FreeTXTEmbedder,
    embed_AAsequences,
    embed_freetxt_cols,
    encode_go,
    encode_ec,
)

# Ensure tuple-based cell format expected by embedding/encoding wrappers.
cleaned = cleaned.copy()
cleaned["Sequence"] = cleaned["Sequence"].map(
    lambda s: (s,) if isinstance(s, str) and s else s
)

# Sequence embeddings (ESM2)
aa = AAChainEmbedder(model_key="esm2_t6_8M_UR50D", device="cpu")
df1 = embed_AAsequences(cleaned, embedder=aa, batch_size=64, inplace=False)

# Free-text embeddings (OpenAI)
txt = FreeTXTEmbedder(
    api_key=os.environ["OPENAI_API_KEY"],
    model="SMALL_OPENAI_MODEL",
    cache_file_path="embeddings.sqlite",
    caching_mode="APPEND",
    max_cache_size_kb=200_000,
)
df2 = embed_freetxt_cols(df1, cols=["Function [CC]"], embedder=txt, batch_size=512, inplace=False)

# Structured label encoding
df3, go_vocab = encode_go(df2, col_name="Gene Ontology (molecular function)", coverage_target=0.8)
df4, ec_vocab = encode_ec(df3, col_name="EC number", examples_per_class=30)
```

Input-shape note for wrappers:
- `embed_AAsequences` expects `"Sequence"` cells to be singleton tuples like `("MSEQ...",)`.
- `embed_freetxt_cols` expects tuple-of-strings per row.
- `encode_go` and `encode_ec` expect tuple-encoded labels per row.

Why staged transformation is useful:
- You can inspect each step and catch schema/coverage issues early.
- Encoder vocabularies (`go_vocab`, `ec_vocab`) are needed for interpretation and consistent inference.

## 5.5 Persist Feature Tables

```python
from M2F import save_df, load_df

save_df(df4, "features.zip", metadata={"source": "uniprot_2026_05_11"})
restored = load_df("features.zip")
```

Persistence format constraints:
- `save_df` requires `.zip` extension.
- DataFrame must include `Entry` accession column.
- Supported payload types per cell: `tuple` (ragged integer encodings) or `np.ndarray` (dense embeddings).

## 6. Graph Dataset Cookbook (PyG)

## 6.1 Build `DatasetInput`

```python
from pathlib import Path
from M2F import DatasetInput

inp = DatasetInput(
    path_to_accession_ids_csv_file=Path("data/uniref_index.csv"),
    path_to_edge_csv_dir=Path("data/edges"),
    X={
        "sequence": "Sequence",
        "go_f": "go_mf",
    },
    Y={
        "ec": "target_ec",
    },
    request_size=25,
    rps=1.0,
    max_retry=20,
    num_feature_batches=8,
    edge_dst_column="j",
    edge_attr_columns=None,
)
```

Why `num_feature_batches` matters:
- Controls UniProt mining shard count for on-disk and FFNN datasets.
- Useful when one-shot feature retrieval is too memory- or network-heavy.

## 6.2 In-Memory Graph (`ProteinGraphInMemoryDataset`)

Use when graph fits in RAM/GPU workflow and you want a single `Data` object.

```python
from pathlib import Path
from M2F import ProteinGraphInMemoryDataset

ds = ProteinGraphInMemoryDataset(
    root=Path("runs/graph_inmem"),
    dataset_input=inp,
    pre_transform=None,  # DataFrame -> DataFrame
    pre_filter=None,     # DataFrame -> boolean mask
    force_reload=False,
    val_set_size=0.1,
    test_set_size=0.1,
)

data = ds[0]
print(data.x.shape, data.edge_index.shape, data.y.shape)
```

Process summary:
1. Download UniProt features to `raw/features.csv`.
2. Materialize index + edge shards into `raw/`.
3. Build node features and labels with `build_features_from_DatasetInput`.
4. Build topology with `build_topology_from_DatasetInput`.
5. Apply `RandomNodeSplit` masks.
6. Save processed graph to `processed/data.pt`.

Key behavior:
- Drops UNK/UPI accessions.
- Drops nodes missing required `X`/`Y` fields after transform/filter.
- Reindexes surviving nodes.
- Duplicates reverse edges to provide undirected message passing.

## 6.3 On-Disk Graph (`ProteinGraphOnDiskDataset`)

Use when graph feature matrix is large and should be streamed from disk.

```python
from pathlib import Path
from M2F import ProteinGraphOnDiskDataset

ondisk = ProteinGraphOnDiskDataset(
    root=Path("runs/graph_ondisk"),
    dataset_input=inp,
    pre_transform=None,
    pre_filter=None,
    force_reload=False,
    val_set_size=0.1,
    test_set_size=0.1,
)
```

Storage layout:
- Raw feature shards: `raw/features_batches/features_<i>.csv`
- Processed zarr store: `processed/feature_store/`
- Processed topology: `processed/edge_index.npy`
- Old->new ID map: `processed/id_map.npy`
- Metadata: `processed/meta.pt`

Two-pass processing logic (important):
1. Pass 1 (features): process each feature shard and append `x`/`y` to zarr.
2. Pass 2 (topology): once global reindex map is complete, build edges and edge attrs.

Why this is necessary:
- Filtering can drop nodes, so final node IDs are known only after feature passes.
- Building topology post-hoc avoids reindexing already-written edge tensors.

### Loaders

```python
train_loader = ondisk.train_loader(num_neighbors=[15, 10], batch_size=1024, shuffle=True)
val_loader   = ondisk.val_loader(num_neighbors=[15, 10], batch_size=1024)
test_loader  = ondisk.test_loader(num_neighbors=[15, 10], batch_size=1024)
```

Under the hood:
- Loader pulls node features from zarr-backed `FeatureStore`.
- Edge index comes from `GraphStore`.
- Edge attributes are attached per mini-batch via batch `e_id` lookup.

Operational note:
- Call `ondisk.close()` when done to release store handles.

## 6.4 Feature and Topology Builders as Standalone Functions

M2F exposes:
- `build_features_from_DatasetInput(...)`
- `build_topology_from_DatasetInput(...)`

Use these if you need custom dataset orchestration.

Important guardrails already implemented:
- Duplicate `Entry` rows in a feature shard raise `ValueError`.
- Duplicate old node assignment across shards raises `ValueError`.
- Inconsistent `X` or `Y` row dimensionality raises `ValueError`.

## 7. FFNN Dataset Cookbook (`ProteinDataset`)

`ProteinDataset` is the non-graph companion for dense feed-forward models.

What it provides:
- Batched UniProt download and processing like on-disk graph pipeline.
- Zarr-backed feature store with `x` and `y` only.
- Train/val/test/all split views.
- Dataloaders for training or prediction.

```python
from pathlib import Path
from M2F import DatasetInput, ProteinDataset

inp_ffnn = DatasetInput(
    path_to_accession_ids_csv_file=Path("data/uniref_index.csv"),
    X={"sequence": "Sequence"},
    Y={"ec": "target_ec"},
    num_feature_batches=8,
)

dset = ProteinDataset(
    root=Path("runs/ffnn_data"),
    dataset_input=inp_ffnn,
    split="train",
    include_targets=True,
    force_reload=False,
)

x, y = dset[0]
print(x.shape, y.shape)
```

Split control:

```python
# Mutate active split on same object
_ = dset.set_split("val", include_targets=True)

# Or create immutable view objects
val_view = dset.view("val", include_targets=True)
pred_view = dset.view("all", include_targets=False)
```

Dataloaders:

```python
train_loader = dset.train_loader(batch_size=512, shuffle=True)
val_loader = dset.val_loader(batch_size=512)
test_loader = dset.test_loader(batch_size=512)
predict_loader = dset.predict_loader(batch_size=512, split="all")
```

Why separate FFNN dataset class:
- Avoids graph store complexity when only node-wise feature prediction is needed.
- Reuses robust batch-processing, filtering, reindexing, and zarr growth path.

Operational note:
- Call `dset.close()` when done.

## 8. Model Training Cookbook

## 8.1 GNN: `GraphConvNodeClassifier`

```python
import torch
from M2F import GraphConvNodeClassifier

model = GraphConvNodeClassifier(
    in_dim=128,
    edge_dim=4,
    msg_dim=64,
    state_dim=64,
    out_dim=1,
    edge_features_used_as="scaling",  # or "catting"
    dropout_p=0.3,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

history = model.fit(
    train=train_loader,
    val=val_loader,
    epochs=50,
    early_stopping=True,
    tolerance=5,
    report_performance_every_kth_epoch=1,
    save_model_to="runs/checkpoints_gnn",
)

metrics = model.test(test_loader, threshold=0.5)
print(history["best_val_loss"], metrics)
```

Implementation details worth knowing:
- Loss: `BCEWithLogitsLoss`.
- During neighbor sampling, only seed nodes are supervised in each batch (`batch_size` mask logic).
- `fit(...)` returns `best_val_loss`, `best_model_path`, and epoch-wise `history`.

## 8.2 GNN Attention: `GATNodeClassifier`

`GATNodeClassifier` is the attention-based graph model. It uses the same `ProteinGraphInMemoryDataset` / `ProteinGraphOnDiskDataset` loaders as `GraphConvNodeClassifier`.

```python
import torch
from M2F import GATNodeClassifier

model = GATNodeClassifier(
    in_dim=128,
    edge_dim=4,
    msg_dim=64,  # retained for constructor compatibility
    state_dim=64,
    out_dim=1,
    heads=4,
    attention_dropout_p=0.1,
    dropout_p=0.3,
)

model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

history = model.fit(
    train=train_loader,
    val=val_loader,
    epochs=50,
    early_stopping=True,
    tolerance=5,
    report_performance_every_kth_epoch=1,
    save_model_to="runs/checkpoints_gat",
)

metrics = model.test(test_loader, threshold=0.5)
print(history["best_val_loss"], metrics)
```

Implementation details worth knowing:
- Internally uses PyTorch Geometric `GATConv`.
- `state_dim` must be divisible by `heads` because head outputs are concatenated.
- Edge attributes are used when `edge_dim > 0`; empty edge-attribute tensors are ignored when `edge_dim=0`.
- Training, evaluation, masking, loss, and returned history match `GraphConvNodeClassifier`.

## 8.3 FFNN: `FFNN`

```python
import torch
from M2F import FFNN

model = FFNN(in_dim=128, hidden_dim1=256, hidden_dim2=128, out_dim=1, dropout_p=0.3)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

history = model.fit(
    train=train_loader,
    val=val_loader,
    epochs=50,
    early_stopping=True,
    tolerance=5,
    report_performance_every_kth_epoch=1,
    save_model_to="runs/checkpoints_ffnn",
)

metrics = model.test(test_loader, threshold=0.5)
print(history["best_val_loss"], metrics)
```

Implementation details:
- Loss: `BCEWithLogitsLoss`.
- `forward(...)` returns logits during training, sigmoid probabilities during eval.

## 8.4 Metrics Utilities

Available helpers (`M2F.testing_utils`):
- `accuracy(logits, y_true, mask, threshold=0.5)`
- `recall(logits, y_true, mask, threshold=0.5)`
- `precision(logits, y_true, mask, threshold=0.5)`
- `f1(logits, y_true, mask, threshold=0.5)`

Use case:
- Fast masked binary metrics on logits for custom loops.

## 9. Advanced Notes and Common Failure Modes

## 9.1 Duplicate Accessions in Feature Shards

`build_features_from_DatasetInput` explicitly rejects duplicate `Entry` values within a shard.

Why:
- One accession must correspond to one node row.
- Duplicate rows would make node feature assignment non-deterministic.

## 9.2 Why Topology Is Built After Features for On-Disk Workflows

If filtering drops nodes, old IDs must be remapped to dense new IDs.
You cannot finalize edges safely until global `id_map` is complete.

Practical consequence:
- Build/append node features first.
- Build graph edges second using final map.

## 9.3 `force_reload` Semantics

For `ProteinGraphOnDiskDataset` and `ProteinDataset`:
- `force_reload=True` deletes processed artifacts and raw feature batch folder before rebuild.

Use `force_reload=True` when:
- You changed preprocessing logic.
- You changed requested fields or split sizes.
- You suspect stale corrupted partial artifacts.

## 9.4 Consistent Vector Dimensions Are Mandatory

M2F validates that all rows produce identical flattened dimensions for:
- X feature vector
- Y target vector

If dimensions vary across rows, processing fails early.

Why:
- Tensor storage and model input layers require fixed dimensionality.

## 9.5 OpenAI Embedding Cost and Caching

For `FreeTXTEmbedder`:
- Always set `cache_file_path` + `caching_mode="APPEND"` for repeated experiments.
- Choose `max_cache_size_kb` large enough to reduce DB churn.

Why:
- Re-embedding repeated biological text can dominate runtime and API cost.

## 10. Full Cookbook Example (Graph Pipeline)

```python
import logging
from pathlib import Path

from M2F import (
    configure_logging,
    DatasetInput,
    ProteinGraphOnDiskDataset,
    GraphConvNodeClassifier,
    GATNodeClassifier,
)

configure_logging("logs", file_level=logging.DEBUG, console_level=logging.INFO)

inp = DatasetInput(
    path_to_accession_ids_csv_file=Path("data/uniref_index.csv"),
    path_to_edge_csv_dir=Path("data/edges"),
    X={"sequence": "Sequence", "go_f": "go_mf"},
    Y={"ec": "target_ec"},
    request_size=25,
    rps=1.0,
    max_retry=20,
    num_feature_batches=8,
)

ds = ProteinGraphOnDiskDataset(
    root=Path("runs/gds"),
    dataset_input=inp,
    force_reload=False,
    val_set_size=0.1,
    test_set_size=0.1,
)

train_loader = ds.train_loader(num_neighbors=[15, 10], batch_size=1024, shuffle=True)
val_loader = ds.val_loader(num_neighbors=[15, 10], batch_size=1024)
test_loader = ds.test_loader(num_neighbors=[15, 10], batch_size=1024)

x_dim = int(ds.meta["x_dim"])
edge_dim = int(ds.meta["edge_attr_dim"])
y_dim = int(ds.meta["y_dim"])

model = GraphConvNodeClassifier(
    in_dim=x_dim,
    edge_dim=edge_dim,
    msg_dim=128,
    state_dim=128,
    out_dim=y_dim,
)

# Drop-in attention variant for the same dataset and loaders:
# model = GATNodeClassifier(
#     in_dim=x_dim,
#     edge_dim=edge_dim,
#     msg_dim=128,
#     state_dim=128,
#     out_dim=y_dim,
#     heads=4,
#     attention_dropout_p=0.1,
# )

history = model.fit(
    train=train_loader,
    val=val_loader,
    epochs=30,
    tolerance=5,
    report_performance_every_kth_epoch=1,
    save_model_to="runs/checkpoints",
)

metrics = model.test(test_loader)
print(history["best_val_loss"], metrics)

ds.close()
```

## 11. Testing and CI

Local test run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Current CI workflows:
- `.github/workflows/test.yml`: runs tests on pushes/PRs (Python 3.11 and 3.12).
- `.github/workflows/build.yml`: runs tests, builds package dists (`sdist` + wheel), validates with twine, and uploads build artifacts.

Install from built artifact (example):

```bash
python -m pip install dist/microbiome2function-0.1.0-py3-none-any.whl
```

## 12. Practical Recommendations

- Start small: validate your `DatasetInput` and preprocessing on a tiny accession subset first.
- Keep transform contracts strict: `pre_transform` must return DataFrame; `pre_filter` must return boolean mask with matching length.
- Use explicit checkpoints: preserve `meta.pt`, vocab maps, and model checkpoints per experiment.
- Close on-disk datasets: call `close()` to release zarr handles after training/inference.
- Avoid silent schema drift: pin requested UniProt fields and return names in code, not notebooks-only state.

## 13. Module Index

- `M2F.logging_utils`: logger configuration.
- `M2F.mining_utils`: accession extraction + UniProt mining.
- `M2F.cleaning_utils`: regex-based annotation cleaning.
- `M2F.embedding_utils`: ESM and OpenAI embedding + GO/EC/multihot encoders.
- `M2F.feature_engineering_utils`: high-level embedding wrappers + zarr zip persistence.
- `M2F.pyg_data_interfaces`: graph and FFNN dataset interfaces + standalone builders.
- `M2F.gnn`: graph convolution, graph attention, and training/eval loops.
- `M2F.ffnn`: feed-forward model and training/eval loops.
- `M2F.testing_utils`: metric helpers.
- `M2F.util`: utility helpers and zarr feature-store backend.
