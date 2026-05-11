![LOGO](https://raw.githubusercontent.com/Yehor-Mishchyriak/microbiome2function/main/assets/M2F_banner.png)

# microbiome2function (M2F)

A toolkit for turning UniProt-linked protein annotations into machine-learning datasets.

M2F supports:
- UniProt mining from UniRef accessions
- Annotation cleaning and normalization
- Embedding/encoding into numeric features
- Dataset interfaces for:
  - PyTorch Geometric GNN training (`ProteinGraphInMemoryDataset`, `ProteinGraphOnDiskDataset`)
  - Plain PyTorch FFNN training (`ProteinDataset`)

## Package Status

- Package name: `microbiome2function`
- Python: `>=3.11,<3.13`
- Source layout: `src/`
- Main package: `src/M2F`

## Installation

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Public API (Top-Level)

```python
import M2F
```

Current top-level exports include:
- Logging: `configure_logging`
- Mining: `extract_accessions_from_humann`, `extract_all_accessions_from_dir`, `fetch_uniprotkb_fields`, `fetch_save_uniprotkb_batches`
- Cleaning: `clean_col`, `clean_cols`
- Embedding/encoding: `AAChainEmbedder`, `FreeTXTEmbedder`, `MultiHotEncoder`, `GOEncoder`, `ECEncoder`, `encode_multihot`, `get_GODag`
- Feature engineering/persistence: `embed_ft_domains`, `embed_AAsequences`, `embed_freetxt_cols`, `encode_go`, `encode_ec`, `empty_tuples_to_NaNs`, `save_df`, `load_df`
- Models: `FFNN`, `GraphConv`, `GraphConvNodeClassifier`
- Metrics: `accuracy`, `recall`, `precision`, `f1`
- Dataset interfaces: `DatasetInput`, `build_topology_from_DatasetInput`, `build_features_from_DatasetInput`, `ProteinGraphInMemoryDataset`, `ProteinGraphOnDiskDataset`, `ProteinDataset`
- Utility namespace: `util`

## Typical Workflow

1. Extract accessions from HUMAnN output (or provide your own UniRef list).
2. Fetch UniProt fields in batches.
3. Clean annotation columns into tuple-based representations.
4. Encode/embed into numeric vectors.
5. Build model dataset (graph or non-graph).
6. Train and evaluate model.

## Quick Start: Mining + Cleaning + Persistence

```python
import logging
import M2F

M2F.configure_logging("logs", file_level=logging.DEBUG, console_level=logging.INFO)

# 1) Mine UniRef IDs
unirefs, _ = M2F.extract_accessions_from_humann("sample_gene_families.tsv")

# 2) Fetch UniProt fields
raw = M2F.fetch_uniprotkb_fields(
    uniref_ids=unirefs,
    fields=["accession", "sequence", "go_f", "ec"],
    request_size=50,
    rps=5,
    max_retry=20,
)

# 3) Clean columns (example)
cleaned = M2F.clean_cols(
    raw,
    col_names=["Gene Ontology (molecular function)", "EC number", "Sequence"],
    inplace=False,
)

# 4) Persist
M2F.save_df(cleaned, "features.zip", metadata={"pipeline": "example"})
restored = M2F.load_df("features.zip")
```

Notes:
- `fields` must be valid UniProt API field identifiers.
- Returned DataFrame column names may differ from query keys; align downstream mappings accordingly.
- `save_df` expects an `Entry` column and `.zip` output path.

## Graph Data Interface (PyG)

M2F includes two graph dataset interfaces:
- `ProteinGraphInMemoryDataset`: one processed graph saved to `processed/data.pt`.
- `ProteinGraphOnDiskDataset`: zarr-backed feature store + on-disk topology for larger datasets.

### Required Input Contract

Use `DatasetInput` to define schema and retrieval parameters:

```python
from pathlib import Path
from M2F import DatasetInput

inp = DatasetInput(
    path_to_accession_ids_csv_file=Path("data/uniref_index.csv"),
    path_to_edge_csv_dir=Path("data/edges"),
    X={"sequence": "Sequence", "go_f": "go_mf"},
    Y={"ec": "target_ec"},
    request_size=25,
    rps=1.0,
    max_retry=20,
    num_feature_batches=8,
    edge_dst_column="j",
)
```

Accession index CSV must contain exactly:
- `uniref` (e.g. `UniRef90_*`)
- `i` (1-based integer node IDs)

For graph mode, edge chunk files must exist and match the expected naming pattern (default `chunk_<i>.csv`).

### On-Disk Graph Example

```python
from pathlib import Path
import torch
from M2F import ProteinGraphOnDiskDataset, GraphConvNodeClassifier

ds = ProteinGraphOnDiskDataset(
    root=Path("runs/graph_ondisk"),
    dataset_input=inp,
    force_reload=False,
    val_set_size=0.1,
    test_set_size=0.1,
)

train_loader = ds.train_loader(num_neighbors=[15, 10], batch_size=1024, shuffle=True)
val_loader = ds.val_loader(num_neighbors=[15, 10], batch_size=1024)
test_loader = ds.test_loader(num_neighbors=[15, 10], batch_size=1024)

model = GraphConvNodeClassifier(
    in_dim=int(ds.meta["x_dim"]),
    edge_dim=int(ds.meta["edge_attr_dim"]),
    msg_dim=128,
    state_dim=128,
    out_dim=int(ds.meta["y_dim"]),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

history = model.fit(
    train=train_loader,
    val=val_loader,
    epochs=30,
    tolerance=5,
    report_performance_every_kth_epoch=1,
    save_model_to="runs/checkpoints_gnn",
)

metrics = model.test(test_loader)
print(history["best_val_loss"], metrics)

ds.close()
```

## FFNN Dataset Interface (`ProteinDataset`)

For non-graph workflows, use `ProteinDataset` (zarr-backed features + targets, no edges):

```python
from pathlib import Path
import torch
from M2F import DatasetInput, ProteinDataset, FFNN

inp_ffnn = DatasetInput(
    path_to_accession_ids_csv_file=Path("data/uniref_index.csv"),
    X={"sequence": "Sequence"},
    Y={"ec": "target_ec"},
    num_feature_batches=8,
)

dset = ProteinDataset(
    root=Path("runs/ffnn_dataset"),
    dataset_input=inp_ffnn,
    split="train",
    include_targets=True,
)

train_loader = dset.train_loader(batch_size=512, shuffle=True)
val_loader = dset.val_loader(batch_size=512)
test_loader = dset.test_loader(batch_size=512)

model = FFNN(in_dim=int(dset.meta["x_dim"]), hidden_dim1=256, hidden_dim2=128, out_dim=int(dset.meta["y_dim"]))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

history = model.fit(train_loader, val_loader, epochs=30, tolerance=5, report_performance_every_kth_epoch=1)
metrics = model.test(test_loader)
print(history["best_val_loss"], metrics)

dset.close()
```

## Important Operational Notes

- `ProteinGraphOnDiskDataset` and `ProteinDataset` process features in batches and build a global node reindex map.
- Topology for on-disk graph datasets is built after feature processing so filtered-node reindexing is stable.
- Feature shards with duplicate `Entry` rows are rejected.
- Inconsistent per-row feature dimensions are rejected.
- `force_reload=True` rebuilds raw/processed artifacts from scratch.

## Logging

```python
import logging
from M2F import configure_logging

configure_logging(
    logs_dir="logs",
    file_level=logging.DEBUG,
    console_level=logging.INFO,
)
```

## CI Workflows

- `/.github/workflows/test.yml`: multi-version test matrix (3.11, 3.12)
- `/.github/workflows/build.yml`: test + build distribution artifacts (`sdist`, wheel)

## Repository Layout

- `src/M2F`: package code
- `tests`: unit tests
- `model_notebooks`: active notebooks
- `legacy_code_examples`: old examples
- `docs.md`: detailed technical guide

## Detailed Documentation

For full API behavior, data contracts, and extended cookbook usage, see:
- [`docs.md`](docs.md)
