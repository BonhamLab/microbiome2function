# M2F Documentation

This document describes the current `src/M2F` codebase.

## Scope

M2F provides:
- logging setup
- HUMAnN/UniProt mining helpers
- regex-based annotation cleaning
- sequence/text embedding utilities
- GO/EC and generic multi-hot encoders
- dataframe persistence (`save_df` / `load_df`)
- a PyG in-memory dataset interface for graph training

## Public API (`import M2F`)

Exported from `src/M2F/__init__.py`:

### Logging
- `configure_logging`

### Mining
- `extract_accessions_from_humann`
- `extract_all_accessions_from_dir`
- `fetch_uniprotkb_fields`
- `fetch_save_uniprotkb_batches`

### Cleaning
- `clean_col`
- `clean_cols`

### Embedding / Encoding
- `AAChainEmbedder`
- `FreeTXTEmbedder`
- `MultiHotEncoder`
- `GOEncoder`
- `ECEncoder`
- `encode_multihot`
- `get_GODag`

### Feature engineering / persistence
- `embed_ft_domains`
- `embed_AAsequences`
- `embed_freetxt_cols`
- `encode_go`
- `encode_ec`
- `empty_tuples_to_NaNs`
- `save_df`
- `load_df`

### Utilities
- `util`

## Module Notes

### Mining (`src/M2F/mining_utils.py`)

`fetch_uniprotkb_fields(...)` performs rate-limited batched UniProt calls, retries HTTP failures with smaller batches, and returns one concatenated DataFrame.

### Cleaning (`src/M2F/cleaning_utils.py`)

`clean_col` / `clean_cols` extract structured values with column-specific regexes, optionally normalize text, and convert cell payloads to tuples.

### Embedding (`src/M2F/embedding_utils.py`)

- `AAChainEmbedder`: mean-pooled ESM2 embeddings.
- `FreeTXTEmbedder`: OpenAI embeddings with optional RAM+SQLite caching.
- `GOEncoder` / `ECEncoder`: depth-cut + label encoding.

Current depth behavior:
- GO/EC terms that do **not** reach requested depth are dropped.
- Rows that become empty are represented as `NaN` in `encode_go` / `encode_ec` outputs.

### Feature engineering (`src/M2F/feature_engineering_utils.py`)

Provides wrappers and higher-level transforms:
- `embed_AAsequences`
- `embed_ft_domains`
- `embed_freetxt_cols`
- `encode_go` (bound GO encoder)
- `encode_ec` (bound EC encoder)

### Persistence (`src/M2F/feature_engineering_utils.py`)

- `save_df` writes heterogeneous payloads to a `.zip` Zarr store.
- `load_df` reconstructs the DataFrame.

## PyG InMemory Graph Interface

File: `src/M2F/pyg_data_interfaces.py`

This module is **not** exported from `M2F.__init__`; import directly.

```python
from M2F.pyg_data_interfaces import DatasetInput, ProteinGraphInMemoryDataset
```

### `DatasetInput`

Validated contract with:
- accession index CSV (`uniref`, `i`)
- edge chunk directory (`chunk_<id>.csv` style by default)
- requested UniProt fields (`uniprot_features`)
- supervised fields (`X`, `Y`)
- UniProt request parameters (`request_size`, `rps`, `max_retry`)
- flexible edge schema:
  - destination column (`edge_dst_column`, default `j`)
  - optional fixed edge attribute columns (`edge_attr_columns`)

### `ProteinGraphInMemoryDataset`

Current lifecycle:
1. `download()`
   - queries UniProt and stores `raw/features.csv`
   - materializes index + edge chunks into `raw/`
2. `process()`
   - reads `features.csv` and index file
   - normalizes UniProt accession alias (`Entry` -> `accession`)
   - aligns graph nodes to features by accession
   - applies dataset-level `pre_transform(node_df) -> DataFrame`
   - applies dataset-level `pre_filter(node_df) -> boolean mask`
   - removes rows missing required `X`/`Y`
   - reindexes nodes after filtering
   - builds `x`, `y`, `edge_index`, `edge_attr`
   - supports variable-dimensional `edge_attr`
   - writes single processed graph to `processed/data.pt`

Stored metadata on `Data` includes:
- `node_id_to_accession`
- `x_fields`
- `y_field`
- `edge_attr_fields`

### On-disk interface status

`ProteinGraphOnDiskDataset` has been removed from the codebase.

## Minimal PyG Usage

```python
from pathlib import Path
from M2F.pyg_data_interfaces import DatasetInput, ProteinGraphInMemoryDataset

inp = DatasetInput(
    path_to_accession_ids_csv_file=Path("untracked/test_data_subset/uniref_index_count.csv"),
    path_to_edge_csv_dir=Path("untracked/test_data_subset"),
    uniprot_features=("accession", "sequence", "go_f"),
    X=("sequence",),
    Y="go_f",
    edge_dst_column="j",
)

ds = ProteinGraphInMemoryDataset(
    root=Path("untracked/prot_graph_root"),
    dataset_input=inp,
    pre_transform=my_dataframe_transform,
    pre_filter=my_dataframe_filter,
    force_reload=True,
)

data = ds[0]
```

## Current Structure

- package: `src/M2F`
- active notebooks: `model_notebooks`
- scratch outputs: `untracked`
- legacy scripts/examples: `legacy_code_examples`
