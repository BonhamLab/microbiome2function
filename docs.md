# M2F Documentation

This document describes the **M2F package** in this repository (`M2F/`) based on the current code.

## Scope

M2F provides:
- logging setup
- UniProt/HUMAnN mining utilities
- regex-based cleaning
- embedding and multi-hot encoders
- feature engineering helpers
- compact dataframe persistence (`.zip` Zarr stores)

It also contains an experimental PyG interfaces module:
- `M2F/pyg_data_interfaces.py`

## Public API (`import M2F`)

`M2F/__init__.py` exports:

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

### Embedding / Encoders
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
- `util` module

---

## 1. Logging

### `configure_logging(logs_dir, file_level=logging.DEBUG, console_level=logging.WARNING)`

Configures the root logger with:
- timed rotating file handler (daily, keep 7)
- console handler

If handlers already exist on root, it returns immediately.

---

## 2. Mining Utilities

### `extract_accessions_from_humann(file_path, out_type=list)`

Reads a HUMAnN gene-families TSV (`READS_UNMAPPED` column), extracts:
- UniRef90 accessions
- UniClust90 accessions

Filters out UniRef accessions starting with `UNK` / `UPI`.

Returns `(unirefs, uniclusts)` converted to `out_type`.

### `extract_all_accessions_from_dir(dir_path, pattern=None, out_type=list)`

Iterates files from `dir_path` (optional regex filter), calls `extract_accessions_from_humann`, unions results, returns both collections as `out_type`.

### `fetch_uniprotkb_fields(uniref_ids, fields, request_size=100, rps=10, max_retry=inf)`

Rate-limited batched UniProtKB TSV fetch.
- retries HTTP failures recursively with smaller batch size
- returns concatenated DataFrame (or empty DataFrame with requested columns)

### `fetch_save_uniprotkb_batches(uniref_ids, fields, batch_size, single_api_request_size=100, rps=10, save_to_dir=None)`

Large-scale wrapper over `fetch_uniprotkb_fields`.
- splits ID list into coarse batches
- saves each batch to parquet (CSV fallback)
- returns output directory path

---

## 3. Cleaning Utilities

### `clean_col(df, col_name, apply_norm=True, apply_strip_pubmed=True, inplace=True)`

Column cleaning pipeline:
- remove PubMed snippets (optional)
- regex extraction by column name (when pattern exists)
- normalization (optional)
- dedupe
- output tuples (NaN -> `()`)

### `clean_cols(df, col_names, apply_norms=None, apply_strip_pubmeds=None, inplace=False)`

Multi-column wrapper around `clean_col` with per-column toggles.

---

## 4. Embedding / Encoding

### 4.1 `AAChainEmbedder`

Mean-pooled ESM2 sequence embeddings.

Constructor:
- `model_key` one of bundled ESM2 checkpoints
- `device` (cpu/cuda)
- `dtype`
- `representation_layer` (`"last"`, `"second_to_last"`, or integer)

Method:
- `embed_sequences(seqs, batch_size=32) -> list[np.ndarray]`

Notes:
- masks out special/padding tokens before pooling
- returns CPU `float32` numpy vectors
- truncates overlong sequences with warning

### 4.2 `FreeTXTEmbedder`

OpenAI embedding wrapper with optional caches:
- RAM LRU
- SQLite disk cache

Method:
- `embed_sequences(seqs, batch_size=1000) -> list[np.ndarray]`

### 4.3 `MultiHotEncoder`

- `encode(sequences: pd.Series)` expects tuple labels per row
- outputs:
  - `encodings`: tuple[int, ...] per row
  - `class_labels`: label -> class index map

### 4.4 `GOEncoder`

- `encode_go(df, col_name, depth=None, coverage_target=None, inplace=False)`
- `cut_to_depth(df, col_name, depth, inplace=False, empty_to_nan=True)`

Behavior:
- collapses GO IDs to a specified depth
- supports auto-depth via `coverage_target`
- then multi-hot encodes

### 4.5 `ECEncoder`

- `encode_ec(df, col_name, depth=None, examples_per_class=30, inplace=False)`
- `cut_to_depth(df, col_name, depth, inplace=False, empty_to_nan=True)`

Behavior:
- collapses EC levels to requested depth (1..4)
- supports auto-depth via class-budget heuristic
- then multi-hot encodes

### 4.6 `encode_multihot(df, col, inplace=False)`

One-shot multihot wrapper using `MultiHotEncoder`.

---

## 5. Feature Engineering and Persistence

### Embedding helpers

- `embed_freetxt_cols(df, cols, embedder, batch_size=1000, inplace=False)`
  - embeds tuple-of-strings free-text columns
  - row output = max-pooled embedding

- `embed_ft_domains(df, embedder, batch_size=128, inplace=False)`
  - extracts domain subsequences from:
    - `Domain [FT]` (ranges)
    - `Sequence`
  - embeds and max-pools

- `embed_AAsequences(df, embedder, batch_size=128, inplace=False)`
  - embeds full protein sequence from `Sequence`

### GO/EC wrappers

In `feature_engineering_utils.py`:
- `encode_go` is a bound alias of `GOEncoder.encode_go`
- `encode_ec` is a bound alias of `ECEncoder.encode_ec`

### Persistence

- `save_df(df, pth, metadata=None)`
  - writes a `.zip` Zarr store
  - expects `Entry` column for accession keys
  - supports tuple and ndarray column payloads

- `load_df(path)`
  - reconstructs dataframe from saved ZipStore

- `empty_tuples_to_NaNs(df, inplace=False)`

---

## 6. Utility Module (`M2F.util`)

- `files_from(dir_path, pattern=None)`
- `compose(*funcs)`
- `suppress_warnings(*warning_types)`

---

## 7. Experimental PyG Interfaces (WIP)

File: `M2F/pyg_data_interfaces.py`

Contains:
- `DatasetInput`
- `ProteinGraphInMemoryDataset`
- `ProteinGraphOnDiskDataset`

Current status:
- `DatasetInput` validation is implemented
- `download()` for `ProteinGraphInMemoryDataset` is partially implemented
- `process()` for `ProteinGraphInMemoryDataset` is still a stub (`pass`)
- `ProteinGraphOnDiskDataset` is mostly scaffold/stub

Important:
- This module is **not** exported in `M2F/__init__.py`
- API here should be treated as in-progress

---

## 8. Minimal Usage Pattern

```python
import M2F

M2F.configure_logging("./logs")

# 1) Mine accessions
unirefs, uniclusts = M2F.extract_accessions_from_humann("sample_genefamilies.tsv")

# 2) Fetch UniProt fields
df = M2F.fetch_uniprotkb_fields(
    uniref_ids=unirefs,
    fields=["accession", "sequence", "go_f", "ec"],
    request_size=100,
    rps=10,
)

# 3) Clean
# (use column names returned by UniProt TSV response)
# df = M2F.clean_cols(...)

# 4) Encode
# df, go_map = M2F.encode_go(df, "Gene Ontology (molecular function)", depth=6)
# df, ec_map = M2F.encode_ec(df, "EC number", depth=3)

# 5) Persist
M2F.save_df(df, "processed_dataset.zip")
restored = M2F.load_df("processed_dataset.zip")
```

---

## 9. Known Data Assumptions

- many encoders/embedders expect tuple-like cleaned columns
- `save_df`/`load_df` expect an `Entry` accession column
- `embed_ft_domains` expects both `Domain [FT]` and `Sequence`
- GO/EC encoders operate on GO/EC string collections per row
