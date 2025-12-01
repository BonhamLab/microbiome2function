# M2F — Microbiome to Function

**A practical pipeline for mining UniProt, cleaning annotations, and turning biology into machine-learnable features**

**Author:**  
Yehor Mishchyriak  
*Summer Research Intern, Bonham Lab, Tufts University School of Medicine (June–July 2025).  
Undergraduate Student, Wesleyan University (2022–2026)*

**Affiliations:**  
Bonham Lab, Tufts University School of Medicine  
Wesleyan University, Middletown, CT, USA

**Contact:**  
ymishchyriak@wesleyan.edu

---

## Contents
1. Overview  
2. Typical Data Flow  
3. API Reference  
   - logging  
   - data mining  
   - data cleaning  
   - numerical data encoding  
   - data persistence  
   - miscellaneous  
4. Extending M2F  
5. Examples  

---

# 1. Overview

**Problem.** Functional annotation projects need clean, machine-ready protein representations at scale.  
Raw UniProt text is messy — free-form prose, PubMed artifacts, heterogeneous annotations (GO, EC, domains, pathways). Most labs reinvent brittle scripts repeatedly.

**Solution (M2F).** A modular toolkit that:

- Mines UniProtKB at scale with rate-limited, batched REST calls.  
- Cleans & normalizes key text fields using targeted regex extractors.  
- Encodes features into numerical tensors:  
  - Dense embeddings for amino-acid sequences and free-text fields  
  - Structured label encodings for GO & EC  
- Persists datasets compactly in a single Zarr ZipStore with easy reconstruction.

**End state.** A reproducible pipeline from HUMAnN outputs or accession IDs → clean UniProt records → tidy vector tables (dense vectors + multihot tuples) ready for GNNs or any downstream ML.

---

# 2. Typical Data Flow

### 1. Accession mining  
HUMAnN gene-families TSV files are passed to:

- `extract_accessions_from_humann(file_path)`
- or `extract_all_accessions_from_dir(dir_path)`

Both return **UniRef** and **UniClust** accession iterables.

### 2. UniProtKB retrieval  
Only UniRef can be mined directly from UniProtKB. UniClust IDs are stashed.

Use:

- `fetch_uniprotkb_fields`
- or `fetch_save_uniprotkb_batches`

to retrieve selected UniProt fields.

### 3. Cleaning  
Using `clean_cols`, per-column regex extractors remove metadata, normalize free text, strip PubMed references, and produce predictable tuple-based columns.

### 4. Feature engineering  
Pass the cleaned DataFrame to feature-engineering utilities to encode:

- dense vector embeddings
- multihot label tuples

### 5. Persistence  
Use:

- `save_df(df, path)`  
- later `load_df(path)`

to store/load the ML-ready dataset in a single ZipStore.

---

# 3. API Reference

## Logging

### `configure_logging(logs_dir, file_level=logging.DEBUG, console_level=logging.WARNING)`
Configures a rotating file logger and console logger. Safe to call multiple times.

**Parameters**  
- `logs_dir`: directory for log files  
- `file_level`: file logging level  
- `console_level`: console logging level  

---

## Data Mining

### `extract_accessions_from_humann(file_path, out_type=list)`
Extracts UniRef and UniClust accessions from a HUMAnN gene-families TSV.  
Filters out `UNK*` and `UPI*` IDs. Raises `KeyError` if `READS_UNMAPPED` is missing.

**Returns:** `(unirefs, uniclusts)`

---

### `extract_all_accessions_from_dir(dir_path, pattern=None, out_type=list)`
Scans a directory of HUMAnN files, collecting UniRef90 and UniClust90 accessions.

**Returns:** `(all_unirefs, all_uniclusts)`

---

### `fetch_uniprotkb_fields(uniref_ids, fields, request_size=100, rps=10, max_retry=inf)`
Rate-limited, batched UniProtKB retrieval using the TSV REST API. Splits `uniref_ids`
into chunks of `request_size`, sleeps to obey `rps`, and on HTTP errors halves the
chunk size recursively until either success or `request_size==1`.

**Notes:**
- `request_size` must be ≥1  
- Failed IDs in the smallest chunk are dropped (with a warning)  
- Returns an empty DataFrame with the requested columns if nothing is retrieved  

---

### `fetch_save_uniprotkb_batches(...)`
Retrieves **very large** ID lists by splitting into coarse batches and writing each batch to Parquet/CSV.  
Designed for HPC/SLURM.

**Notes:** uses `fetch_uniprotkb_fields` under the hood (`single_api_request_size` per HTTP call), falls back to CSV if Parquet fails, and returns the output directory path.

---

## Data Cleaning

### `clean_col(df, col_name, apply_norm=True, apply_strip_pubmed=True, inplace=True)`
Cleans a single text column by:

- removing PubMed refs  
- applying column-specific regex extraction  
- normalizing (unless disabled)  
- de-duplicating while preserving order  
- returning tuple-based representations (NaNs become `()`)  

Raises `KeyError` if the column is missing. If no regex is defined for `col_name`,
the raw string is used.

---

### `clean_cols(df, col_names, apply_norms=None, apply_strip_pubmeds=None, inplace=False)`
Multi-column wrapper for `clean_col`. Defaults to `apply_norm=True` and
`apply_strip_pubmed=True` per column unless overridden via the provided dicts.
Raises `KeyError` if any column is absent.

---

# Numerical Data Encoding

## `AAChainEmbedder`
Mean-pooled ESM-2 embeddings for amino-acid sequences.

### Methods
**`.embed_sequences(seqs, batch_size=32)`**  
Returns a CPU list of `float32` vectors, one per sequence. Sequences longer than
the model’s max length are truncated (with a warning). `representation_layer`
accepts `"last"`, `"second_to_last"`, or an integer index. `model_key` must be one
of the bundled ESM-2 checkpoints (e.g., `esm2_t6_8M_UR50D`, `esm2_t36_3B_UR50D`).

---

## `FreeTXTEmbedder`
Embeds free-text strings using OpenAI embeddings with:

- RAM LRU cache  
- SQLite disk cache  

### Methods
**`.embed_sequences(seqs, batch_size=1000)`**

**Caching details:** caching is enabled only when both `cache_file_path` is set and
`caching_mode != "NOT_CACHING"`. `CREATE/OVERRIDE` currently behaves like `APPEND`
and does not wipe an existing DB; delete the file yourself to start fresh. LRU size
is approximate (KB-based); evicted entries are flushed to SQLite. Models must be
specified via the provided aliases (`SMALL_OPENAI_MODEL` / `LARGE_OPENAI_MODEL`).

---

## `MultiHotEncoder`
Encodes tuple-based string labels → tuple of integer indices. Raises `ValueError`
if any entry is not a tuple. Returns both the encoded tuples and the class→index map.

---

## `GOEncoder(obo_path)`
Collapses GO IDs to a chosen depth or auto-selects depth via coverage statistics
(depth = percentile of observed depths). Unknown GO IDs are skipped. Empty tuples
become NaN in the returned DataFrame.

---

## `ECEncoder()`
Collapses EC numbers (depth 1–4) or auto-selects optimal depth based on target density.
Auto-depth searches {4,3,2,1} for a class count closest to
`N/examples_per_class` (where `N` is total annotations). Empty tuples become NaN.

---

## `embed_freetxt_cols(df, cols, embedder, batch_size=1000, inplace=False)`
Embeds tuple-of-strings columns using FreeTXTEmbedder; keeps the embedding with the
largest L2 norm per row. Empty tuples become NaN.

---

## `embed_ft_domains(df, embedder, batch_size=128, inplace=False)`
Extracts 1-based domain ranges from the `Domain [FT]` column, slices the corresponding
`Sequence` string (stored as a singleton tuple), embeds each subsequence with ESM-2,
and keeps the domain embedding with the largest L2 norm (empty tuples → NaN).

---

## `embed_AAsequences(df, embedder, batch_size=128, inplace=False)`
Embeds full sequences. Expects the `Sequence` column to contain a tuple with the raw
sequence as its first element; empty tuples become NaN.

---

## Convenience wrappers

### `encode_go(df, col_name, depth=None, coverage_target=None, inplace=False)`
Bound to the packaged GO DAG (`dependencies/go-basic.obo`). Returns `(df, class_labels)`
where `class_labels` maps GO term → index.

### `encode_ec(df, col_name, depth=None, examples_per_class=30, inplace=False)`
Wrapper around `ECEncoder.encode_ec`. Returns `(df, class_labels)` with EC code → index.

### `encode_multihot(df, col, inplace=False)`
One-shot wrapper around `MultiHotEncoder`. Returns `(df, class_labels)`.

---

# Data Persistence

## `save_df(df, pth, metadata=None)`
Saves a heterogeneous DataFrame into a single **Zarr ZipStore**:

- strings → fixed-width ASCII  
- int-tuples → (flat array, offsets)  
- vectors → 2-D float32 arrays

Requires an `Entry` column of strings and a `.zip` path (raises `ValueError` otherwise).
Rows are sorted by `Entry` before writing. Columns that are completely empty are
tagged with `is_empty=True` in the store metadata. Unsupported dtypes raise `ValueError`.

---

## `load_df(path)`
Reconstructs the DataFrame saved via `save_df`. Appends `.zip` if missing in `path`,
restores ASCII strings and tuples/arrays, attaches stored attributes, and raises
`ValueError` on unknown column layouts. Columns are sorted alphabetically on load.

---

# Miscellaneous

## `empty_tuples_to_NaNs(df, inplace=False)`
Replaces all `()` with `np.nan`.

---

## util

### `files_from(dir_path, pattern=None)`  
Yields sorted filenames matching a regex pattern.

### `compose(*funcs)`  
Function composition that threads a value through the provided callables. The
returned wrapper expects per-function positional args keyed by the callable objects
(`fun_args_map[fn]`); as written this helper is not used elsewhere and will need
adjustment before practical use.

### `suppress_warnings(*warning_types)`  
Decorator for temporarily disabling warnings.

---

# 4. Extending M2F

- **New free-text column?**  
  Add regex to `AVAILABLE_EXTRACTION_PATTERNS`, run through `clean_cols`, then embed via `embed_freetxt_cols`.

- **New ontology?**  
  Subclass `MultiHotEncoder` with collapse-to-depth logic.

- **Different sequence model?**  
  Clone `AAChainEmbedder` and swap HF repo while keeping pooling semantics.

- **Custom serialization?**  
  Mirror `save_df` / `load_df`.

---

# 5. Examples

## Data Mining

```python
import M2F
import pandas as pd
import os

gene_fam_data = "/path/to/humann/files"

# note '_' because UniClust IDs also returned
unirefs, _ = M2F.extract_accessions_from_humann(gene_fam_data)

df = M2F.fetch_uniprotkb_fields(
    unirefs,
    fields=["accession", "ft_domain", "cc_function", "go_f", "go_p", "sequence"],
    request_size=100,
    max_retry=5
)

df.to_csv("my_uniprot_data.csv")
```

## Cleaning & Feature Engineering
```python
import M2F
import pandas as pd

col_names = [
    "Domain [FT]",
    "Gene Ontology (molecular function)",
    "Gene Ontology (biological process)",
    "Function [CC]",
    "Sequence"
]

apply_norms = {
    "Domain [FT]": False,
    "Gene Ontology (molecular function)": False,
    "Gene Ontology (biological process)": False,
    "Function [CC]": True,
    "Sequence": False
}

aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t6_8M_UR50D", device="cuda:0")
txt_embedder = M2F.FreeTXTEmbedder(
    api_key="my-openai-api-key",
    model="LARGE_OPENAI_MODEL",
    cache_file_path="example.db",
    caching_mode="CREATE/OVERRIDE",
    max_cache_size_kb=20000
)

def process_df_inplace(df, *, col_names, apply_norms):
    M2F.clean_cols(df, col_names=col_names, apply_norms=apply_norms, inplace=True)

    M2F.embed_ft_domains(df, aa_embedder, inplace=True)
    M2F.embed_AAsequences(df, aa_embedder, inplace=True)
    M2F.embed_freetxt_cols(df, ["Function [CC]"], txt_embedder, inplace=True)

    _, gomf_meta = M2F.encode_go(
        df, "Gene Ontology (molecular function)", coverage_target=0.8, inplace=True
    )
    _, gobp_meta = M2F.encode_go(
        df, "Gene Ontology (biological process)", coverage_target=0.8, inplace=True
    )

    return {"gomf_meta": gomf_meta, "gobp_meta": gobp_meta}

for file in M2F.util.files_from("/path/to/input/dir"):
    file_name = os.path.basename(file)
    out_pth = os.path.join("/path/to/output/dir", file_name.replace(".csv", ".zip"))

    df = pd.read_csv(file)
    meta = process_df_inplace(df, col_names=col_names, apply_norms=apply_norms)

    M2F.save_df(df, out_pth, metadata=meta)
```
