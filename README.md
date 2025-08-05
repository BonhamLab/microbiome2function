# UniProt Data Engineering Pipeline  

A modular and scalable data engineering toolkit for **protein functional annotation** using UniProtKB data. This package automates:  

- 🔎 **Data mining** from UniProtKB (batched API calls on HPC or locally)  
- 🧹 **Data cleaning & normalization** (regex-based extraction of domains, GO IDs, EC numbers, etc.)  
- ⚙️ **Feature engineering** into numeric representations (dense embeddings & multi-hot encodings)  
- 💾 **Dataset assembly** into graph-ready tensors for ML models (e.g. GNNs)  

This pipeline was built for large-scale **bioinformatics + ML workflows** and is fully HPC-compatible.  

---

## Features  

- **UniProtKB API batching**  
  - Retrieve millions of protein records (`process_uniref_batches`)  
  - Features automatic retries, ID filtering, and per-batch logging  

- **Flexible cleaning utilities**  
  - Regex-based column parsers (`clean_col`, `clean_all_cols`)  
  - Handles nested data: FT domains, GO terms, Rhea IDs, cofactors, etc.  

- **Embedding + encoding**  
  - Dense amino acid sequence embeddings: **ProtT5** & **ESM2** via HuggingFace
  - Free-text functional annotations → OpenAI text embeddings with caching on your machine
  - GO terms, EC numbers, Rhea IDs → multi-hot vectors (with automatic feature space size control)  

- **HPC-first design**  
  - Example **SLURM job script** (`job.sh`)  
  - Timed log rotation and batch-level progress monitoring  

---

## Example Usage
Data Mining:
```python
from M2F import (extract_all_accessions_from_dir,
                fetch_save_uniprotkb_batches,
                configure_logging)
import re
import os

gene_fam_files_dir = os.getenv("SAMPLE_FILES")
output_dir = os.getenv("SAVE_DATA_TO_DIR")
job_name = os.getenv("JOB_NAME")
logs_dir = os.getenv("LOGS_DIR")

assert gene_fam_files_dir, "SAMPLE_FILES env var was not set!"
assert output_dir, "SAVE_DATA_TO_DIR env var was not set!"
assert job_name, "JOB_NAME env var was not set!"
assert logs_dir, "LOGS_DIR env var was not set!"

configure_logging(logs_dir)

accession_nums = extract_all_accessions_from_dir(gene_fam_files_dir, 
                                             pattern=re.compile(r".*_genefamilies\.tsv$"))


out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

fetch_save_uniprotkb_batches(
    uniref_ids=accession_nums,
    fields=["accession", "ft_domain", "cc_domain",
            "protein_families", "go_f", "go_p",
            "cc_function", "cc_catalytic_activity",
            "ec", "cc_pathway", "rhea", "cc_cofactor", "sequence"],
    batch_size=40_000,
    single_api_request_size=100,
    rps=10,
    save_to_dir=out,
)

print(f"Mined data is available at {out}")
```
Data Cleaning and Feature engineering:
```python
# TO BE ADDED
```

## Repository Structure
```graphql
dataset_curation/
├── scripts/
│   ├── mining_utils.py           # API batching & data retrieval
│   ├── cleaning_utils.py         # regex-based text cleaning
│   ├── feature_engineering_utils.py # embeddings & multi-hot encodings
│   ├── embedding_utils.py        # ProtT5, ESM2, OpenAI embedder
│   └── logging_utils.py          # logging setup (HPC-friendly)
├── for_hpc_use/
│   └── data_mining.py            # entrypoint for HPC jobs (example; alter in the way that fits your case)
├── job.sh                        # SLURM job script (example; alter in the way that fits your case)
├── notebooks/
    └── data_processing.ipynb     # notebook example of data preparation (end-to-end)
```

## Why This Matters
This pipeline solves the biggest pain point in protein functional annotation projects:
* Automates the entire data engineering lifecycle
* Scales from hundreds to millions of proteins
* Produces fully numeric datasets ready for deep learning (esp. Graph Neural Networks)
