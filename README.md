# M2F: Mining to Features

A practical pipeline for mining UniProt, cleaning annotations, and turning biology into machine-learnable features.

## Overview

**Problem**: Functional annotation projects need clean, machine-ready representations of proteins at scale. Raw UniProt text is messy (free-form prose, PubMed debris, lots of metadata), and annotations are heterogeneous (GO terms, EC numbers, functional domains, pathways). Researchers end up reinventing the same brittle scripts over and over.

**Solution**: M2F is a modular toolkit that provides a reproducible path from HUMAnN outputs or protein accession IDs to mined UniProt records to a tidy, numeric table (vectors + multi-hot tuples) ready for GNNs or any downstream ML.

## Key Features

- **Scale-friendly mining**: Rate-limited, batched REST calls to UniProtKB
- **Intelligent cleaning**: Targeted regex extractors for key text fields
- **Rich feature encoding**:
  - Dense embeddings for amino acid sequences (ESM-2) and free-text fields (OpenAI)
  - Structured label encodings for GO terms and EC numbers
- **Compact persistence**: Single Zarr ZipStore with easy dataset reconstruction
- **HPC-ready**: Designed for SLURM batch processing

## Installation

```bash
pip install m2f  # (once available)
# or clone and install from source
```

## Quick Start

### 1. Extract protein accessions from HUMAnN outputs

```python
import M2F

# Single file
unirefs, uniclusts = M2F.extract_accessions_from_humann("gene_families.tsv")

# Entire directory
unirefs, uniclusts = M2F.extract_all_accessions_from_dir("/path/to/humann/outputs/")
```

### 2. Mine UniProtKB data

```python
# Fetch specific fields for your proteins
df = M2F.fetch_uniprotkb_fields(
    uniref_ids=unirefs,
    fields=["accession", "ft_domain", "cc_function", "go_f", "go_p", "sequence"],
    request_size=100,
    rps=10  # respect rate limits
)
```

### 3. Clean and encode features

```python
# Set up embedders
aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t6_8M_UR50D", device="cuda:0")
txt_embedder = M2F.FreeTXTEmbedder(
    api_key="your-openai-key",
    model="LARGE_OPENAI_MODEL",
    cache_file_path="embeddings.db"
)

# Clean raw text
col_names = ["Domain [FT]", "Function [CC]", "Gene Ontology (molecular function)", "Sequence"]
apply_norms = {"Function [CC]": True, "Sequence": False, ...}  # normalize free text
M2F.clean_cols(df, col_names=col_names, apply_norms=apply_norms, inplace=True)

# Generate embeddings
M2F.embed_AAsequences(df, aa_embedder, inplace=True)
M2F.embed_ft_domains(df, aa_embedder, inplace=True) 
M2F.embed_freetxt_cols(df, ["Function [CC]"], txt_embedder, inplace=True)

# Encode structured annotations
df, go_vocab = M2F.encode_go(df, "Gene Ontology (molecular function)", coverage_target=0.8)
df, ec_vocab = M2F.encode_ec(df, "EC_column", examples_per_class=30)
```

### 4. Save and load datasets

```python
# Persist everything in a single compressed file
metadata = {"go_vocab": go_vocab, "ec_vocab": ec_vocab, "created": "2025-08-14"}
M2F.save_df(df, "protein_features.zip", metadata=metadata)

# Load later
df_loaded = M2F.load_df("protein_features.zip")
print(df_loaded.attrs)  # access metadata
```

## Data Flow

1. **Accession Mining** → Extract UniRef/UniClust IDs from HUMAnN gene-families files
2. **UniProtKB Retrieval** → Fetch protein data via batched REST API calls  
3. **Cleaning** → Remove metadata, extract information with regex, normalize text
4. **Feature Engineering** → Convert to numerical representations (embeddings + multi-hot encodings)
5. **Persistence** → Save as compressed, easily-loadable datasets

## Core Components

### Embedders
- **AAChainEmbedder**: ESM-2 embeddings for amino acid sequences
- **FreeTXTEmbedder**: OpenAI embeddings for free-text with intelligent caching

### Encoders  
- **GOEncoder**: Collapse GO terms to fixed depth, encode as integer tuples
- **ECEncoder**: Handle EC number hierarchies with auto-depth selection
- **MultiHotEncoder**: Generic multi-label encoding without dense materialization

### Utilities
- Rate-limited UniProtKB API client
- Regex-based text cleaning with PubMed reference removal
- Zarr-based persistence for heterogeneous data types
- HPC batch processing support

## Advanced Usage

### Custom Text Extraction
Add new regex patterns to `AVAILABLE_EXTRACTION_PATTERNS` in `cleaning_utils.py`:

```python
# Your custom pattern will be automatically applied during cleaning
AVAILABLE_EXTRACTION_PATTERNS["new_field"] = r"your_regex_here"
```

### Different Sequence Models
```python
# Swap in different Hugging Face models
custom_embedder = M2F.AAChainEmbedder(
    model_key="esm2_t30_150M_UR50D",  # larger model
    device="cuda:1",
    representation_layer="last"
)
```

### Batch Processing for HPC
```python
# Process large datasets in chunks, save intermediate results
output_dir = M2F.fetch_save_uniprotkb_batches(
    uniref_ids=large_id_list,
    fields=fields,
    batch_size=10000,  # IDs per file
    save_to_dir="/scratch/protein_batches/"
)
```

## Requirements

- Python 3.11+
- pandas, numpy
- torch (for ESM-2 embeddings)
- openai (for text embeddings)  
- zarr (for data persistence)
- requests (for UniProt API)

## Contributing

M2F is designed to be easily extensible:

- **New ontologies**: Subclass `MultiHotEncoder` with your collapse/encoding logic
- **Different embeddings**: Clone existing embedder classes and swap models
- **Custom serialization**: Follow the `save_df`/`load_df` pattern for new data types

## License

[Add your license here]

## Citation

```bibtex
@misc{mishchyriak2025m2f,
  author = {Yehor Mishchyriak},
  title = {M2F: A practical pipeline for mining UniProt and turning biology into machine-learnable features},
  year = {2025},
  note = {Summer Research Intern, Bonham Lab, Tufts University School of Medicine}
}
```

## Contact

- **Author**: Yehor Mishchyriak (ymishchyriak@wesleyan.edu)
- **Affiliations**: Bonham Lab (Tufts University School of Medicine), Wesleyan University