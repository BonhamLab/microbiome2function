"""Top-level public API for the M2F package."""

# logging
from .logging_utils import configure_logging

# mining
from .mining_utils import (
    extract_accessions_from_humann,
    extract_all_accessions_from_dir,
    fetch_uniprotkb_fields,
    fetch_save_uniprotkb_batches,
)

# cleaning
from .cleaning_utils import clean_col, clean_cols

# embedding / encoding
from .embedding_utils import (
    AAChainEmbedder,
    FreeTXTEmbedder,
    MultiHotEncoder,
    GOEncoder,
    ECEncoder,
    encode_multihot,
    get_GODag,
)

# feature engineering / persistence
from .feature_engineering_utils import (
    embed_ft_domains,
    embed_AAsequences,
    embed_freetxt_cols,
    encode_go,
    encode_ec,
    empty_tuples_to_NaNs,
    save_df,
    load_df,
)

# models
from .ffnn import FFNN
from .gnn import GraphConv, GraphConvNodeClassifier, GATNodeClassifier

# metrics
from .testing_utils import accuracy, recall, precision, f1

# dataset interfaces
from .pyg_data_interfaces import (
    DatasetInput,
    build_topology_from_DatasetInput,
    build_features_from_DatasetInput,
    ProteinGraphInMemoryDataset,
    ProteinGraphOnDiskDataset,
    ProteinDataset,
)

# utility module namespace
from . import util


__all__ = [
    # logging
    "configure_logging",
    # mining
    "extract_accessions_from_humann",
    "extract_all_accessions_from_dir",
    "fetch_uniprotkb_fields",
    "fetch_save_uniprotkb_batches",
    # cleaning
    "clean_col",
    "clean_cols",
    # embedding / encoding
    "AAChainEmbedder",
    "FreeTXTEmbedder",
    "MultiHotEncoder",
    "GOEncoder",
    "ECEncoder",
    "encode_multihot",
    "get_GODag",
    # feature engineering / persistence
    "embed_ft_domains",
    "embed_AAsequences",
    "embed_freetxt_cols",
    "encode_go",
    "encode_ec",
    "empty_tuples_to_NaNs",
    "save_df",
    "load_df",
    # models
    "FFNN",
    "GraphConv",
    "GraphConvNodeClassifier",
    "GATNodeClassifier",
    # metrics
    "accuracy",
    "recall",
    "precision",
    "f1",
    # dataset interfaces
    "DatasetInput",
    "build_topology_from_DatasetInput",
    "build_features_from_DatasetInput",
    "ProteinGraphInMemoryDataset",
    "ProteinGraphOnDiskDataset",
    "ProteinDataset",
    # utility module namespace
    "util"
]
