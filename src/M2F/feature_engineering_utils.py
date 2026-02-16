# builtins:
import re
import os
from typing import Union, List, Tuple, Optional
import logging

# third-party:
import pandas as pd
import numpy as np
import zarr

# local:
from .embedding_utils import (FreeTXTEmbedder,
                              AAChainEmbedder,
                              GOEncoder,
                              ECEncoder,
                              encode_multihot)
from . import util

# *-----------------------------------------------*
#                      GLOBALS
# *-----------------------------------------------*

_logger = logging.getLogger(__name__)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def max_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Select the embedding with the maximum L2 norm from a list of embeddings.

    Args:
        embeddings: A list of numpy arrays representing embedding vectors.

    Returns:
        The embedding (numpy array) with the highest L2 norm.

    Raises:
        ValueError: If the list is empty or if embeddings have differing shapes.
    """
    if not embeddings:
        raise ValueError("Cannot max pool empty list of embeddings")
    
    if len({emb.shape for emb in embeddings}) != 1:
        raise ValueError(f"Embedding shapes differ: {[emb.shape for emb in embeddings]}")

    if len(embeddings) == 1:
        return embeddings[0]
    
    # calculate L2 norm for each embedding
    norms = [np.linalg.norm(emb) for emb in embeddings]
    
    # return the embedding with maximum norm
    max_idx = np.argmax(norms)
    return embeddings[max_idx]

def vals2embs_map(df: pd.DataFrame, col: str, embedder: Union[AAChainEmbedder, FreeTXTEmbedder], batch_size: int):
    """
    Create a mapping from each individual value in a DataFrame column to its embedding.

    Args:
        df: Input DataFrame containing the column to embed.
        col: Name of the column in df whose values are iterables of items to embed.
        embedder: An instance of AAChainEmbedder or FreeTXTEmbedder for generating embeddings.
        batch_size: Number of items to embed per batch call.

    Returns:
        A dictionary mapping each unique item to its embedding (numpy array).
        Returns an empty dict if there are no items to embed.
    """
    unique_vals = df[col].dropna().unique()
    vals = list(dict.fromkeys([item
            for val in unique_vals
            for item in val
            if item
        ])) # flatten
    if not vals:
        return {} 
    val2emb_map = dict(zip(vals, embedder.embed_sequences(vals, batch_size)))
    return val2emb_map

# there's a bunch of harmless warnings caused by zipfile lib,
# and one zarr warning regarding a dtype I used for strings -- also harmless
@util.suppress_warnings(zarr.core.dtype.common.UnstableSpecificationWarning, UserWarning)
def save_df(df: pd.DataFrame, pth: str, metadata: Optional[dict] = None) -> None:
    """
    Persist a heterogeneous DataFrame to a Zarr ZipStore (.zip).

    Supports:
      - ASCII string IDs        → fixed-length byte arrays
      - Ragged int tuples       → flat int32 values + int32 offsets
      - Dense float arrays      → 2D float32 datasets

    Args:
        df: DataFrame to save. Must contain an 'Entry' column of strings.
        pth: Full filename for the output ZipStore (.zip extension is required).
        metadata: Optional dict of key-value attributes to add to the Zarr root.

    Raises:
        ValueError: If a column contains an unsupported dtype; or if 'pth' is not a zip file.
    """
    if not pth.endswith(".zip"):
        raise ValueError(f"'pth' must have .zip extension. Instead, {pth} was given")

    # ---------define-helpers---------
    def encode_strings(strings: np.ndarray) -> Tuple[str, np.ndarray]:
        max_len = max(len(s) for s in strings)
        # dtype='S⟨max_len⟩' == “fixed-length bytes” (1 byte per char)
        dtype = f"S{max_len}"
        data = np.asarray(strings, dtype=dtype)
        return dtype, data

    def flatten_offset(tuples: tuple) -> Tuple[np.ndarray, np.ndarray]:
        flat_vals = np.fromiter((x for tup in tuples for x in tup), dtype='int32')
        lengths = np.array([len(t) for t in tuples], dtype='int32')
        # offsets assume an extra entry at end for slicing convenience
        offsets = np.empty(len(tuples)+1, dtype='int32')
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        return np.array(flat_vals, dtype=np.uint16), np.array(offsets, dtype=np.uint16)
    # --------------------------------

    _logger.info(f"Saving DataFrame with {df.shape[0]} rows and {df.shape[1]} columns to {pth}")

    with zarr.storage.ZipStore(pth, mode="w") as store:
        root = zarr.group(store, overwrite=True)
        
        # -----save-accession-pointers----
        _logger.debug(f"Saving accession numbers")
        df = df.sort_values(by="Entry").copy()
        dtype, data_bytes = encode_strings(df["Entry"].to_numpy())
        root.create_array('accessions', data=data_bytes, compressors=zarr.codecs.BloscCodec(cname="zlib", clevel=3, shuffle=zarr.codecs.BloscShuffle.noshuffle))
        # --------------------------------

        # ---------save-cols-data---------
        for col_name in sorted([col for col in df.columns if col != "Entry"]):
            _logger.debug(f"Saving the {col_name} column data")
            col_storage = root.create_group(col_name)
            notna_mask = ~df[col_name].isna()
            # filter out missing data
            notna_accessions = df["Entry"][notna_mask]
            notna_vals = df[col_name][notna_mask]
            if notna_vals.empty:
                _logger.debug(f"{col_name} has no data -- just empty cells")
                col_storage.attrs["is_empty"] = True
                continue
            else:
                col_storage.attrs["is_empty"] = False

            accession_dtype, data_bytes = encode_strings(notna_accessions.to_numpy())
            col_storage.create_array('accessions', data=data_bytes,
                                    compressors=zarr.codecs.BloscCodec(cname="zlib", clevel=3, shuffle=zarr.codecs.BloscShuffle.noshuffle))

            if isinstance(notna_vals.iloc[0], tuple):
                _logger.debug(f"{col_name} contains tuples: saving as flattened list + offsets")
                flat_vals, offsets = flatten_offset(notna_vals)
                col_storage.create_array('flat_vals', data=flat_vals,
                                        compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=2, shuffle=zarr.codecs.BloscShuffle.bitshuffle))
                col_storage.create_array('offsets', data=offsets,
                                        compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=2, shuffle=zarr.codecs.BloscShuffle.bitshuffle))

            elif isinstance(notna_vals.iloc[0], np.ndarray):
                data = np.vstack(notna_vals, dtype=np.float32)
                _logger.debug(f"{col_name} contains numpy arrays: saving as a vstacked array of shape {data.shape}")
                col_storage.create_array(name="data", data=data,
                                        compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle))

            else:
                _logger.error(f"The dataframe contains unsupported dtype: {type(notna_vals.iloc[0])}. Expected: 'tuple' or 'ndarray'")
                raise ValueError(f"The dataframe contains unsupported dtype: {type(notna_vals.iloc[0])}. Expected: 'tuple' or 'ndarray'")

        # -----------add-metadata---------
        if metadata is None:
            _logger.info(f"Finished saving DataFrame to {pth} (no metadata)")
            return
        for attr, desc in metadata.items():
            root.attrs[attr] = desc
        # --------------------------------
        _logger.info(f"Finished saving DataFrame with metadata to {pth}")

def load_df(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from a Zarr ZipStore (.zip) created by save_df.

    Args:
        path: Path to the .zip file or base name without extension.

    Returns:
        Reconstructed pandas DataFrame with original columns and metadata.

    Raises:
        ValueError: If a column layout in the store is unrecognized.
    """
    if not path.endswith(".zip"):
        path += ".zip"
    _logger.info(f"Loading DataFrame from {path}")
    store = zarr.storage.ZipStore(path, mode="r")
    root = zarr.open_group(store, mode="r")

    def decode_strings(b: np.ndarray) -> np.ndarray:
        return np.char.decode(b, encoding="ascii")

    full_acc = decode_strings(root["accessions"][:])
    df = pd.DataFrame({"Entry": full_acc})

    for col_name in root.group_keys():
        _logger.debug(f"Loading {col_name} data")
        grp = root[col_name]
        is_empty = grp.attrs.get("is_empty")

        if is_empty:
            df[col_name] = np.nan
            continue

        col_acc = decode_strings(grp["accessions"][:])

        if {"flat_vals", "offsets"} <= set(grp.array_keys()):
            flat = grp["flat_vals"][:].astype(int)
            offs = grp["offsets"][:].astype(int)
            values = [
                tuple(flat[offs[i]:offs[i + 1]])
                for i in range(len(col_acc))
            ]

        elif "data" in grp.array_keys():
            data   = grp["data"][:].astype(np.float32)
            values = [row for row in data]

        else:
            raise ValueError(
                f"Unknown layout in column '{col_name}': {grp.array_keys()}"
            )

        mapping = dict(zip(col_acc, values))
        df[col_name] = df["Entry"].map(mapping)

    df.attrs.update(dict(root.attrs))
    df.sort_index(axis=1, inplace=True)
    _logger.info(f"Loaded DataFrame with {df.shape[0]} rows and {df.shape[1]} columns from {path}")

    return df

def empty_tuples_to_NaNs(df: pd.DataFrame, inplace=False) -> pd.DataFrame:
    """
    Replace all empty tuple entries in a DataFrame with NaN values.

    Args:
        df: Input DataFrame.
        inplace: If True, modify the DataFrame in place; otherwise, return a copy.

    Returns:
        DataFrame where every occurrence of an empty tuple is replaced by np.nan.
    """
    if not inplace:
        df = df.copy(deep=True)
    df.loc[:, :] = df.map(lambda x: np.nan if x == () else x)
    return df

# *--------------------------------------------------------*
# cc_domain, cc_function, cc_catalytic_activity, cc_pathway
# *--------------------------------------------------------*

def embed_freetxt_cols(df: pd.DataFrame, cols: List[str], embedder: FreeTXTEmbedder,
                    batch_size: int = 1000, inplace=False) -> pd.DataFrame:
    """
    Embed specified free-text columns of a DataFrame using FreeTXTEmbedder.

    Args:
        df: Input DataFrame.
        cols: List of column names containing iterables of strings to embed.
        embedder: FreeTXTEmbedder instance to generate embeddings.
        batch_size: Batch size for embedding calls.
        inplace: If True, modify df in place; otherwise, return a new DataFrame.

    Returns:
        DataFrame with the specified columns replaced by their max-pooled embeddings.
    """
    if not inplace:
        df = df.copy(deep=True)
    for col in cols:
        embedding_map = vals2embs_map(df, col, embedder, batch_size)
        df.loc[:, col] = df.loc[:, col].map(lambda entry: max_pool([embedding_map[s] for s in entry])
                                            if entry else np.nan)
    return df

# *-----------------------------------------------*
#               ft_domain, sequence
# *-----------------------------------------------*

def _domain_aa_ranges(domains: Tuple[str]) -> List[Tuple[int,int]]:
    """
    Parse domain range strings of the form 'start..end' into 0-based index tuples.

    Args:
        domains: Tuple of domain range strings (e.g., ('5..10', '20..30')).

    Returns:
        List of (start_index, end_index) tuples where start is inclusive and end is exclusive.
    """
    ranges = []
    for d in domains:
        m = re.match(r"^(\d+)\.\.(\d+)$", d)
        if not m:
            continue
        start, end = map(int, m.groups())
        ranges.append((start - 1, end))
    return ranges

def _get_domain_sequences(domains: Tuple[str, ...], full_seq: Tuple[str]) -> List[str]:
    """
    Extract subsequences of a full amino acid sequence for given domain ranges.

    Args:
        domains: Tuple of domain range strings.
        full_seq: Tuple containing the full sequence as its first element.

    Returns:
        List of substring sequences for each valid domain range.
    """
    # Note: need to do full_seq[0] because it's a singleton tuple
    return [full_seq[0][s:e] for s, e in _domain_aa_ranges(domains)]

def embed_ft_domains(df: pd.DataFrame, embedder: AAChainEmbedder,
                     batch_size: int = 128, inplace: bool = False) -> pd.DataFrame:
    """
    Embed domain-specific subsequences of protein chains using AAChainEmbedder.

    Extracts domain sequences from 'Domain [FT]' and 'Sequence' columns,
    computes embeddings for each domain, and replaces 'Domain [FT]' with the max-pooled embedding.

    Args:
        df: Input DataFrame.
        embedder: AAChainEmbedder instance for generating embeddings.
        batch_size: Batch size for embedding calls.
        inplace: If True, modify df in place; otherwise, return a copy.

    Returns:
        DataFrame with the 'Domain [FT]' column replaced by embeddings.
    """
    if not inplace:
        df = df.copy(deep=True)

    df.loc[:, "tmp_domain_seqs"] = df.apply(
        lambda row: tuple(_get_domain_sequences(row["Domain [FT]"], row["Sequence"])), axis=1
    )

    embedding_map = vals2embs_map(df, "tmp_domain_seqs", embedder, batch_size)

    df.loc[:, "Domain [FT]"] = df["tmp_domain_seqs"].map(
        lambda entry: max_pool([embedding_map[s] for s in entry]) if entry else np.nan
    )
    df.drop(columns="tmp_domain_seqs", inplace=True)
    
    return df

def embed_AAsequences(df: pd.DataFrame, embedder: AAChainEmbedder,
                      batch_size: int = 128, inplace: bool = False) -> pd.DataFrame:
    """
    Embed full amino acid sequences in the DataFrame using AAChainEmbedder.

    Args:
        df: Input DataFrame with a 'Sequence' column.
        embedder: AAChainEmbedder instance for generating embeddings.
        batch_size: Batch size for embedding calls.
        inplace: If True, modify df in place; otherwise, return a copy.

    Returns:
        DataFrame with the 'Sequence' column replaced by embedding arrays.
    """
    if not inplace:
        df = df.copy(deep=True)

    embedding_map = vals2embs_map(df, "Sequence", embedder, batch_size)
    df.loc[:, "Sequence"] = df["Sequence"].map(lambda s: embedding_map[s[0]] if s else np.nan)

    return df

# *-----------------------------------------------*
#                   go_mf & go_bp
# *-----------------------------------------------*

_go_enc = GOEncoder(os.path.join(SCRIPT_DIR, "dependencies", "go-basic.obo"))
encode_go = _go_enc.encode_go

# *-----------------------------------------------*
#                       ec
# *-----------------------------------------------*

_ec_enc = ECEncoder()
encode_ec = _ec_enc.encode_ec


__all__ = [
    "embed_ft_domains",
    "embed_AAsequences",
    "embed_freetxt_cols",
    "empty_tuples_to_NaNs",
    "encode_go",
    "encode_ec",
    "encode_multihot",
    "save_df",
    "load_df"
]

if __name__ == "__main__":
    pass
