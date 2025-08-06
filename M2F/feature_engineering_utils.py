# builtins:
import re
import os
from typing import Union, List, Tuple

# third-party:
import pandas as pd
import numpy as np
import zarr
from .embedding_utils import (FreeTXTEmbedder,
                              AAChainEmbedder,
                              GOEncoder,
                              ECEncoder,
                              encode_multihot)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def max_pool(embeddings: List[np.ndarray]) -> np.ndarray:
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
    unique_vals = df[col].dropna().unique()
    vals = [item
            for val in unique_vals
            for item in val
            if item
        ] # flatten
    if not vals:
        return {} 
    val2emb_map = dict(zip(vals, embedder.embed_sequences(vals, batch_size)))
    return val2emb_map

def save_df(df: pd.DataFrame, zarr_file: str) -> None:
    root = zarr.group(zarr_file, overwrite=True)
    for col in df.columns:
       isna_mask = df[col].isna()
       notna_mask = ~isna_mask
       isna_accessions = df.index[isna_mask].sort()
       notna_accessions = df.index[notna_mask].sort()
       vecs = notna_accessions.to_list()

def load_df(zarr_file: str) -> pd.DataFrame:
    pass

def empty_tuples_to_NaNs(df: pd.DataFrame, inplace=False) -> None:
    if not inplace:
        df = df.copy(deep=True)
    df.loc[:, :] = df.map(lambda x: np.nan if x == () else x)
    return df

# *--------------------------------------------------------*
# cc_domain, cc_function, cc_catalytic_activity, cc_pathway
# *--------------------------------------------------------*

def embed_freetxt_cols(df: pd.DataFrame, cols: List[str], embedder: FreeTXTEmbedder,
                    batch_size: int = 1000, inplace=False) -> pd.DataFrame:
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
    ranges = []
    for d in domains:
        m = re.match(r"^(\d+)\.\.(\d+)$", d)
        if not m:
            continue
        start, end = map(int, m.groups())
        ranges.append((start - 1, end))
    return ranges

def _get_domain_sequences(domains: Tuple[str, ...], full_seq: Tuple[str]) -> List[str]:
    # Note: need to do full_seq[0] because it's a singleton tuple
    return [full_seq[0][s:e] for s, e in _domain_aa_ranges(domains)]

def embed_ft_domains(df: pd.DataFrame, embedder: AAChainEmbedder,
                     batch_size: int = 128, inplace: bool = False) -> pd.DataFrame:
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
