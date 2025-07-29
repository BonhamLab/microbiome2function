import re
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict
from .embedding_utils import (FreeTXTEmbedder,
                              AAChainEmbedder,
                              GOEncoder,
                              ECEncoder,
                              MultiHotEncoder)

__all__ = list()
def export(func):
    __all__.append(func.__name__)
    return func 

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def max_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Cannot max pool empty list of embeddings")
    
    if len(embeddings) == 1:
        return embeddings[0]
    
    # calculate L2 norm for each embedding
    norms = [np.linalg.norm(emb) for emb in embeddings]
    
    # return the embedding with maximum norm
    max_idx = np.argmax(norms)
    return embeddings[max_idx]

# *-----------------------------------------------*
#                   ft_domain
# *-----------------------------------------------*

def _domain_aa_ranges(domain_entry: Union[str, Tuple[str, ...]]) -> List[Tuple[int,int]]:

    if isinstance(domain_entry, str):
        entries = [domain_entry]
    else:
        entries = domain_entry

    ranges = []
    for ent in entries:
        m = re.match(r"^(\d+)\.\.(\d+)$", ent)
        if not m:
            continue
        start, end = map(int, m.groups())
        ranges.append((start - 1, end))
    return ranges

def _get_domain_sequences(domain_entry: str, full_seq: str) -> List[str]:
    ranges = _domain_aa_ranges(domain_entry)
    return [full_seq[s:e] for s, e in ranges]

def _pool_domain_embeddings(seqs: List[str], embedder: AAChainEmbedder) -> Union[np.ndarray, float]:
    if not seqs:
        # if no annotated domains, return NaN
        return np.nan
    
    # embed each domain sequence
    embs = embedder.embed_sequences(seqs)  # list of [hidden_dim]
    
    if len(embs) == 1:
        return embs[0]
    else:
        # pool the embedding with largest L2 norm
        return max_pool(embs)

@export
def embed_ft_domains(df: pd.DataFrame, embedder: AAChainEmbedder, drop_redundant_cols: bool = True, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy(deep=True)
    
    # build list of domain sequences, with empty list for NaN
    def extract_or_empty(row):
        dom = row["Domain [FT]"]
        if pd.isna(dom):
            return []
        return _get_domain_sequences(dom, row["Sequence"])
    
    df["tmp_domain_seqs"] = df.apply(extract_or_empty, axis=1)
    
    # embed + max pool into one vector
    df["domain_embedding"] = df["tmp_domain_seqs"].apply(_pool_domain_embeddings, embedder=embedder)
    
    # drop raw columns if desired
    if drop_redundant_cols:
        df = df.drop(columns=["Sequence", "tmp_domain_seqs", "Domain [FT]"])
    
    return df

# *--------------------------------------------------------*
# cc_domain, cc_function, cc_catalytic_activity, cc_pathway
# *--------------------------------------------------------*

def unique_vals2embs_map(df: pd.DataFrame, col: str, embedder: FreeTXTEmbedder):

    unique_vals = df[col].dropna().unique()
    vals = [item
            for val in unique_vals
            for item in ((val,) if not isinstance(val, tuple) else val)
        ] # flatten

    val2emb_map = dict(zip(vals, embedder.request_embedding_for(vals)))

    return val2emb_map

@export
def embed_freetxt_cols(df: pd.DataFrame, cols: List[str], embedder: FreeTXTEmbedder, inplace=False) -> pd.DataFrame:
    
    if not inplace:
        df = df.copy(deep=True)

    for col in cols:
        embedding_map = unique_vals2embs_map(df, col, embedder)
        df[col] = df[col].map(lambda entry: max_pool([embedding_map(s) for s in entry]))

    return df

# *-----------------------------------------------*
#                   go_mf & go_bp
# *-----------------------------------------------*

_go_enc = GOEncoder()
encode_go = _go_enc.process_go
export(encode_go)

# *-----------------------------------------------*
#                       ec
# *-----------------------------------------------*

_ec_enc = ECEncoder()
encode_ec = _ec_enc.process_ec
export(encode_ec)

# *-----------------------------------------------*
#                     rhea
# *-----------------------------------------------*

_rhea_enc = MultiHotEncoder()
encode_rhea = _rhea_enc.encode
export(encode_rhea)

# *-----------------------------------------------*
#                   cc_cofactor
# *-----------------------------------------------*

_cofactor_enc = MultiHotEncoder()
encode_cofactor = _cofactor_enc.encode
export(encode_cofactor)


if __name__ == "__main__":
    pass
