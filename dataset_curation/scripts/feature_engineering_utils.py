import re
import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict
from .embedding_utils import FreeTXTEmbedder, AAChainEmbedder, EncodeGO

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*


# *-----------------------------------------------*
# ft_domain <-- embedding all the rows in a df
# *-----------------------------------------------*

# HELPERS FOR DOMAIN SLICING:

def _extract_aa_ranges(domain_entry: Union[str, Tuple[str, ...]]) -> List[Tuple[int,int]]:
    """
    ("54..144", "224..288") → [(53,144), (223,288)]
    """

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
    """
    Pull out the substring for each domain range
    """
    ranges = _extract_aa_ranges(domain_entry)
    return [full_seq[s:e] for s, e in ranges]

# embed domains' AA sequences
def _pool_domain_embeddings(seqs: List[str], embedder: AAChainEmbedder) -> Union[np.ndarray, float]:
    if not seqs:
        # if no annotated domains, return a zero vector
        return np.nan
    embs = [embedder.embed_sequence(s) for s in seqs] # list of [hidden_dim]
    
    # stack into shape [N, hidden_dim] -> mean‑pool over the first axis -> [hidden_dim]
    return embs[0] if len(embs) == 1 else embs

# apply to the whole 'Domain [FT]' col
def process_ft_domain(df: pd.DataFrame, embedder: AAChainEmbedder, drop_redundant_cols: bool = True, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy(deep=True)

    # build list of domain sequences, with empty list for NaN
    def extract_or_empty(row):
        dom = row["Domain [FT]"]
        if pd.isna(dom):
            return []
        return _get_domain_sequences(dom, row["Sequence"])

    df["tmp_domain_seqs"] = df.apply(extract_or_empty, axis=1)

    # embed + pool into one vector
    df["domain_embedding"] = df["tmp_domain_seqs"].apply(_pool_domain_embeddings, embedder=embedder)

    # drop raw columns if desired
    if drop_redundant_cols:
        df = df.drop(columns=["Sequence", "tmp_domain_seqs", "Domain [FT]"])

    return df

# *-----------------------------------------------*
# cc_domain, cc_function, cc_catalytic_activity
# ^^^ unique values extraction, embedding, then
# mapping embeddings to values from the df.
# It's less costly (fewer API requests)
# *-----------------------------------------------*

def unique_vals2embs_map(df: pd.DataFrame, col: str, embedder: FreeTXTEmbedder):

    unique_vals = df[col].dropna().unique()
    vals = [item
            for val in unique_vals
            for item in ((val,) if not isinstance(val, tuple) else val)
        ] # flatten

    val2emb_map = dict(zip(vals, embedder.request_embedding_for(vals)))

    return val2emb_map

# note: this will fail for tuples
# and also, I am not sure yet what to do with tuples of text.
# I think I may need to embed multiple functional annotation fields at once, because if I embed them separately,
# then I am not sure how to combine them, though max-pooling seems like the easiest approach (averaging will lose
# the meanings in case those functional annotations discuss distinct domains with distinct functions).
# Ideally, I would use attention for these and let the training decide what weights to use and then do "weighted-average-pooling".
def embed_freetxt_cols(df: pd.DataFrame, cols: List[str],
                    embedding_map: Dict[str, np.ndarray], inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy(deep=True)

    for col in cols:
        df[col] = df[col].map(embedding_map)

    return df


# *-----------------------------------------------*
#                   go_f & go_p
# *-----------------------------------------------*



# *-----------------------------------------------*
#                       ec
# *-----------------------------------------------*



# *-----------------------------------------------*
#                    cc_pathway
# *-----------------------------------------------*



# *-----------------------------------------------*
#                     rhea
# *-----------------------------------------------*



# *-----------------------------------------------*
#                   cc_cofactor
# *-----------------------------------------------*



# *-----------------------------------------------*
#                    sequence
# *-----------------------------------------------*



# *-----------------------------------------------*
#                      API
# *-----------------------------------------------*

# BTW:
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# df.apply(fn, axis=1) → “I need to look at the whole row.”
# series.apply(fn) → “I need to call this function on each element of the series (and I have extra args).”
# series.map(fn or dict) → “I need a simple one‑to‑one mapping on this series.”
# df.applymap(fn) → “I need to change every single cell with the same function.”
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
