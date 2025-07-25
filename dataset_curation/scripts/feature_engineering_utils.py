import re
import os
import pandas as pd
import numpy as np
from typing import Union, List, Tuple
import torch

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*



# *-----------------------------------------------*
#                    ft_domain
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

# EMBEDDING FUNCTIONS:

# NOTE: really need to implement batching for this!
@torch.no_grad()
def _embed_sequence(seq: str, model, tokenizer) -> np.ndarray:
    """
    Turn one amino-acid string into a ProtT5 embedding vector of shape [D]
    """
    spaced = " ".join(seq) # ProtT5 expects spaces between letters
    tokens = tokenizer(spaced, return_tensors="pt")
    out = model(**tokens).last_hidden_state # (batch=1, seq_len, hidden_dim), so (seq_len, hidden_dim) basically
    residues = out[0, 1:-1, :] # drop special tokens -> (seq_len-2, hidden_dim)
    emb_tensor = residues.mean(dim=0) # pool over length: (seq_len-2, hidden_dim) -> (hidden_dim)

    return emb_tensor.cpu().numpy()

def _pool_domain_embeddings(seqs: List[str], model, tokenizer) -> Union[np.ndarray, float]:
    """
    Given N domain sequences, embed each and then mean-pool → [D]
    """
    if not seqs:
        # if no annotated domains, return a zero vector
        return np.nan
    embs = [_embed_sequence(s, model, tokenizer) for s in seqs] # list of [hidden_dim]
    
    # stack into shape [N, hidden_dim] -> mean‑pool over the first axis -> [hidden_dim]
    return np.vstack(embs).mean(axis=0)

# DATAFRAME PROCESSOR:

def process_ft_domain(df: pd.DataFrame, model, tokenizer, drop_redundant_cols: bool = True) -> pd.DataFrame:
    df = df.copy(deep=True)

    # build list of domain sequences, with empty list for NaN
    def extract_or_empty(row):
        dom = row["Domain [FT]"]
        if pd.isna(dom):
            return []
        return _get_domain_sequences(dom, row["Sequence"])

    df["tmp_domain_seqs"] = df.apply(extract_or_empty, axis=1)

    # embed + pool into one vector
    df["domain_embedding"] = df["tmp_domain_seqs"].apply(_pool_domain_embeddings, model=model, tokenizer=tokenizer)

    # drop raw columns if desired
    if drop_redundant_cols:
        df = df.drop(columns=["Sequence", "tmp_domain_seqs", "Domain [FT]"])

    return df

# *-----------------------------------------------*
#                    cc_domain
# *-----------------------------------------------*



# *-----------------------------------------------*
#                 protein_families
# *-----------------------------------------------*



# *-----------------------------------------------*
#                      go_f
# *-----------------------------------------------*



# *-----------------------------------------------*
#                      go_p
# *-----------------------------------------------*



# *-----------------------------------------------*
#                    cc_function
# *-----------------------------------------------*



# *-----------------------------------------------*
#               cc_catalytic_activity
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
