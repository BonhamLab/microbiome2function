import re
import pandas as pd
import torch
from typing import Union, List, Tuple
from transformers import T5EncoderModel, AutoTokenizer

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

# *-----------------------------------------------*
#                    ft_domain
# *-----------------------------------------------*

# CONFIGURE PROT-T5:

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
model      = T5EncoderModel.from_pretrained(MODEL_NAME)
model.eval() # <-- eval mode

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

@torch.no_grad()
def _embed_sequence(seq: str) -> torch.Tensor:
    """
    Turn one amino-acid string into a ProtT5 embedding vector of shape [D]
    """
    spaced = " ".join(seq) # ProtT5 expects spaces between letters
    tokens = tokenizer(spaced, return_tensors="pt")
    out = model(**tokens).last_hidden_state # [1, L, D]
    residues = out[0, 1:-1, :] # drop special tokens → [L-2, D]
    return residues.mean(dim=0) # pool over length → [D]

def _pool_domain_embeddings(seqs: List[str]) -> torch.Tensor:
    """
    Given N domain sequences, embed each and then mean-pool → [D]
    """
    if not seqs:
        # if no annotated domains, return a zero vector
        return torch.zeros(model.config.d_model)
    embs = [_embed_sequence(s) for s in seqs]    # list of [D]
    return torch.stack(embs, dim=0).mean(dim=0)  # [D]

# DATAFRAME PROCESSOR:

def process_ft_domain(df: pd.DataFrame, drop_redundant_cols: bool = True) -> pd.DataFrame:
    df = df.copy(deep=True)

    # build list of domain sequences, with empty list for NaN
    def extract_or_empty(row):
        dom = row["Domain [FT]"]
        if pd.isna(dom):
            return []
        return _get_domain_sequences(dom, row["Sequence"])

    df["__domain_seqs"] = df.apply(extract_or_empty, axis=1)

    # embed + pool into one vector
    df["domain_embedding"] = df["__domain_seqs"].apply(_pool_domain_embeddings)

    # drop raw columns if desired
    if drop_redundant_cols:
        df = df.drop(columns=["Sequence", "__domain_seqs", "Domain [FT]"])

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
