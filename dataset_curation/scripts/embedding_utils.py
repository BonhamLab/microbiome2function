from openai import OpenAI
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
import os
from typing import List, Union, Tuple
import atexit
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from goatools.obo_parser import GODag


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                           DENSE EMBEDDINGS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class AAChainEmbedder:
    
    HF_MODELS = {
        "ESM2":   "facebook/esm2_t6_8M_UR50D", # light
        "PROTT5": "Rostlab/prot_t5_xl_uniref50" # heavy
    }

    def __init__(self, model_key: str, device: str = "cpu"):
        if model_key not in self.HF_MODELS:
            raise ValueError(f"model_key must be one of {list(self.HF_MODELS)}")

        self.repo_id   = self.HF_MODELS[model_key]
        self.is_prott5 = (model_key == "PROTT5")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id, use_fast=False, legacy=False
        )
        self.model = AutoModel.from_pretrained(self.repo_id).to(device).eval()

    @torch.no_grad()
    def embed_sequence(self, seq: str) -> np.ndarray:
        if self.is_prott5:
            seq = " ".join(seq) # ProtT5 expects spaces

        toks = self.tokenizer(seq, return_tensors="pt").to(self.model.device)
        hidden = self.model(**toks).last_hidden_state # (batch=1, seq_len, hidden_dim), so (seq_len, hidden_dim) basically
        # |  drop special tokens -> (seq_len-2, hidden_dim)
        # v  and pool over residues: (seq_len-2, hidden_dim) -> (hidden_dim)
        emb = hidden[0, 1:-1].mean(dim=0) 
        return emb.cpu().numpy()


# NOTE: this one is not parallelism-safe yet (I didn't bother including threading here as I didn't need to)
# But, if one ever does, be sure to use locks so that a thread writing to the cache acquires a lock and then releases it.
# Another thing:
# We load cache once, buffer in a dict, then write exactly one Parquet file at exit;
# The reason we don't use .csv is because it'd serialize embeddings into strings resulting into huge files.
# By the way, that is why for the final dataset, we'll either use .csv storing paths to .npy's, or parquet again.
# If curios: https://parquet.apache.org/; https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html.

class FreeTXTEmbedder:
    MODELS = {
        "SMALL_OPENAI_MODEL": "text-embedding-3-small",
        "LARGE_OPENAI_MODEL": "text-embedding-3-large"
    }
    CACHING_MODES = {"NOT_CACHING", "APPEND", "CREATE/OVERRIDE"}

    def __init__(
        self, api_key: str, model: str,
        cache_file_path: Union[str, None] = None,
        caching_mode: str = "NOT_CACHING",
    ):

        if model not in self.MODELS:
            raise ValueError(f"model must be one of {list(self.MODELS)}")
        if caching_mode not in self.CACHING_MODES:
            raise ValueError(f"caching_mode must be one of {self.CACHING_MODES}")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_file_path = cache_file_path
        self.caching_mode = caching_mode

        self.cache_map: dict[str, np.ndarray] = {}
        if cache_file_path and caching_mode == "APPEND" and os.path.exists(cache_file_path):
            if os.path.getsize(cache_file_path) > 1_000_000_000:
                raise RuntimeError("Cache >1 GB - refuse to load")
            df = pd.read_parquet(cache_file_path)
            self.cache_map = {t: np.array(e) for t, e in zip(df["text"], df["embedding"])}

        if cache_file_path and caching_mode != "NOT_CACHING":
            atexit.register(self._save_cache)

    def _save_cache(self):
        if not (self.cache_file_path and self.caching_mode != "NOT_CACHING"):
            return
        df = pd.DataFrame(
            {"text": list(self.cache_map.keys()),
             "embedding": [e.tolist() for e in self.cache_map.values()]}
        )
        df.to_parquet(self.cache_file_path, index=False, compression="snappy")

    def _lookup(self, s: str):
        return self.cache_map.get(s)

    def _store(self, s: str, emb: np.ndarray):
        if self.caching_mode != "NOT_CACHING":
            self.cache_map[s] = emb

    def request_embedding_for(self, inp: Union[str, List[str]], batch_size: int = 1000):
        if isinstance(inp, str):
            inp = [inp]

        out = [None] * len(inp)
        to_send, idx = [], []

        for i, s in enumerate(inp):
            cached = self._lookup(s)
            if cached is not None:
                out[i] = cached
            else:
                to_send.append(s); idx.append(i)

        for start in range(0, len(to_send), batch_size):
            chunk  = to_send[start:start+batch_size]
            ichunk = idx[start:start+batch_size]
            resp   = self.client.embeddings.create(input=chunk, model=self.model)
            for s, i, emb in zip(chunk, ichunk, resp.data):
                vec = np.asarray(emb.embedding, dtype=np.float32)
                out[i] = vec
                self._store(s, vec)

        return out[0] if len(out) == 1 else out

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                           Multi-hot Encodings
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class MultiHotEncoder:
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def encode(self, sequences: pd.Series):
        sequences = sequences.fillna(tuple())
        enc = self.mlb.fit_transform(sequences)
        return {"encodings": enc, "class_labels": {c: i for i, c in enumerate(self.mlb.classes_)}}

class EncodeGO(MultiHotEncoder):
    def __init__(self, obo_path: Union[str, None] = None):
        super().__init__()
        obo_path = obo_path if obo_path is not None else "dataset_curation/dependencies/go-basic.obo"
        if not os.path.exists(obo_path):
            raise FileNotFoundError(f"OBO not found: {obo_path}")
        self.godag = GODag(obo_path)

    def _collapse_to_depth(self, go_ids, k: int) -> Tuple[str, ...]:
        kept = set()
        for gid in go_ids:
            if gid not in self.godag:
                continue
            node = self.godag[gid]
            ancestors  = set(gid, *node.get_all_parents())
            at_k = {n for n in ancestors if self.godag[n].depth == k}
            kept.update(at_k if at_k else {min(ancestors, key=lambda x: self.godag[x].depth)})
        return tuple(kept)

    def process_go(self, df: pd.DataFrame, col: str, depth: int, inplace=False):
        df = df if inplace else df.copy()
        collapsed = df[col].map(lambda terms: self._collapse_to_depth(terms, depth))
        enc_info = self.encode(collapsed)
        df[col] = enc_info["encodings"].tolist()
        return df, enc_info["class_labels"]


if __name__ == "__main__":
    pass
