# builtins:
import os
import sys
import logging
from typing import List, Union, Tuple, Dict
import sqlite3
import atexit
from collections import OrderedDict

# third-party:
import torch
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from goatools.obo_parser import GODag
import psutil

# *-----------------------------------------------*
#                      GLOBALS
# *-----------------------------------------------*

_logger = logging.getLogger(__name__)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                           DENSE EMBEDDINGS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=	
class AAChainEmbedder:
    """
    Mean-pooled ESM-2 sequence encoder.

    Wraps any Facebook ESM-2 checkpoint and returns a single
    float32 vector per amino-acid chain.

    Parameters
    ----------
    model_key : str
        Key from `HF_MODELS`.
    device : str
        Anything accepted by ``torch.device`` (e.g. ``"cuda:0"`` or ``"cpu"``).
    dtype : torch.dtype | None
        Precision override for model weights. ``None`` keeps HF default.
    representation_layer : {'last', 'second_to_last', int}
        Which hidden layer to pool.
    """

    HF_MODELS: Dict[str, str] = {
        "esm2_t48_15B_UR50D":  "facebook/esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D":   "facebook/esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D": "facebook/esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D": "facebook/esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D":  "facebook/esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D":    "facebook/esm2_t6_8M_UR50D",
    }

    def __init__(
        self,
        model_key: str = "esm2_t6_8M_UR50D",
        device: str = "cpu",
        dtype: Union[torch.dtype, None] = None,
        representation_layer: Union[int, str] = "second_to_last",
    ):
        if model_key not in self.HF_MODELS:
            raise ValueError(f"`model_key` must be one of {list(self.HF_MODELS)}")

        self.repo_id = self.HF_MODELS[model_key]

        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, use_fast=False)
        self.model = AutoModel.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None
        )
        if device.startswith("cuda"):                    
            self.model.to(torch.device(device))          
        else:
            self.model.to(device)
        self.model.eval()

        n_layers = self.model.config.num_hidden_layers
        self.repr_idx = (
            n_layers - 1 if representation_layer == "last" else
            n_layers - 2 if representation_layer == "second_to_last" else
            int(representation_layer)
        )
        if not (0 <= self.repr_idx < n_layers):
            raise ValueError(f"representation_layer must be in [0,{n_layers-1}] or a valid alias")

    # ------------------------------------------------------------
    @torch.no_grad()
    def embed_sequences(self, seqs: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Return a mean-pooled embedding for every sequence.

        Parameters
        ----------
        seqs : list[str]
            Raw amino-acid strings (no BOS/EOS).
        batch_size : int, default 32
            Forward-pass chunk size.

        Returns
        -------
        list[np.ndarray]
            One ``(hidden_dim,)`` float32 vector per input sequence,
            always on CPU for downstream neutrality.
        """
        if not seqs:
            return []

        out: list[np.ndarray] = []
        max_len = self.model.config.max_position_embeddings

        for start in range(0, len(seqs), batch_size):
            chunk = seqs[start : start + batch_size]

            for s in chunk:
                if len(s) > max_len - 2:   # minus BOS/EOS
                    _logger.warning(
                        f"Sequence length {len(s)} exceeds {max_len-2}; truncating."
                    )

            toks = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                return_special_tokens_mask=True, # Adds a mask (1 = BOS/EOS, 0 = residue/PAD)
            ).to(self.model.device)

            # toks:
            # input_ids            [B, L]
            # attention_mask       [B, L]  (1 for real tokens (residues + BOS/EOS), 0 for PAD)
            # special_tokens_mask  [B, L]  (1 for BOS/EOS/<mask>, 0 otherwise)
            spec_mask = toks.pop("special_tokens_mask") 
            hidden_states = self.model(
                **toks, output_hidden_states=True # returns all layer outputs.
            ).hidden_states[self.repr_idx]  # [B, L, D]

            # ^^^ B = num sequences; L = num tokens per seq (AAs + BOS/EOS); D = dims per embedding

            # build a mask that excludes specials AND padding
            attn_mask = toks["attention_mask"].bool()
            spec_mask = spec_mask.bool()
            keep_mask = attn_mask & (~spec_mask) # attn_mask AND NOT(spec_mask)

            # pool
            for i in range(hidden_states.size(0)):
                emb = hidden_states[i][keep_mask[i]].mean(0)
                out.append(emb.to(dtype=torch.float32, device="cpu").numpy())

        return out


class FreeTXTEmbedder:
    """
    Text-embedding wrapper with disk-backed and in-RAM LRU cache.

    Uses OpenAI's `/embeddings` endpoint and stores vectors in
    SQLite (`embeddings` table), with a small in-RAM LRU front-cache
    to minimise latency.

    Parameters
    ----------
    api_key : str
        OpenAI API key.
    model : {'SMALL_OPENAI_MODEL', 'LARGE_OPENAI_MODEL'}
        Friendly key resolved via `MODELS`.
    cache_file_path : str | None
        Path to the SQLite cache file. If ``None`` no on-disk cache is used.
    caching_mode : {'NOT_CACHING','APPEND','CREATE/OVERRIDE'}
        * ``NOT_CACHING`` - disable both SQLite + LRU  
        * ``APPEND`` - keep existing DB and add new rows  
        * ``CREATE/OVERRIDE`` - recreate DB from scratch.
    max_cache_size_kb : int
        RAM budget for the LRU (approximate; rows measured in KB).
    """

    MODELS = {
        "SMALL_OPENAI_MODEL": "text-embedding-3-small",
        "LARGE_OPENAI_MODEL": "text-embedding-3-large"
    }
    CACHING_MODES = {"NOT_CACHING", "APPEND", "CREATE/OVERRIDE"}

    def __init__(
        self, api_key: str, model: str,
        cache_file_path: Union[str, None] = None,
        caching_mode: str = "NOT_CACHING", max_cache_size_kb: int = 1000
    ):

        if model not in self.MODELS:
            raise ValueError(f"model must be one of {list(self.MODELS)}")
        if caching_mode not in self.CACHING_MODES:
            raise ValueError(f"caching_mode must be one of {self.CACHING_MODES}")
        if not (0 < max_cache_size_kb):
            raise ValueError(f"Invalid 'max_cache_size_kb' value: expected a positive integer"
                             f"received {max_cache_size_kb}")

        # -DB-----------------------------------------------
        if cache_file_path and caching_mode != "NOT_CACHING":
            self._conn = sqlite3.connect(cache_file_path)
            self._db = self._conn.cursor()
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text TEXT PRIMARY KEY,
                    vec BLOB 
                )
            """)
            self._conn.commit()
            atexit.register(self._conn.close)
            self._LRU_cache = OrderedDict()
        else:
            self._db = None
            self._LRU_cache = None
        self.cache_file_path = cache_file_path
        self.caching_mode = caching_mode
        self.max_cache_size_kb = max_cache_size_kb
        # --------------------------------------------------

        self.client = OpenAI(api_key=api_key)
        self.model = FreeTXTEmbedder.MODELS[model]
        self._LRU_cache_size_kb = 0

    # ---------PRIVATE-----------
    def __db_lookup(self, s: str):
        self._db.execute("""
                SELECT vec FROM embeddings
                WHERE text = :text;
            """, {"text": s})
        return self._db.fetchone()

    def __LRU_lookup(self, s: str):
        return self._LRU_cache.get(s)

    @staticmethod
    def __row_size_kb(s: str, emb: np.ndarray):
        return (sys.getsizeof(s) + emb.nbytes) / 1024

    def __update_LRU_cache_size(self, s: str, emb: np.ndarray, *, how: str):
        if how == "ADD":
            self._LRU_cache_size_kb += self.__row_size_kb(s, emb)
            return
        elif how == "DEL":
            self._LRU_cache_size_kb -= self.__row_size_kb(s, emb)
            return
        else:
            raise ValueError(f"Invalid 'how' argument was passed. Expected 'ADD' or 'DEL', but {how} was given")

    def __store_in_DB(self, s: str, emb: np.ndarray):
        self._db.execute("""
            INSERT OR REPLACE INTO embeddings VALUES (:text, :vec)
            """, {"text": s, "vec": emb.tobytes()})
        self._conn.commit()

    def __store_in_LRU(self, s: str, emb: np.ndarray):
        self._LRU_cache[s] = emb

    # ---------PROTECTED---------
    def _lookup(self, s: str):
        out = self.__LRU_lookup(s) if self._LRU_cache is not None else None
        if out is not None:
            self._LRU_cache.move_to_end(s)
            return out
        if self._db is None:
            return None
        row = self.__db_lookup(s)
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def _store(self, s: str, emb: np.ndarray):
        if self.caching_mode == "NOT_CACHING":
            return
        item_size = self.__row_size_kb(s, emb)
        if (self._LRU_cache_size_kb + item_size) > self.max_cache_size_kb:
            try:
                old_s, old_emb = self._LRU_cache.popitem(last=False)
                self.__update_LRU_cache_size(old_s, old_emb, how="DEL")
                self.__store_in_DB(old_s, old_emb)
            except KeyError as e:
                if self._LRU_cache_size_kb == 0.0:
                    _logger.warning("Item size is bigger than the LRU cache capacity "
                                    f"{item_size} > {self._LRU_cache_size_kb}. "
                                    "Consider increasing the cache capacity."
                                    "If not, expect slower performance due to in-disk cache lookup overhead.")
                    self.__store_in_DB(old_s, old_emb)
                else:
                    raise KeyError(e)
                return 
        self.__store_in_LRU(s, emb)
        self.__update_LRU_cache_size(s, emb, how="ADD")
        
    # ---------PUBLIC---------  
    def embed_sequences(self, seqs: List[str], batch_size: int = 1000):
        """
        Embed a list of free-text strings, reusing cached vectors when possible.

        Returns a list aligned with *seqs* of ``np.ndarray`` objects.
        Internally obeys OpenAI batch limits and fills the cache on misses.
        """
        out = [None] * len(seqs)
        to_send, idx = [], []

        for i, s in enumerate(seqs):
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

        return out

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                           Multi-hot Encodings
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class MultiHotEncoder:
    """Generic helper: tuple-of-labels ➜ tuple-of-int-indices (memory-light)."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def encode(self, sequences: pd.Series):
        """
        Fit a `sklearn.preprocessing.MultiLabelBinarizer`
        and encode each sample as a tuple of integer indices.

        Raises
        ------
        ValueError
            If any element in *sequences* is not a ``tuple``.
        """
        # ----checking-everything-is-tuple-----------------------------------
        if not sequences.map(lambda x: isinstance(x, tuple)).all():
            bad = sequences[~sequences.map(lambda x: isinstance(x, tuple))].index[:5]
            raise ValueError(f"Non-tuple entries detected at rows {list(bad)}")
        # -------------------------------------------------------------------
        self.mlb.fit(sequences)
        cls_to_idx = {c: i for i, c in enumerate(self.mlb.classes_)}

        encoded = [
            tuple(sorted(cls_to_idx[lbl] for lbl in seq)) for seq in sequences
        ]
        return {
            "encodings": encoded,
            "class_labels": cls_to_idx,
        }


class GOEncoder(MultiHotEncoder):
    """
    Encode Gene Ontology annotations at a fixed GO depth.

    ``depth`` is absolute; if omitted it is auto-selected so that
    *coverage_target* fraction of annotations are retained.
    """

    def __init__(self, obo_path: str):
        super().__init__()
        if not os.path.exists(obo_path):
            raise FileNotFoundError(f"OBO not found: {obo_path}")
        self.godag = GODag(obo_path)
    
    # ---------PROTECTED---------
    def _auto_depth(self, series: pd.Series, coverage_target: float = 0.8) -> int:
        depths = [
            self.godag[gid].depth
            for terms in series.dropna()
            for gid in terms
            if gid in self.godag
        ]
        if not depths:
            raise ValueError("No valid GO IDs found to compute automatic depth.")
        return int(np.percentile(depths, coverage_target * 100))

    def _collapse_to_depth(self, go_ids: Tuple[str], k: int) -> Tuple[str]:
        kept = set()
        for gid in go_ids:
            if gid not in self.godag:
                continue
            node = self.godag[gid]
            ancestors = {gid}.union(node.get_all_parents())
            at_k = {n for n in ancestors if self.godag[n].depth == k}
            kept.update(at_k if at_k else {min(ancestors, key=lambda x: self.godag[x].depth)})
        return tuple(sorted(kept))

    # ---------PUBLIC---------
    def encode_go(
        self, df: pd.DataFrame, col_name: str, depth: Union[None, int] = None,
        coverage_target: Union[float, None] = None, inplace: bool = False):
        """
        Collapse GO term lists to a single depth and encode as indices.

        Parameters
        ----------
        df : pandas.DataFrame
            Source table.
        col_name : str
            Column containing tuples of GO term strings.
        depth : int | None
            Target GO depth. Mutually exclusive with *coverage_target*.
        coverage_target : float | None
            Percentile (0-1) of depth distribution to keep if *depth* not given.
        inplace : bool, default False
            Whether to mutate *df* or return a copy.

        Returns
        -------
        (DataFrame, Dict[str,int])
            The dataframe with encoded column, and the term→index map.
        """
        df = df if inplace else df.copy(deep=True)

        if depth is None:
            if coverage_target is None:
                raise ValueError(
                    "Either `depth` or `coverage_target` must be provided."
                )
            depth = self._auto_depth(df[col_name], coverage_target)

        collapsed = df.loc[:, col_name].map(lambda terms: self._collapse_to_depth(terms, depth))
        enc_info = self.encode(collapsed)

        df.loc[:, col_name] = pd.Series(list(enc_info["encodings"]), index=df.index, dtype=object)
        df.loc[:, col_name] = df[col_name].map(lambda x: np.nan if x == () else x)

        return df, enc_info["class_labels"]


class ECEncoder(MultiHotEncoder):
    """
    Encode Enzyme Commission numbers at a fixed annotation depth.

    ``depth`` is absolute; if omitted it is auto-selected so that
    *coverage_target* fraction of annotations are retained.
    """
    def __init__(self):
        super().__init__()

    # ---------PRIVATE---------
    def __extract_ec_codes(self, EC: str):
        present_entries = [i for i in EC.split(".") if i.isdigit()]
        return present_entries
    
    def __depth(self, EC: str):
        return len(self.__extract_ec_codes(EC))

    # ---------PROTECTED---------
    def _auto_depth(self, series: pd.Series, coverage_target: float = 0.8) -> int:
        depths = [
            self.__depth(ec)
            for terms in series.dropna()
            for ec in terms
        ]
        if not depths:
            raise ValueError("No valid EC numbers found to compute automatic depth.")
        return int(np.percentile(depths, coverage_target * 100))

    def _collapse_to_depth_helper(self, EC: str, depth: int):
        pieces = self.__extract_ec_codes(EC)[:depth]
        if not pieces:
            return None
        return ".".join(pieces)
        
    def _collapse_to_depth(self, ECs: Tuple[str], depth: int) -> Tuple[str]:
        cleaned = filter(None, (self._collapse_to_depth_helper(ec, depth) for ec in ECs))
        return tuple(sorted(set(cleaned)))

    # ---------PUBLIC---------
    def encode_ec(self, df: pd.DataFrame, col_name: str, depth: Union[int, None] = None,
                coverage_target: Union[float, None] = None, inplace=False):
        """
        Collapse EC numbers to a fixed depth then encode as index tuples.

        Parameters mirror `GOEncoder.encode_go`.
        """
        df = df if inplace else df.copy(deep=True)

        if depth is None:
            if coverage_target is None:
                raise ValueError(
                    "Either `depth` or `coverage_target` must be provided."
                )
            depth = self._auto_depth(df[col_name], coverage_target)

        collapsed = df.loc[:, col_name].map(lambda terms: self._collapse_to_depth(terms, depth))

        enc_info = self.encode(collapsed)
        df.loc[:, col_name] = pd.Series(list(enc_info["encodings"]), index=df.index, dtype=object)
        df.loc[:, col_name] = df[col_name].map(lambda x: np.nan if x == () else x)

        return df, enc_info["class_labels"]


def encode_multihot(df: pd.DataFrame, col: str, inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    One-shot convenience wrapper around `MultiHotEncoder`.

    Fits on *df[col]* and overwrites (or copies) that column
    with index tuples.

    Returns
    -------
    (DataFrame, Dict[str,int])
        Modified frame and the label vocabulary.
    """
    df = df if inplace else df.copy(deep=True)

    encoder = MultiHotEncoder()
    enc_info = encoder.encode(df[col])
    df.loc[:, col] = pd.Series(list(enc_info["encodings"]), index=df.index, dtype=object)
    df.loc[:, col] = df[col].map(lambda x: np.nan if x == () else x)

    return df, enc_info["class_labels"]


__all__ = [
    "MultiHotEncoder",
    "GOEncoder",
    "FreeTXTEmbedder",
    "AAChainEmbedder",
    "ECEncoder"
]

if __name__ == "__main__":
    pass
