import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("src"))

from M2F.embedding_utils import (
    silent_transformers,
    AAChainEmbedder,
    FreeTXTEmbedder,
    MultiHotEncoder,
    GOEncoder,
    ECEncoder,
    encode_multihot,
)


class _DummyBatch(dict):
    def to(self, _device):
        return self


class _DummyTokenizer:
    def __call__(self, seqs, return_tensors=None, padding=True, truncation=True, max_length=None, return_special_tokens_mask=True):
        bsz = len(seqs)
        lens = [min(len(s), max_length - 2) for s in seqs]
        L = (max(lens) + 2) if lens else 2
        input_ids = torch.zeros((bsz, L), dtype=torch.long)
        attention_mask = torch.zeros((bsz, L), dtype=torch.long)
        special_tokens_mask = torch.zeros((bsz, L), dtype=torch.long)
        for i, n in enumerate(lens):
            attention_mask[i, : n + 2] = 1
            special_tokens_mask[i, 0] = 1
            special_tokens_mask[i, n + 1] = 1
        return _DummyBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
        )


class _DummyConfig:
    num_hidden_layers = 2
    max_position_embeddings = 64


class _DummyModel:
    def __init__(self):
        self.config = _DummyConfig()
        self.device = "cpu"

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        b, L = kwargs["input_ids"].shape
        D = 4
        base = torch.arange(b * L * D, dtype=torch.float32).reshape(b, L, D)
        hidden_states = [torch.zeros_like(base), base + 1.0, base + 2.0]
        return type("Out", (), {"hidden_states": hidden_states})


class _DummyEmbObj:
    def __init__(self, emb):
        self.embedding = emb


class _DummyEmbeddingsEndpoint:
    def __init__(self):
        self.calls = []

    def create(self, input, model):
        self.calls.append((tuple(input), model))
        data = [_DummyEmbObj([float(len(s)), 1.0, 0.0]) for s in input]
        return type("Resp", (), {"data": data})


class _DummyOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.embeddings = _DummyEmbeddingsEndpoint()


class TestEmbeddingUtils(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_embed_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_silent_transformers_restores_level(self):
        logger = logging.getLogger("transformers.modeling_utils")
        old = logger.level
        logger.setLevel(logging.INFO)
        with silent_transformers():
            self.assertEqual(logger.level, logging.ERROR)
        self.assertEqual(logger.level, logging.INFO)
        logger.setLevel(old)

    def test_aa_chain_embedder_with_mocked_hf(self):
        with patch("M2F.embedding_utils.AutoTokenizer.from_pretrained", return_value=_DummyTokenizer()):
            with patch("M2F.embedding_utils.AutoModel.from_pretrained", return_value=_DummyModel()):
                emb = AAChainEmbedder(model_key="esm2_t6_8M_UR50D", device="cpu", representation_layer="last")
                out = emb.embed_sequences(["AAAA", "CC"], batch_size=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, (4,))
        self.assertTrue(np.isfinite(out[0]).all())

    def test_freetxt_embedder_cache_paths(self):
        db_path = os.path.join(self.tmp, "cache.sqlite")
        with patch("M2F.embedding_utils.OpenAI", _DummyOpenAIClient):
            emb = FreeTXTEmbedder(
                api_key="k",
                model="SMALL_OPENAI_MODEL",
                cache_file_path=db_path,
                caching_mode="APPEND",
                max_cache_size_kb=1,
            )
            out1 = emb.embed_sequences(["abc", "def"], batch_size=2)
            out2 = emb.embed_sequences(["abc", "def"], batch_size=2)
            # second call should be fully cached
            self.assertEqual(len(emb.client.embeddings.calls), 1)
            self.assertEqual(len(out1), 2)
            self.assertEqual(len(out2), 2)
            emb._flush_and_close()

    def test_freetxt_embedder_create_override(self):
        db_path = os.path.join(self.tmp, "cache.sqlite")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings (text TEXT PRIMARY KEY, vec BLOB)")
        cur.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", ("old", b"abc"))
        conn.commit()
        conn.close()

        with patch("M2F.embedding_utils.OpenAI", _DummyOpenAIClient):
            emb = FreeTXTEmbedder(
                api_key="k",
                model="SMALL_OPENAI_MODEL",
                cache_file_path=db_path,
                caching_mode="CREATE/OVERRIDE",
            )
            emb._flush_and_close()

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_multihot_encoder_and_wrapper(self):
        s = pd.Series([("a", "b"), ("b",), tuple()])
        enc = MultiHotEncoder().encode(s)
        self.assertIn("class_labels", enc)
        self.assertEqual(len(enc["encodings"]), 3)

        df = pd.DataFrame({"col": s})
        out_df, labels = encode_multihot(df, "col", inplace=False)
        self.assertIn("col", out_df.columns)
        self.assertIsInstance(labels, dict)

    def _write_tiny_obo(self):
        p = os.path.join(self.tmp, "tiny.obo")
        with open(p, "w", encoding="utf-8") as f:
            f.write("format-version: 1.2\n\n")
            f.write("[Term]\n")
            f.write("id: GO:0000001\nname: root\nnamespace: biological_process\n\n")
            f.write("[Term]\n")
            f.write("id: GO:0000002\nname: child\nnamespace: biological_process\nis_a: GO:0000001 ! root\n\n")
            f.write("[Term]\n")
            f.write("id: GO:0000003\nname: grandchild\nnamespace: biological_process\nis_a: GO:0000002 ! child\n\n")
        return p

    def test_go_encoder_depth_and_encode(self):
        obo = self._write_tiny_obo()
        enc = GOEncoder(obo)
        collapsed = enc._collapse_to_depth(("GO:0000003",), 1)
        self.assertIn("GO:0000002", collapsed)

        df = pd.DataFrame({"go": [("GO:0000003",), ("GO:0000002",), tuple()]})
        out_df = enc.cut_to_depth(df, "go", depth=1, inplace=False)
        self.assertIn("go", out_df.columns)

        enc_df, labels = enc.encode_go(df, "go", depth=1, inplace=False)
        self.assertIsInstance(labels, dict)
        self.assertIn("go", enc_df.columns)

    def test_ec_encoder(self):
        enc = ECEncoder()
        df = pd.DataFrame({"ec": [("1.2.3.4", "1.2.3.5"), ("2.7.11.1",), tuple()]})
        cut = enc.cut_to_depth(df, "ec", depth=2, inplace=False)
        self.assertIn("ec", cut.columns)

        out_df, labels = enc.encode_ec(df, "ec", depth=2, inplace=False)
        self.assertIn("ec", out_df.columns)
        self.assertIsInstance(labels, dict)


if __name__ == "__main__":
    unittest.main()
