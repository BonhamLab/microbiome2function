import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from M2F.feature_engineering_utils import (
    max_pool,
    vals2embs_map,
    save_df,
    load_df,
    empty_tuples_to_NaNs,
    _domain_aa_ranges,
    _get_domain_sequences,
    embed_ft_domains,
    embed_AAsequences,
    embed_freetxt_cols,
)


class _DummyEmbedder:
    def embed_sequences(self, seqs, batch_size=1000):
        out = []
        for s in seqs:
            out.append(np.array([float(len(s)), 1.0, 0.0], dtype=np.float32))
        return out


class TestFeatureEngineeringUtils(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_feat_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_max_pool(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 0.0], dtype=np.float32)
        out = max_pool([a, b])
        np.testing.assert_allclose(out, b)

    def test_max_pool_errors(self):
        with self.assertRaises(ValueError):
            max_pool([])
        with self.assertRaises(ValueError):
            max_pool([np.array([1.0]), np.array([1.0, 2.0])])

    def test_vals2embs_map(self):
        df = pd.DataFrame({"col": [("a", "bb"), ("bb", "ccc"), tuple()]})
        m = vals2embs_map(df, "col", _DummyEmbedder(), batch_size=2)
        self.assertIn("a", m)
        self.assertIn("bb", m)
        self.assertEqual(m["ccc"].shape, (3,))

    def test_save_load_df_roundtrip(self):
        path = os.path.join(self.tmp, "df.zip")
        df = pd.DataFrame(
            {
                "Entry": ["A", "B", "C"],
                "tuples": [(1, 2), tuple(), (3,)],
                "arr": [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32), np.array([5.0, 6.0], dtype=np.float32)],
            }
        )
        save_df(df, path, metadata={"m": "v"})
        out = load_df(path)
        self.assertEqual(set(out.columns), {"Entry", "arr", "tuples"})
        self.assertEqual(out.attrs.get("m"), "v")

    def test_empty_tuples_to_nans(self):
        df = pd.DataFrame({"a": [(), (1,), ()]})
        out = empty_tuples_to_NaNs(df, inplace=False)
        self.assertTrue(pd.isna(out.loc[0, "a"]))
        self.assertEqual(out.loc[1, "a"], (1,))

    def test_domain_helpers(self):
        ranges = _domain_aa_ranges(("1..3", "5..6", "bad"))
        self.assertEqual(ranges, [(0, 3), (4, 6)])
        seqs = _get_domain_sequences(("1..3",), ("ABCDEFG",))
        self.assertEqual(seqs, ["ABC"])

    def test_embed_ft_domains_and_sequences_and_text(self):
        emb = _DummyEmbedder()
        df = pd.DataFrame(
            {
                "Domain [FT]": [("1..3",), tuple()],
                "Sequence": [("ABCDEFG",), ("XYZ",)],
                "txt": [("hello", "world"), tuple()],
            }
        )

        out1 = embed_ft_domains(df, emb, batch_size=10, inplace=False)
        self.assertEqual(out1.loc[0, "Domain [FT]"].shape, (3,))

        out2 = embed_AAsequences(df, emb, batch_size=10, inplace=False)
        self.assertEqual(out2.loc[0, "Sequence"].shape, (3,))

        out3 = embed_freetxt_cols(df, ["txt"], emb, batch_size=10, inplace=False)
        self.assertEqual(out3.loc[0, "txt"].shape, (3,))


if __name__ == "__main__":
    unittest.main()
