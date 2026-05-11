import io
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath("src"))

from M2F import mining_utils


class DummyResponse:
    def __init__(self, text: str, status_ok: bool = True):
        self.text = text
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise requests.HTTPError("boom")


class TestMiningUtils(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_mining_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_humann_file(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# metadata row to skip\n")
            f.write("READS_UNMAPPED\tVALUE\n")
            for row in rows:
                f.write(f"{row}\t1\n")

    def test_extract_accessions_from_humann(self):
        p = os.path.join(self.tmp, "sample.tsv")
        self._write_humann_file(
            p,
            [
                "UniRef90_P12345",
                "UniRef90_UNK123",
                "UniClust90_9876",
                "garbage",
            ],
        )
        unirefs, uniclusts = mining_utils.extract_accessions_from_humann(p, out_type=list)
        self.assertIn("P12345", unirefs)
        self.assertNotIn("UNK123", unirefs)
        self.assertIn("9876", uniclusts)

    def test_extract_all_accessions_from_dir(self):
        p1 = os.path.join(self.tmp, "a.tsv")
        p2 = os.path.join(self.tmp, "b.tsv")
        self._write_humann_file(p1, ["UniRef90_P1"])
        self._write_humann_file(p2, ["UniRef90_P2", "UniClust90_22"])
        unirefs, uniclusts = mining_utils.extract_all_accessions_from_dir(self.tmp, out_type=set)
        self.assertEqual(unirefs, {"P1", "P2"})
        self.assertEqual(uniclusts, {"22"})

    def test_fetch_uniprotkb_fields_success_and_zero_ids(self):
        tsv = "Entry\tFunction [CC]\nP1\tfoo\nP2\tbar\n"
        with patch("M2F.mining_utils.requests.get", return_value=DummyResponse(tsv)):
            df = mining_utils.fetch_uniprotkb_fields(["P1", "P2"], ["accession", "cc_function"], request_size=2, rps=1000)
        self.assertFalse(df.empty)
        self.assertIn("Entry", df.columns)

        # zero ids should not crash
        with patch("M2F.mining_utils.requests.get", return_value=DummyResponse(tsv)):
            df0 = mining_utils.fetch_uniprotkb_fields([], ["accession"], request_size=2)
        self.assertTrue(df0.empty)

    def test_fetch_uniprotkb_fields_http_error_drop(self):
        with patch("M2F.mining_utils.requests.get", side_effect=requests.HTTPError("boom")):
            df = mining_utils.fetch_uniprotkb_fields(["P1"], ["accession"], request_size=1, max_retry=0)
        self.assertTrue(df.empty)

    def test_fetch_save_uniprotkb_batches_csv_fallback(self):
        out_dir = os.path.join(self.tmp, "out")
        fake_df = pd.DataFrame({"Entry": ["P1"], "X": [1]})

        with patch("M2F.mining_utils.fetch_uniprotkb_fields", return_value=fake_df):
            with patch.object(pd.DataFrame, "to_parquet", side_effect=RuntimeError("no parquet")):
                path = mining_utils.fetch_save_uniprotkb_batches(
                    uniref_ids=["P1", "P2"],
                    fields=["accession"],
                    batch_size=1,
                    save_to_dir=out_dir,
                    rps=1000,
                )

        self.assertEqual(path, out_dir)
        files = os.listdir(out_dir)
        self.assertTrue(any(name.endswith(".csv") for name in files))


if __name__ == "__main__":
    unittest.main()
