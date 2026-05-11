import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from M2F.cleaning_utils import strip_pubmed, normalize, clean_col, clean_cols


class TestCleaningUtils(unittest.TestCase):
    def test_strip_pubmed(self):
        s = "Catalyzes reaction (PubMed:12345, PubMed:99999) and more {ECO:0000|PubMed:777}"
        out = strip_pubmed(s)
        self.assertNotIn("PubMed", out)
        self.assertIn("Catalyzes reaction", out)

    def test_normalize(self):
        s = "  Hello, World!!  EC-1/2  "
        out = normalize(s)
        self.assertEqual(out, "hello world ec-1/2")

    def test_clean_col_extracts_pattern(self):
        df = pd.DataFrame({"EC number": ["1.1.1.1; 2.7.7.7", None]})
        out = clean_col(df, "EC number", inplace=False)
        self.assertEqual(out.loc[0, "EC number"], ("1111", "2777"))
        self.assertEqual(out.loc[1, "EC number"], ())

    def test_clean_cols_multi(self):
        df = pd.DataFrame(
            {
                "Function [CC]": ["FUNCTION: Oxidoreductase {ECO:1|PubMed:123}", "FUNCTION: Transport"],
                "Pathway": ["PATHWAY: foo; PATHWAY: bar", "PATHWAY: baz"],
            }
        )
        out = clean_cols(df, ["Function [CC]", "Pathway"], inplace=False)
        self.assertIsInstance(out.loc[0, "Function [CC]"], tuple)
        self.assertIsInstance(out.loc[0, "Pathway"], tuple)

    def test_clean_col_missing_column_raises(self):
        df = pd.DataFrame({"A": [1]})
        with self.assertRaises(KeyError):
            clean_col(df, "B")


if __name__ == "__main__":
    unittest.main()
