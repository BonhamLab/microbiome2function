import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("src"))

import M2F


class TestPackageInit(unittest.TestCase):
    def test_public_api_exports(self):
        expected = {
            "configure_logging",
            "extract_accessions_from_humann",
            "clean_col",
            "MultiHotEncoder",
            "embed_ft_domains",
            "util",
        }
        self.assertTrue(expected.issubset(set(M2F.__all__)))


if __name__ == "__main__":
    unittest.main()
