# --- top of entry_preprocessing_tests.py -----------------
import sys, pathlib
# 1 directory up from /tests  -->  /data_prep_scripts
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dataset_curation.scripts.cleaning_utils import _preprocess_col_helper
# ---------------------------------------------------------

import unittest
import json
from os import getenv
import atexit

from dotenv import load_dotenv
load_dotenv()

DATA_EXTRACTION_TESTS = getenv("DATA_EXTRACTION_TESTS")

def display_stats(stats):
    print("=============================================================")
    for test_case in stats:
        P = stats[test_case]["Passed"]
        F = stats[test_case]["Failed"]
        print(f"'{test_case}' passed {P}/{P+F} test(s)")
    print("=============================================================")

class TestingDataExtraction(unittest.TestCase):
    
    stats = dict()

    @classmethod
    def setUpClass(cls):
        with open(DATA_EXTRACTION_TESTS, "r") as f:
            cls.data = json.load(f)

    def _test(self, col_name: str):
        TestingDataExtraction.stats[col_name] = {"Passed": 0, "Failed": 0}
        local_stats = TestingDataExtraction.stats[col_name]
        helper = _preprocess_col_helper(col_name, apply_norm=False, apply_strip_pubmed=False)
        for i, (raw, expected) in enumerate(TestingDataExtraction.data[col_name].items(), start=1):
            print(f"Running test {i} for {col_name}")
            expected = tuple(dict.fromkeys(expected)) if isinstance(expected, list) else expected
            with self.subTest(raw=raw):
                try:
                    self.assertEqual(helper(raw), expected)
                except AssertionError as e:
                    print(f"Test {i} has failed")
                    local_stats["Failed"] += 1
                    raise AssertionError(e)
                print(f"Test {i} was successful!")
                local_stats["Passed"] += 1

    def test_Domain_FT(self): # PASSES ALL
        self._test("Domain [FT]")

    def test_Domain_CC(self): # PASSES ALL
        self._test("Domain [CC]")

    def test_Protein_families(self): # PASSES ALL
        self._test("Protein families")

    def test_GO_MF(self): # PASSES ALL
        self._test("Gene Ontology (molecular function)")

    def test_GO_BP(self): # PASSES ALL
        self._test("Gene Ontology (biological process)")

    def test_Interacts_with(self): # PASSES ALL
        self._test("Interacts with")

    def test_Function_CC(self): # PASSES ALL
        self._test("Function [CC]")

    def test_Catalytic_activity(self): # PASSES ALL
        self._test("Catalytic activity")

    def test_ECN(self): # PASSES ALL
        self._test("EC number")

    def test_Pathway(self): # PASSES ALL
        self._test("Pathway")

    def test_Rhea_ID(self): # PASSES ALL
        self._test("Rhea ID")

    def test_Cofactor(self): # PASSES ALL
        self._test("Cofactor")
    
    def test_Activity_regulation(self): # PASSES ALL
        self._test("Activity regulation")

atexit.register(display_stats, TestingDataExtraction.stats)

if __name__ == "__main__":
    unittest.main()
