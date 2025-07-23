# --- top of entry_preprocessing_tests.py -----------------
import sys, pathlib
# 1 directory up from /tests  -->  /data_prep_scripts
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data_preparation_utils import _preprocess_col_helper
# ---------------------------------------------------------

import unittest
import json
from os import getenv
import atexit

from dotenv import load_dotenv
load_dotenv()

TEST_DATA = getenv("TEST_DATA")


class Testing(unittest.TestCase):
    
    stats = dict()

    @classmethod
    def setUpClass(cls):
        with open(TEST_DATA, "r") as f:
            cls.data = json.load(f)

    @staticmethod
    def display_stats(stats):
        print("=============================================================")
        for test_case in stats:
            P = stats[test_case]["Passed"]
            F = stats[test_case]["Failed"]
            print(f"'{test_case}' passed {P} test(s) and failed {F} test(s)")
        print("=============================================================")

    def _test(self, col_name: str):
        Testing.stats[col_name] = {"Passed": 0, "Failed": 0}
        local_stats = Testing.stats[col_name]
        helper = _preprocess_col_helper(col_name, apply_norm=False, apply_strip_pubmed=False)
        for i, (raw, expected) in enumerate(Testing.data[col_name].items(), start=1):
            print(f"Running test {i} for {col_name}")
            expected = tuple(expected) if isinstance(expected, list) else expected
            with self.subTest(raw=raw):
                try:
                    self.assertEqual(helper(raw), expected)
                except AssertionError as e:
                    print(f"Test {i} has failed")
                    local_stats["Failed"] += 1
                    raise AssertionError(e)
                print(f"Test {i} was successful!")
                local_stats["Passed"] += 1

    def test_Domain_FT(self):
        self._test("Domain [FT]")

    def test_Domain_FT(self):
        self._test("Domain [FT]")

    def test_Domain_CC(self):
        self._test("Domain [CC]")

    def test_Protein_families(self):
        self._test("Protein families")

    def test_GO_MF(self):
        self._test("Gene Ontology (molecular function)")

    def test_GO_BP(self):
        self._test("Gene Ontology (biological process)")

    def test_Interacts_with(self):
        self._test("Interacts with")

    def test_Function_CC(self):
        self._test("Function [CC]")

    def test_Catalytic_activity(self):
        self._test("Catalytic activity")

    def test_ECN(self):
        self._test("EC number")

    def test_Pathway(self):
        self._test("Pathway")

    def test_Rhea_ID(self):
        self._test("Rhea ID")

    def test_Cofactor(self):
        self._test("Cofactor")
    
    def test_Activity_regulation(self):
        self._test("Activity regulation")

atexit.register(Testing.display_stats, Testing.stats)

if __name__ == "__main__":
    unittest.main()
    """ July 23, 2025
    =============================================================
    'Activity regulation' passed 1 test(s) and failed 19 test(s)
    'Catalytic activity' passed 18 test(s) and failed 2 test(s)
    'Cofactor' passed 16 test(s) and failed 2 test(s)
    'Domain [CC]' passed 19 test(s) and failed 0 test(s)
    'Domain [FT]' passed 20 test(s) and failed 0 test(s)
    'EC number' passed 0 test(s) and failed 20 test(s)
    'Function [CC]' passed 20 test(s) and failed 0 test(s)
    'Gene Ontology (biological process)' passed 17 test(s) and failed 1 test(s)
    'Gene Ontology (molecular function)' passed 17 test(s) and failed 3 test(s)
    'Interacts with' passed 0 test(s) and failed 20 test(s)
    'Pathway' passed 19 test(s) and failed 1 test(s)
    'Protein families' passed 18 test(s) and failed 1 test(s)
    'Rhea ID' passed 20 test(s) and failed 0 test(s)
    =============================================================
    """