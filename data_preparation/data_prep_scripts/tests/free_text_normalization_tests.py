# --- top of entry_preprocessing_tests.py -----------------
import sys, pathlib
# 1 directory up from /tests  -->  /data_prep_scripts
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data_preparation_utils import normalize
# ---------------------------------------------------------

import unittest
import json
from os import getenv
import atexit

from dotenv import load_dotenv
load_dotenv()

FREE_TEXT_NOMRALIZATION_TESTS = getenv("FREE_TEXT_NOMRALIZATION_TESTS")

def display_stats(stats):
    print("=============================================================")
    P = stats["Passed"]
    F = stats["Failed"]
    print(f"Passed {P}/{P+F} test(s)")
    print("=============================================================")

class TestingFreeTxtNorm(unittest.TestCase):
    
    stats = {"Passed": 0, "Failed": 0}

    @classmethod
    def setUpClass(cls):
        with open(FREE_TEXT_NOMRALIZATION_TESTS, "r") as f:
            cls.data = json.load(f)

    def test_normalize(self):
        stats = TestingFreeTxtNorm.stats
        for i, (raw, expected) in enumerate(TestingFreeTxtNorm.data.items(), start=1):
            print(f"Running test {i}: ")
            with self.subTest(raw=raw):
                try:
                    self.assertEqual(normalize(raw), expected)
                except AssertionError as e:
                    print(f"Test {i} has failed")
                    stats["Failed"] += 1
                    raise AssertionError(e)
                print(f"Test {i} was successful!")
                stats["Passed"] += 1

atexit.register(display_stats, TestingFreeTxtNorm.stats)

if __name__ == "__main__":
    unittest.main()