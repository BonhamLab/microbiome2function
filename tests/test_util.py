import os
import re
import shutil
import sys
import tempfile
import unittest
import warnings

import numpy as np
import torch
from torch_geometric.data import TensorAttr

sys.path.insert(0, os.path.abspath("src"))

from M2F.util import files_from, compose, suppress_warnings, current_time, ZarrFeatureStore


class TestUtilFunctions(unittest.TestCase):
    def test_files_from_filters_and_sorts(self):
        tmp = tempfile.mkdtemp(prefix="m2f_files_")
        try:
            for name in ["b.txt", "a.csv", "c.txt"]:
                with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
                    f.write("x")
            out = list(files_from(tmp, re.compile(r".*\.txt$")))
            self.assertEqual([os.path.basename(p) for p in out], ["b.txt", "c.txt"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_compose(self):
        def add(x, n):
            return x + n

        def mul(x, n):
            return x * n

        fn = compose(add, mul)
        out = fn(3, add=(2,), mul=(4,))
        self.assertEqual(out, 20)

    def test_suppress_warnings(self):
        @suppress_warnings(UserWarning)
        def f():
            warnings.warn("warn", UserWarning)
            return 1

        self.assertEqual(f(), 1)

    def test_current_time_format(self):
        ts = current_time()
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}_\d{6}$")


class TestZarrFeatureStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_zarr_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_write_read_remove_append(self):
        store = ZarrFeatureStore(self.tmp, read_only=False)
        try:
            attr = TensorAttr(None, "x", None)
            store.add_location(attr, shape=(3, 2), dtype="float32")
            store.add_data_to_location(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), TensorAttr(None, "x", slice(0, 2)))

            read = store.read_data_from_location(TensorAttr(None, "x", slice(0, 2)))
            np.testing.assert_allclose(read, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

            store.remove_data_from_location(TensorAttr(None, "x", 1))
            with self.assertRaises(IndexError):
                store.read_data_from_location(TensorAttr(None, "x", 1))

            out_slice = store.append(np.array([[5.0, 6.0]], dtype=np.float32), TensorAttr(None, "x", None))
            self.assertEqual(out_slice.start, 3)
            self.assertEqual(out_slice.stop, 4)

            meta = store.which_tensors
            self.assertIn("x", meta)
            self.assertEqual(meta["x"][0], (4, 2))
        finally:
            store.close()

    def test_read_only_create_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            ZarrFeatureStore(os.path.join(self.tmp, "missing"), read_only=True)


if __name__ == "__main__":
    unittest.main()
