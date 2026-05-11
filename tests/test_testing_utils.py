import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.abspath("src"))

from M2F.testing_utils import accuracy, recall, precision, f1


class TestTestingUtils(unittest.TestCase):
    def test_metrics_nonempty(self):
        logits = torch.tensor([[2.0], [-2.0], [2.0], [-2.0]])
        y = torch.tensor([[1.0], [0.0], [0.0], [0.0]])
        mask = torch.tensor([True, True, True, False])

        acc = accuracy(logits, y, mask)
        rec = recall(logits, y, mask)
        pre = precision(logits, y, mask)
        f1_score = f1(logits, y, mask)

        self.assertAlmostEqual(acc, 2.0 / 3.0, places=5)
        self.assertAlmostEqual(rec, 1.0, places=5)
        self.assertAlmostEqual(pre, 0.5, places=5)
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(f1_score, 1.0)

    def test_metrics_empty_mask(self):
        logits = torch.tensor([[0.0], [0.0]])
        y = torch.tensor([[0.0], [1.0]])
        mask = torch.tensor([False, False])
        self.assertEqual(accuracy(logits, y, mask), 0.0)
        self.assertEqual(recall(logits, y, mask), 0.0)
        self.assertEqual(precision(logits, y, mask), 0.0)
        self.assertEqual(f1(logits, y, mask), 0.0)


if __name__ == "__main__":
    unittest.main()
