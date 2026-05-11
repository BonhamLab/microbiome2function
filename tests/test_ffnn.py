import os
import shutil
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath("src"))

from M2F.ffnn import FFNN


class TestFFNN(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_ffnn_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _loaders(self):
        X = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        y = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
        ds = TensorDataset(X, y)
        train = DataLoader(ds, batch_size=2, shuffle=True)
        val = DataLoader(ds, batch_size=2, shuffle=False)
        test = DataLoader(ds, batch_size=2, shuffle=False)
        return train, val, test

    def test_forward_train_eval(self):
        model = FFNN(2, 4, 4, 1)
        X = torch.randn(3, 2)

        model.train()
        logits = model(X)
        self.assertEqual(logits.shape, (3, 1))

        model.eval()
        probs = model(X)
        self.assertTrue(torch.all((probs >= 0.0) & (probs <= 1.0)))

    def test_fit_and_test(self):
        model = FFNN(2, 4, 4, 1, dropout_p=0.0)
        train, val, test = self._loaders()

        hist = model.fit(
            train,
            val,
            epochs=2,
            report_performance_every_kth_epoch=1,
            save_model_to=self.tmp,
            early_stopping=False,
        )
        self.assertIn("best_val_loss", hist)
        self.assertEqual(len(hist["history"]), 2)

        metrics = model.test(test)
        self.assertIn("test_loss", metrics)
        self.assertIn("test_acc", metrics)
        self.assertIn("test_recall", metrics)

    def test_fit_input_validation(self):
        model = FFNN(2, 4, 4, 1)
        train, val, _ = self._loaders()
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=0)
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=1, report_performance_every_kth_epoch=0)
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=1, tolerance=-1)


if __name__ == "__main__":
    unittest.main()
