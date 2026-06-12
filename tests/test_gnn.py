import os
import shutil
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.abspath("src"))

from M2F.gnn import GATNodeClassifier, GraphConv, GraphConvNodeClassifier


class TestGNN(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_gnn_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _batch(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.ones((4, 1), dtype=torch.float32)
        y = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        d.batch_size = 2  # seed nodes in NeighborLoader semantics
        return d

    def test_graphconv_validation_and_forward(self):
        with self.assertRaises(ValueError):
            GraphConv(2, 0, 4, 4, edge_features_used_as="scaling")
        with self.assertRaises(ValueError):
            GraphConv(2, 1, 4, 4, edge_features_used_as="bad")

        conv = GraphConv(2, 1, 4, 4, edge_features_used_as="scaling")
        b = self._batch()
        out = conv(b.x, b.edge_index, b.edge_attr)
        self.assertEqual(out.shape, (3, 4))

    def test_classifier_fit_and_test(self):
        model = GraphConvNodeClassifier(in_dim=2, edge_dim=1, msg_dim=4, state_dim=4, out_dim=1, dropout_p=0.0)
        train = [self._batch()]
        val = [self._batch()]
        test = [self._batch()]

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

    def test_gat_classifier_fit_and_test(self):
        model = GATNodeClassifier(in_dim=2, edge_dim=1, msg_dim=4, state_dim=4, out_dim=1, heads=2, dropout_p=0.0)
        train = [self._batch()]
        val = [self._batch()]
        test = [self._batch()]

        logits = model._forward_logits(self._batch().x, self._batch().edge_index, self._batch().edge_attr)
        self.assertEqual(logits.shape, (3, 1))

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

    def test_gat_classifier_accepts_empty_edge_attrs(self):
        batch = self._batch()
        batch.edge_attr = torch.empty((batch.edge_index.size(1), 0), dtype=torch.float32)
        model = GATNodeClassifier(in_dim=2, edge_dim=0, msg_dim=4, state_dim=4, out_dim=1, dropout_p=0.0)

        logits = model._forward_logits(batch.x, batch.edge_index, batch.edge_attr)
        self.assertEqual(logits.shape, (3, 1))

    def test_gat_validation(self):
        with self.assertRaises(ValueError):
            GATNodeClassifier(in_dim=2, edge_dim=-1, msg_dim=4, state_dim=4, out_dim=1)
        with self.assertRaises(ValueError):
            GATNodeClassifier(in_dim=2, edge_dim=1, msg_dim=4, state_dim=4, out_dim=1, heads=0)
        with self.assertRaises(ValueError):
            GATNodeClassifier(in_dim=2, edge_dim=1, msg_dim=4, state_dim=5, out_dim=1, heads=2)

    def test_fit_validation(self):
        model = GraphConvNodeClassifier(in_dim=2, edge_dim=1, msg_dim=4, state_dim=4, out_dim=1)
        train = [self._batch()]
        val = [self._batch()]
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=0)
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=1, report_performance_every_kth_epoch=0)
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=1, tolerance=-1)


if __name__ == "__main__":
    unittest.main()
