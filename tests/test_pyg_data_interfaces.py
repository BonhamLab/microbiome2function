import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("src"))

from M2F.pyg_data_interfaces import (
    DatasetInput,
    build_features_from_DatasetInput,
    build_topology_from_DatasetInput,
    ProteinGraphInMemoryDataset,
    ProteinGraphOnDiskDataset,
    ProteinDataset,
)


class TestPygDataInterfaces(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m2f_pyg_")
        self.root_inmem = os.path.join(self.tmp, "inmem")
        self.root_ondisk = os.path.join(self.tmp, "ondisk")
        self.root_ffnn = os.path.join(self.tmp, "ffnn")
        self.edge_dir = os.path.join(self.tmp, "edges")
        os.makedirs(self.edge_dir, exist_ok=True)

        self.accessions_path = os.path.join(self.tmp, "accessions.csv")
        pd.DataFrame(
            {
                "uniref": ["UniRef90_A", "UniRef90_B", "UniRef90_C"],
                "i": [1, 2, 3],
            }
        ).to_csv(self.accessions_path, index=False)

        pd.DataFrame({"j": [2, 3], "w": [0.1, 0.2]}).to_csv(os.path.join(self.edge_dir, "chunk_1.csv"), index=False)
        pd.DataFrame({"j": [1, 3], "w": [0.3, 0.4]}).to_csv(os.path.join(self.edge_dir, "chunk_2.csv"), index=False)
        pd.DataFrame({"j": [1], "w": [0.5]}).to_csv(os.path.join(self.edge_dir, "chunk_3.csv"), index=False)

        self.mapping = {
            "A": {"feat": 1.0, "label": 0.0},
            "B": {"feat": 2.0, "label": 1.0},
            "C": {"feat": 3.0, "label": 0.0},
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset_input_graph(self, num_feature_batches=None):
        return DatasetInput(
            path_to_accession_ids_csv_file=self.accessions_path,
            path_to_edge_csv_dir=self.edge_dir,
            X={"feat": "feat"},
            Y={"label": "label"},
            num_feature_batches=num_feature_batches,
        )

    def _dataset_input_ffnn(self, num_feature_batches=None):
        return DatasetInput(
            path_to_accession_ids_csv_file=self.accessions_path,
            X={"feat": "feat"},
            Y={"label": "label"},
            num_feature_batches=num_feature_batches,
        )

    @staticmethod
    def _fake_fetch(uniref_ids, fields, request_size=100, rps=10, max_retry=float("inf")):
        mapping = {
            "A": {"feat": 1.0, "label": 0.0},
            "B": {"feat": 2.0, "label": 1.0},
            "C": {"feat": 3.0, "label": 0.0},
        }
        cols = ["Entry"] + [f for f in fields if f != "accession"]
        rows = []
        for uid in uniref_ids:
            row = {"Entry": uid}
            if uid in mapping:
                for f in cols:
                    if f == "Entry":
                        continue
                    row[f] = mapping[uid][f]
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def test_dataset_input_validation_modes(self):
        d = self._dataset_input_ffnn()
        d.validate(require_graph=False)
        with self.assertRaises(ValueError):
            d.validate(require_graph=True)

        dg = self._dataset_input_graph()
        dg.validate(require_graph=True)

    def test_build_features_and_duplicate_guard(self):
        features_path = os.path.join(self.tmp, "features.csv")
        pd.DataFrame(
            {
                "Entry": ["A", "B", "C"],
                "feat": [1.0, 2.0, 3.0],
                "label": [0.0, 1.0, 0.0],
            }
        ).to_csv(features_path, index=False)

        id_map = np.full(3, -1, dtype=np.int64)
        x, y = build_features_from_DatasetInput(
            pre_transform=None,
            pre_filter=None,
            accessions_path=self.accessions_path,
            features_path=features_path,
            required_cols=["label", "feat", "Entry"],
            global_id_map=id_map,
            X_return_field_names=("feat", "Entry"),
            Y_return_field_name="label",
        )
        self.assertEqual(x.shape, (3, 1))
        self.assertEqual(y.shape, (3, 1))
        np.testing.assert_array_equal(id_map, np.array([0, 1, 2], dtype=np.int64))

        dup_features_path = os.path.join(self.tmp, "dup_features.csv")
        pd.DataFrame(
            {
                "Entry": ["A", "A"],
                "feat": [1.0, 1.5],
                "label": [0.0, 0.0],
            }
        ).to_csv(dup_features_path, index=False)

        with self.assertRaises(ValueError):
            build_features_from_DatasetInput(
                pre_transform=None,
                pre_filter=None,
                accessions_path=self.accessions_path,
                features_path=dup_features_path,
                required_cols=["label", "feat", "Entry"],
                global_id_map=np.full(3, -1, dtype=np.int64),
                X_return_field_names=("feat", "Entry"),
                Y_return_field_name="label",
            )

    def test_build_topology(self):
        id_map = np.array([0, 1, -1], dtype=np.int64)
        edge_index, edge_attr, edge_cols = build_topology_from_DatasetInput(
            id_map=id_map,
            csv_dir=self.edge_dir,
            edge_csv_file_name_pattern=re.compile(r"chunk_\d+\.csv"),
            edge_attr_columns=["w"],
            chunk_name_pattern=re.compile(r"chunk_(\d+)\.csv$"),
            edge_dst_column="j",
        )
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_attr.shape[1], 1)
        self.assertEqual(edge_cols, ("w",))
        # all mapped node ids should be in {0,1}
        self.assertTrue(set(edge_index.flatten().tolist()).issubset({0, 1}))

    def test_protein_graph_inmemory_dataset(self):
        di = self._dataset_input_graph()
        with patch("M2F.pyg_data_interfaces.fetch_uniprotkb_fields", side_effect=self._fake_fetch):
            ds = ProteinGraphInMemoryDataset(
                root=self.root_inmem,
                dataset_input=di,
                force_reload=True,
                val_set_size=0.2,
                test_set_size=0.2,
            )
        data = ds[0]
        self.assertEqual(int(data.num_nodes), 3)
        self.assertEqual(data.x.shape[1], 1)
        self.assertEqual(data.y.shape[1], 1)

    def test_protein_graph_ondisk_dataset(self):
        di = self._dataset_input_graph(num_feature_batches=2)
        with patch("M2F.pyg_data_interfaces.fetch_uniprotkb_fields", side_effect=self._fake_fetch):
            ds = ProteinGraphOnDiskDataset(
                root=self.root_ondisk,
                dataset_input=di,
                force_reload=True,
                val_set_size=0.2,
                test_set_size=0.2,
            )
        self.assertTrue(os.path.exists(ds.edge_index_path))
        self.assertTrue(os.path.exists(ds.id_map_path))
        self.assertTrue(os.path.exists(ds.meta_path))
        self.assertEqual(ds.meta["num_nodes"], 3)
        ds.close()

    def test_protein_dataset_ffnn_interface(self):
        di = self._dataset_input_ffnn(num_feature_batches=2)
        with patch("M2F.pyg_data_interfaces.fetch_uniprotkb_fields", side_effect=self._fake_fetch):
            ds = ProteinDataset(
                root=self.root_ffnn,
                dataset_input=di,
                force_reload=True,
                split="train",
                val_set_size=0.2,
                test_set_size=0.2,
            )

        self.assertGreaterEqual(len(ds), 1)
        item = ds[0]
        self.assertEqual(len(item), 2)  # x, y
        x, y = item
        self.assertEqual(tuple(x.shape), (1,))
        self.assertEqual(tuple(y.shape), (1,))

        pred_loader = ds.predict_loader(batch_size=2)
        first = next(iter(pred_loader))
        self.assertEqual(first.ndim, 2)
        ds.close()


if __name__ == "__main__":
    unittest.main()
