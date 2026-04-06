from __future__ import annotations

# third-party
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch_geometric.data import \
    (InMemoryDataset, 
    Data,
    FeatureStore,
    GraphStore,
    TensorAttr,
    EdgeAttr)
from torch_geometric.typing import EdgeTensorType, FeatureTensorType
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import pandas as pd

# built-in
from dataclasses import dataclass, field
from typing import Iterator, Any, Optional
from pathlib import Path
from math import floor
import re
import shutil
import logging

# local
from . import util
from .mining_utils import fetch_uniprotkb_fields

_logger = logging.getLogger(__name__)
KeyType = tuple[Optional[str], Optional[str]]

@dataclass
class DatasetInput:
    """
    Input contract consumed by PyG dataset interfaces (InMemory / OnDisk).

    Expected raw format:
    - accession index CSV: columns ['uniref', 'i'] (1-based node ids)
    - edge chunk CSVs: file names like chunk_<i>.csv, must contain a destination id
      column (default: 'j'); all other columns can be used as edge attributes.
    - X: feature field names to query and their return names used as model inputs (e.g. {'sequence': 'Sequence'})
    - Y: target field name to query and its return name used as model output
    """
    # core dataset attrs
    path_to_accession_ids_csv_file: Path
    path_to_edge_csv_dir: Path
    X: dict[str, str]
    Y: dict[str, str]

    # internals -- uniprot query params
    request_size: int = 25
    rps: float = 1
    max_retry: int | float = 20
    edge_dst_column: str = "j"
    edge_attr_columns: list[str] | tuple[str, ...] | None = None
    edge_csv_file_name_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"chunk_\d+\.csv")
    )

    # internals -- dataclass specifics (ensuring data quality)
    _validation_ctx: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _accession_ids_df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path_to_accession_ids_csv_file = Path(self.path_to_accession_ids_csv_file)
        self.path_to_edge_csv_dir = Path(self.path_to_edge_csv_dir)
        self._normalize_edge_schema()
        self._normalize_xy()
        self.validate()
        self.X["accession"] = "Entry" # <-- we always want to request accession

    def validate(self) -> None:
        self._validate_uniprot_request_params()
        self._validate_xy()
        self._validate_accession_ids_csv_file()
        self._validate_edge_csv_files()
    
    def _normalize_xy(self) -> None:
        self.X = {k.strip(): v.strip() for k, v in self.X.items()}
        self.Y = {k.strip(): v.strip() for k, v in self.Y.items()}

    def _normalize_edge_schema(self) -> None:
        if not isinstance(self.edge_dst_column, str):
            raise TypeError(
                f"`edge_dst_column` must be str, got {type(self.edge_dst_column)}"
            )
        self.edge_dst_column = self.edge_dst_column.strip()
        if not self.edge_dst_column:
            raise ValueError("`edge_dst_column` cannot be empty")

    def _validate_xy(self) -> None:
        if not all([isinstance(fields, dict) for fields in [self.X, self.Y]]):
            raise TypeError(f"Both X and Y must be of type dict[str, str]")
        if len(self.X) == 0:
            raise ValueError("`X` cannot be empty")
        if len(self.Y) != 1:
            raise ValueError("`Y` must be a singleton dictionary")
        if "accession" in self.Y:
            raise ValueError("`Y` cannot be 'accession'")
        if set(self.Y.keys()).intersection(self.X.keys()):
            raise ValueError("Y field must not be present in X fields")
        if set(self.Y.values()).intersection(self.X.values()):
            raise ValueError("Y field must not be present in X fields")
        self._validation_ctx["X"] = self.X
        self._validation_ctx["Y"] = self.Y
        self._validation_ctx["num_X_fields"] = len(self.X)

    def _validate_uniprot_request_params(self) -> None:
        if self.request_size < 1:
            raise ValueError("`request_size` must be >= 1")
        if self.rps <= 0:
            raise ValueError("`rps` must be > 0")
        if self.max_retry < 0:
            raise ValueError("`max_retry` must be >= 0")

        self._validation_ctx["request_size"] = self.request_size
        self._validation_ctx["rps"] = self.rps
        self._validation_ctx["max_retry"] = self.max_retry

    def _validate_accession_ids_csv_file(self) -> None:
        if not self.path_to_accession_ids_csv_file.exists():
            raise FileNotFoundError(
                f"Accession index CSV not found: {self.path_to_accession_ids_csv_file}"
            )

        df = self.accession_ids
        expected_cols = ["uniref", "i"]
        if df.columns.tolist() != expected_cols:
            raise ValueError(
                f"Expected accession index columns {expected_cols}, got {df.columns.tolist()}"
            )

        if not pd.api.types.is_integer_dtype(df["i"]):
            raise ValueError("Column 'i' in accession index CSV must be integer dtype")

        if (df["i"] < 1).any():
            raise ValueError("Column 'i' must contain 1-based positive node ids")

        if not df["uniref"].astype(str).str.startswith("UniRef90_").all():
            raise ValueError("Column 'uniref' must contain UniRef90_* identifiers")

        if df["i"].duplicated().any():
            raise ValueError("Column 'i' contains duplicate node ids")

        self._validation_ctx["min_node_id"] = int(df["i"].min())
        self._validation_ctx["max_node_id"] = int(df["i"].max())
        self._validation_ctx["num_nodes"] = int(df.shape[0])
    
    def _validate_edge_csv_files(self) -> None:
        if not self.path_to_edge_csv_dir.exists():
            raise FileNotFoundError(f"Edge CSV directory not found: {self.path_to_edge_csv_dir}")

        files = list(util.files_from(str(self.path_to_edge_csv_dir), self.edge_csv_file_name_pattern))
        if not files:
            raise ValueError(
                f"No edge CSV files found in {self.path_to_edge_csv_dir} "
                f"matching {self.edge_csv_file_name_pattern.pattern}"
            )

        # validate only a small prefix for speed
        for path in files[:5]:
            df = pd.read_csv(path)
            if self.edge_dst_column not in df.columns:
                raise ValueError(
                    f"Expected destination column '{self.edge_dst_column}' in {path}"
                )

            if self.edge_attr_columns is not None:
                missing = [col for col in self.edge_attr_columns if col not in df.columns]
                if missing:
                    raise ValueError(
                        f"Missing requested edge attribute columns {missing} in {path}"
                    )

            if not df.empty and not pd.api.types.is_integer_dtype(df[self.edge_dst_column]):
                raise ValueError(
                    f"Column '{self.edge_dst_column}' must be integer dtype in {path}"
                )

        # Enforce one edge chunk file per node row in the index CSV.
        expected_num_edge_files = int(self.accession_ids.shape[0])
        if len(files) != expected_num_edge_files:
            raise ValueError(
                "Number of edge CSV files does not match accession index row count: "
                f"{len(files)} files vs {expected_num_edge_files} index rows."
            )

        self._validation_ctx["num_edge_files"] = len(files)

    @property
    def accession_ids(self) -> pd.DataFrame:
        if self._accession_ids_df is None:
            self._accession_ids_df = pd.read_csv(self.path_to_accession_ids_csv_file)
        return self._accession_ids_df

    @property
    def edge_csv_paths(self) -> Iterator[Path]:
        for file in util.files_from(str(self.path_to_edge_csv_dir), self.edge_csv_file_name_pattern):
            yield Path(file)

    @property
    def edge_csv_files(self) -> Iterator[pd.DataFrame]:
        for path in self.edge_csv_paths:
            yield pd.read_csv(path)

    @property
    def node_id_bounds(self) -> tuple[int, int]:
        if "min_node_id" not in self._validation_ctx or "max_node_id" not in self._validation_ctx:
            self._validate_accession_ids_csv_file()
        return self._validation_ctx["min_node_id"], self._validation_ctx["max_node_id"]
    
    @property
    def X_query_field_names(self) -> tuple[str, ...]:
        return tuple(self.X.keys())
    
    @property
    def X_return_field_names(self) -> tuple[str, ...]:
        return tuple(self.X.values())

    @property
    def Y_query_field_name(self) -> str:
        return str(*self.Y.keys())
    
    @property
    def Y_return_field_name(self) -> str:
        return str(*self.Y.values())


# In-RAM/VRAM data interface for GNNs
class ProteinGraphInMemoryDataset(InMemoryDataset):

    def __init__(
        self,
        root: str | Path,
        dataset_input: DatasetInput,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
        force_reload: bool = False,
        *,
        val_set_size: float = 0.1,
        test_set_size: float = 0.1
    ) -> None:
        self.dataset_input = dataset_input
        self.dataset_input.validate()

        if ((val_set_size + test_set_size) >= 1.0
            or val_set_size < 0.0
            or test_set_size < 0.0):
            raise ValueError()
        
        self.val_set_size   = val_set_size
        self.test_set_size  = test_set_size

        super().__init__(
            root=str(root),
            log=log,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )
        
        processed_path = Path(self.processed_paths[0])
        if processed_path.exists():
            self.data, self.slices = torch.load(processed_path, weights_only=False)

    @property
    def original_node_accessions(self):
        return [str(row.uniref).replace("UniRef90_", "", 1)
            for row in self.dataset_input.accession_ids.itertuples(index=False)
            if not str(row.uniref).startswith(("UniRef90_UNK", "UniRef90_UPI"))]

    @property
    def raw_file_names(self) -> list[str]:
        return [
            "features.csv",
            self.dataset_input.path_to_accession_ids_csv_file.name,
            *[path.name for path in self.dataset_input.edge_csv_paths],
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    @staticmethod
    def _materialize(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    def download(self) -> None:
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        features_path = raw_dir / "features.csv"
        if not features_path.exists():
            fetched_features = fetch_uniprotkb_fields(
                        uniref_ids=self.original_node_accessions,
                        fields=[self.dataset_input.Y_query_field_name, *self.dataset_input.X_query_field_names],
                        request_size=self.dataset_input.request_size,
                        rps=self.dataset_input.rps,
                        max_retry=self.dataset_input.max_retry
                    )
            fetched_features.to_csv(features_path, index=False)

        # put index + edge files into raw/ so raw_file_names is satisfied
        self._materialize(
            self.dataset_input.path_to_accession_ids_csv_file,
            raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name,
        )
        for edge_path in self.dataset_input.edge_csv_paths:
            self._materialize(edge_path, raw_dir / edge_path.name)

    @staticmethod
    def _to_tensor(value: Any, *, field_name: str, cast_float: bool = True) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise ValueError(f"Empty value for field '{field_name}'")
            tensor = torch.tensor(value)
        elif isinstance(value, (int, float, np.number, bool)):
            tensor = torch.tensor([value])
        else:
            raise TypeError(
                f"Field '{field_name}' has unsupported type {type(value)}. "
                "Apply a pre_transform that converts it to numeric tensors."
            )
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.flatten()
        return tensor.float() if cast_float else tensor

    def process(self) -> None:
        # ------------------------- get the paths ------------------------
        raw_dir = Path(self.raw_dir)
        features_path = raw_dir / "features.csv"
        index_path = raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name
        # ----------------------------------------------------------------

        # --------------------------- fail fast --------------------------
        if not features_path.exists():
            raise FileNotFoundError(f"Expected raw features at {features_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Expected accession index at {index_path}")
        # ----------------------------------------------------------------

        # ------------------------ read the data -------------------------
        index_df = pd.read_csv(index_path)
        features_df = pd.read_csv(features_path)
        # ----------------------------------------------------------------

        # ------------- align features with graph node order -------------
        if "Entry" not in features_df.columns:
            raise KeyError(
                    "features.csv is missing merge key 'Entry'. "
                    "UniProt fetch likely returned no usable schema."
                )

        index_df = index_df.copy()
        index_df["Entry"] = index_df["uniref"].astype(str).str.replace("UniRef90_", "", regex=False)
        index_df["_orig_node_id"] = index_df["i"].astype(np.int64) - 1

        # align node table to graph index order:
        # keep every node from index_df (left side), preserving its row order
        # join feature rows by accession; unmatched accessions get NaN features
        # filtering of invalid/missing nodes happens later (after transform/filter logic)
        node_df = index_df.merge(features_df, on="Entry", how="left", sort=False) # BASICALLY INDEX THE FEATURES
        # ----------------------------------------------------------------

        # 1) transform (dataset/table level)

        # ---------------------- transform the table ---------------------
        if self.pre_transform is not None:
            transformed = self.pre_transform(node_df)
            if not isinstance(transformed, pd.DataFrame):
                raise TypeError("`pre_transform` must return a pandas DataFrame in this interface")
            node_df = transformed
        # ----------------------------------------------------------------

        # 2) filter (dataset/table level)
        
        # --------------------- create the keep_mask ---------------------
        keep_mask = ~node_df["Entry"].astype(str).str.startswith(("UNK", "UPI")) # throw away the UNK & UPI prefixed entries
        if self.pre_filter is not None:
            filtered = self.pre_filter(node_df)
            if not isinstance(filtered, (pd.Series, np.ndarray, list, tuple)):
                raise TypeError("`pre_filter` must return a boolean mask for the node table")
            filtered = pd.Series(filtered, index=node_df.index)
            if filtered.shape[0] != node_df.shape[0]:
                raise ValueError("`pre_filter` mask length does not match number of nodes")
            keep_mask &= filtered.astype(bool)
        # ----------------------------------------------------------------

        # --------- always require non-missing supervised fields ---------
        required_cols = [self.dataset_input.Y_return_field_name, *self.dataset_input.X_return_field_names]
        missing_required = [col for col in required_cols if col not in node_df.columns]
        if missing_required:
            raise KeyError(f"Required columns missing after transform: {missing_required}")
        # ----------------------------------------------------------------

        # ------------------- expand and apply the mask ------------------
        # V V V what this does is it AND-combines the current keep mask
        # with "consider all rows and all required columns, then compute
        # True/False for each entry it that table based on isna, then collapse
        # all the columns within each row (note axis=1) to get a 1-D Series
        # where True values denote entries with no missing values."
        keep_mask &= ~node_df.loc[:, required_cols].isna().any(axis=1)
        node_df = node_df[keep_mask].copy()
        if node_df.empty:
            raise ValueError("All nodes were filtered out; cannot build dataset")
        # ----------------------------------------------------------------

        # --- build old->new node id map for edge filtering/reindexing ---
        max_old_id = int(index_df["_orig_node_id"].max())
        id_map = -np.ones(max_old_id + 1, dtype=np.int64)
        old_ids = node_df["_orig_node_id"].to_numpy(dtype=np.int64)
        new_ids = np.arange(node_df.shape[0], dtype=np.int64)
        id_map[old_ids] = new_ids # @ old ids write new ids
        # ^ ^ ^ -- for example: Original graph node ids (_orig_node_id): 0,1,2,3,4,5
        # After filtering, kept nodes are old ids: 1,4,5. So node_df has 3 rows (new ids will be 0,1,2).
        # max_old_id = 5, id_map = [-1, -1, -1, -1, -1, -1], old_ids = [1, 4, 5]
        # new_ids = [0, 1, 2]
        # id_map = [-1, 0, -1, -1, 1, 2] (due to id_map[old_ids] = new_ids)
        # ----------------------------------------------------------------

        # ------------- build X from configured input fields -------------
        x_rows = []
        x_cols = [c for c in self.dataset_input.X_return_field_names if c != "Entry"]
        for vals in node_df[x_cols].itertuples(index=False, name=None):
            parts = [self._to_tensor(v, field_name=c, cast_float=True) for c, v in zip(x_cols, vals)]
            x_rows.append(torch.cat(parts, dim=0))
        x = torch.stack(x_rows, dim=0)
        # ----------------------------------------------------------------

        # ------------- build Y from configured target field -------------
        y_rows = []
        y_col = self.dataset_input.Y_return_field_name
        for v in node_df[self.dataset_input.Y_return_field_name]:
            y_rows.append(self._to_tensor(v, field_name=y_col, cast_float=True))
        y = torch.stack(y_rows, dim=0)
        # ----------------------------------------------------------------

        # 3) construct edge_index/edge_attr

        # ------------------ accumulator / helper vars ------------------
        edge_src: list[np.ndarray] = []
        edge_dst: list[np.ndarray] = []
        edge_attr_blocks: list[np.ndarray] = []
        chunk_name_pattern = re.compile(r"chunk_(\d+)\.csv$")
        edge_paths = [
            Path(p)
            for p in util.files_from(str(raw_dir), self.dataset_input.edge_csv_file_name_pattern)
        ]
        # ----------------------------------------------------------------

        # --------- get configured edge attrs or infer from files --------
        if self.dataset_input.edge_attr_columns is not None:
            edge_attr_cols = list(self.dataset_input.edge_attr_columns)
        else:
            edge_attr_cols = []
            for edge_path in edge_paths:
                header_cols = pd.read_csv(edge_path, nrows=0).columns.tolist()
                for col in header_cols:
                    # inferred from all non-dst columns
                    if col != self.dataset_input.edge_dst_column and col not in edge_attr_cols:
                        edge_attr_cols.append(col)
        # ----------------------------------------------------------------

        # --- process individual edge sets pruning out filtered nodes ----
        for edge_path in edge_paths:
            match = chunk_name_pattern.match(edge_path.name)
            if not match:
                continue

            src_old = int(match.group(1)) - 1 # make the src 0-indexed
            if src_old < 0 or src_old >= id_map.shape[0]:
                continue
            src_new = id_map[src_old] # get the new index
            
            if src_new < 0: # this node was dropped then, so move on
                continue

            edge_df = pd.read_csv(edge_path) # read the destinations
            if edge_df.empty:
                continue

            if self.dataset_input.edge_dst_column not in edge_df.columns:
                raise ValueError(
                    f"Edge file {edge_path} is missing '{self.dataset_input.edge_dst_column}'"
                )
            
            # make the dst 0-indexed
            dst_old = edge_df[self.dataset_input.edge_dst_column].to_numpy(dtype=np.int64) - 1

            in_bounds = (dst_old >= 0) & (dst_old < id_map.shape[0])
            if not in_bounds.any():
                continue

            # initialize dst_mapped with -1 everywhere
            dst_mapped = np.full(dst_old.shape, -1, dtype=np.int64)
            # for in-bounds destinations, assign id_map[dst_old] (new id or -1)
            dst_mapped[in_bounds] = id_map[dst_old[in_bounds]]
            # keep only those where dst is mapped (that is, dst node was kept)
            keep_edges = dst_mapped >= 0 # use it throw away -1 entries down the road
            if not keep_edges.any():
                continue
            # src_arr is just [src_new, src_new, ..., src_new] repeated once per kept edge
            src_arr = np.full(keep_edges.sum(), src_new, dtype=np.int64) # note keep_edges is a binary array
            dst_arr = dst_mapped[keep_edges] # is the mapped destination ids

            if edge_attr_cols:
                # Reindex allows missing columns in some chunks; missing attrs become 0.0.
                # reindex(columns=edge_attr_cols) ensures the attribute matrix has exactly those columns in that order
                attr_df = edge_df.reindex(columns=edge_attr_cols)
                attr_np = attr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(
                    dtype=np.float32
                ) # to_numeric(errors="coerce") converts strings to numbers; non-convertible becomes NaN
                attr_arr = attr_np[keep_edges] # keep attributes only for the kept edges
            else:
                attr_arr = np.empty((keep_edges.sum(), 0), dtype=np.float32) # (E, 0) -- if no attrs

            # source files represent upper triangle; add reverse edges for full message passing
            edge_src.append(src_arr)
            edge_dst.append(dst_arr)
            edge_attr_blocks.append(attr_arr)
            edge_src.append(dst_arr)
            edge_dst.append(src_arr)
            edge_attr_blocks.append(attr_arr)
        # ----------------------------------------------------------------

        # ---- collate the edge attrs and index and convert to torch -----
        if edge_src:
            edge_index_np = np.vstack([np.concatenate(edge_src), np.concatenate(edge_dst)])
            edge_attr_np = np.concatenate(edge_attr_blocks, axis=0)
        else:
            edge_index_np = np.empty((2, 0), dtype=np.int64)
            edge_attr_np = np.empty((0, len(edge_attr_cols)), dtype=np.float32)

        edge_index = torch.from_numpy(edge_index_np).long()
        edge_attr = torch.from_numpy(edge_attr_np).float()
        # ----------------------------------------------------------------

        # 4) store everything

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.node_id_to_accession = {
            int(i): acc for i, acc in enumerate(node_df["Entry"].astype(str).tolist())
        }
        data.x_fields = [i for i in self.dataset_input.X_return_field_names if i != "Entry"]
        data.y_field = self.dataset_input.Y_return_field_name
        data.edge_attr_fields = tuple(edge_attr_cols)

        if self.pre_filter is not None:
            data.pre_filter_applied = True
        if self.pre_transform is not None:
            data.pre_transform_applied = True

        # ---------------- attach train, val, test masks -----------------
        total_samples = x.size(dim=0)
        val_set_size = floor(self.val_set_size * total_samples)
        test_set_size = floor(self.test_set_size * total_samples)
        data_split_transform = RandomNodeSplit(num_val=val_set_size, num_test=test_set_size)
        split_data = data_split_transform(data)
        # ----------------------------------------------------------------

        torch.save(self.collate([split_data]), self.processed_paths[0])
        _logger.info(
            "Processed graph saved to %s (nodes=%d, edges=%d, x_dim=%d, y_dim=%d)",
            self.processed_paths[0],
            split_data.num_nodes,
            split_data.num_edges,
            split_data.x.size(-1),
            split_data.y.size(-1) if split_data.y.ndim > 1 else 1,
        )
    
    def train_loader(self):
        pass

    def val_loader(self):
        pass

    def test_loader(self):
        pass


class _ProteinGraphStore(GraphStore):

    def __init__(self, edge_attr_cls = None):
        super().__init__(edge_attr_cls)
        self.store: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = dict()

    @staticmethod
    def key(attr: EdgeAttr) -> tuple:
        return (attr.edge_type, attr.layout.value, attr.is_sorted, attr.size)

    def _put_edge_index(
        self,
        edge_index: EdgeTensorType,
        edge_attr: EdgeAttr,
    ) -> bool:
        self.store[self.key(edge_attr)] = edge_index
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self.store.get(self.key(edge_attr), None)

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        return self.store.pop(self.key(edge_attr), None) is not None

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        return [EdgeAttr(*key) for key in self.store.keys()]


class ProteinDataset:
    pass


__all__ = [

]


if __name__ == "__main__":
    pass
