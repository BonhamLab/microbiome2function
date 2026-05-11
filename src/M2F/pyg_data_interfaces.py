from __future__ import annotations

# third-party
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader as pt_DataLoader
from torch_geometric.data import \
    (InMemoryDataset, 
    Data,
    FeatureStore,
    GraphStore,
    TensorAttr,
    EdgeAttr)
from torch_geometric.typing import EdgeTensorType
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import pandas as pd

# built-in
from dataclasses import dataclass, field
from typing import Iterator, Any, Optional, Literal
from pathlib import Path
from math import floor
import re
import shutil
import logging

# local
from . import util
from .mining_utils import fetch_uniprotkb_fields

_logger = logging.getLogger(__name__)


def _to_numeric_vector(
    value: Any,
    *,
    field_name: str,
    cast_float: bool = True,
) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(f"Empty value for field '{field_name}'")
        arr = np.asarray(value)
    elif isinstance(value, (int, float, np.number, bool)):
        arr = np.asarray([value])
    else:
        raise TypeError(
            f"Field '{field_name}' has unsupported type {type(value)}. "
            "Apply a pre_transform that converts it to numeric arrays."
        )

    if arr.ndim == 0:
        arr = arr.reshape(1)

    arr = arr.reshape(-1)
    if cast_float:
        try:
            arr = arr.astype(np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Field '{field_name}' could not be converted to float32."
            ) from exc
    return arr


@dataclass
class DatasetInput:
    """
    Input contract consumed by PyG dataset interfaces (InMemory / OnDisk).

    Expected raw format:
    - accession index CSV: columns ['uniref', 'i'] (1-based node ids)
    - edge chunk CSVs (optional for non-graph use): file names like chunk_<i>.csv,
      must contain a destination id column (default: 'j'); all other columns can
      be used as edge attributes.
    - X: feature field names to query and their return names used as model inputs (e.g. {'sequence': 'Sequence'})
    - Y: target field name to query and its return name used as model output
    """
    # core dataset attrs
    path_to_accession_ids_csv_file: Path
    X: dict[str, str]
    Y: dict[str, str]
    path_to_edge_csv_dir: Path | None = None

    # internals -- uniprot query params
    request_size: int = 25
    rps: float = 1
    max_retry: int | float = 20
    num_feature_batches: int | None = None
    edge_dst_column: str | None = "j"
    edge_attr_columns: list[str] | tuple[str, ...] | None = None
    edge_csv_file_name_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"chunk_\d+\.csv")
    )

    # internals -- dataclass specifics (ensuring data quality)
    _validation_ctx: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _accession_ids_df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path_to_accession_ids_csv_file = Path(self.path_to_accession_ids_csv_file)
        if self.path_to_edge_csv_dir is not None:
            self.path_to_edge_csv_dir = Path(self.path_to_edge_csv_dir)
        self._normalize_edge_schema()
        self._normalize_xy()
        self.validate(require_graph=False)
        self.X["accession"] = "Entry" # <-- we always want to request accession

    def validate(self, *, require_graph: bool = False) -> None:
        self._validate_uniprot_request_params()
        self._validate_xy()
        self._validate_accession_ids_csv_file()
        if require_graph:
            self._validate_edge_csv_files()
    
    def _normalize_xy(self) -> None:
        self.X = {k.strip(): v.strip() for k, v in self.X.items()}
        self.Y = {k.strip(): v.strip() for k, v in self.Y.items()}

    def _normalize_edge_schema(self) -> None:
        if self.edge_dst_column is None:
            return
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
        if self.num_feature_batches is not None:
            if isinstance(self.num_feature_batches, bool) or not isinstance(self.num_feature_batches, int):
                raise TypeError("`num_feature_batches` must be int | None")
            if self.num_feature_batches < 1:
                raise ValueError("`num_feature_batches` must be >= 1 when provided")

        self._validation_ctx["request_size"] = self.request_size
        self._validation_ctx["rps"] = self.rps
        self._validation_ctx["max_retry"] = self.max_retry
        self._validation_ctx["num_feature_batches"] = self.num_feature_batches

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
        if self.path_to_edge_csv_dir is None:
            raise ValueError(
                "`path_to_edge_csv_dir` is required for graph datasets."
            )
        if self.edge_csv_file_name_pattern is None:
            raise ValueError(
                "`edge_csv_file_name_pattern` is required for graph datasets."
            )
        if self.edge_dst_column is None:
            raise ValueError(
                "`edge_dst_column` is required for graph datasets."
            )

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
        if self.path_to_edge_csv_dir is None:
            return
        if self.edge_csv_file_name_pattern is None:
            return
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


def build_topology_from_DatasetInput(
        id_map: np.ndarray,
        csv_dir: Path | str,
        edge_csv_file_name_pattern: re.Pattern,
        edge_attr_columns: list[str] | tuple[str, ...] | None,
        chunk_name_pattern: re.Pattern,
        edge_dst_column: str
        ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    # ------------------ accumulator / helper vars ------------------
    edge_src: list[np.ndarray] = []
    edge_dst: list[np.ndarray] = []
    edge_attr_blocks: list[np.ndarray] = []
    processed_chunks = 0
    kept_directed_edges = 0
    edge_paths = [
        Path(p)
        for p in util.files_from(str(csv_dir), edge_csv_file_name_pattern)
    ]
    _logger.info("Building topology from %d edge shard(s) in %s", len(edge_paths), csv_dir)
    # ----------------------------------------------------------------

    # --------- get configured edge attrs or infer from files --------
    if edge_attr_columns is not None:
        edge_attr_cols = list(edge_attr_columns)
    else:
        edge_attr_cols = []
        for edge_path in edge_paths:
            header_cols = pd.read_csv(edge_path, nrows=0).columns.tolist()
            for col in header_cols:
                # inferred from all non-dst columns
                if col != edge_dst_column and col not in edge_attr_cols:
                    edge_attr_cols.append(col)
    _logger.debug(
        "Using %d edge attribute column(s): %s",
        len(edge_attr_cols),
        edge_attr_cols[:10],
    )
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

        if edge_dst_column not in edge_df.columns:
            raise ValueError(
                f"Edge file {edge_path} is missing '{edge_dst_column}'"
            )
        
        # make the dst 0-indexed
        dst_old = edge_df[edge_dst_column].to_numpy(dtype=np.int64) - 1

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
        processed_chunks += 1
        kept_directed_edges += int(keep_edges.sum())
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
    _logger.info(
        "Built topology: kept_chunks=%d directed_edges=%d stored_edges=%d edge_attr_dim=%d",
        processed_chunks,
        kept_directed_edges,
        int(edge_index_np.shape[1]),
        int(edge_attr_np.shape[1]),
    )
    
    return edge_index_np, edge_attr_np, tuple(edge_attr_cols)
    # ----------------------------------------------------------------


def build_features_from_DatasetInput(
        pre_transform,
        pre_filter,
        accessions_path: Path | str,
        features_path: Path | str,
        required_cols: list[str],
        global_id_map: np.ndarray,
        X_return_field_names: list[str] | tuple[str, ...],
        Y_return_field_name: str) -> tuple[np.ndarray, np.ndarray]:
    _logger.info(
        "Building features from shard %s using accession index %s",
        features_path,
        accessions_path,
    )
    # ------------------------ read the data -------------------------
    accessions_df = pd.read_csv(accessions_path)
    features_df = pd.read_csv(features_path)
    _logger.debug(
        "Loaded accession rows=%d and feature rows=%d from shard %s",
        int(accessions_df.shape[0]),
        int(features_df.shape[0]),
        features_path,
    )
    # ----------------------------------------------------------------

    # ------------- align features with graph node order -------------
    if "Entry" not in features_df.columns:
        raise KeyError(
            f"{features_path} is missing merge key 'Entry'. "
            "UniProt fetch likely returned no usable schema."
        )
    features_df = features_df.copy()
    features_df["Entry"] = features_df["Entry"].astype(str).str.strip()
    dup_mask = features_df["Entry"].duplicated(keep=False)
    if dup_mask.any():
        duplicate_entries = sorted(features_df.loc[dup_mask, "Entry"].unique().tolist())
        preview = duplicate_entries[:10]
        raise ValueError(
            "Feature shard contains duplicate 'Entry' values; each accession "
            "must appear at most once per shard. "
            f"Found {len(duplicate_entries)} duplicated accession(s). "
            f"Examples: {preview}"
        )

    index_df = accessions_df.copy()
    index_df["Entry"] = index_df["uniref"].astype(str).str.replace("UniRef90_", "", regex=False)
    index_df["_orig_node_id"] = index_df["i"].astype(np.int64) - 1
    node_df = index_df.merge(features_df, on="Entry", how="left", sort=False)
    _logger.debug("Merged shard into node table with %d row(s)", int(node_df.shape[0]))
    # ----------------------------------------------------------------

    # ---------------------- transform the table ---------------------
    if pre_transform is not None:
        transformed = pre_transform(node_df)
        if not isinstance(transformed, pd.DataFrame):
            raise TypeError("`pre_transform` must return a pandas DataFrame in this interface")
        node_df = transformed
    # ----------------------------------------------------------------

    # --------------------- create the keep_mask ---------------------
    keep_mask = ~node_df["Entry"].astype(str).str.startswith(("UNK", "UPI"))
    if pre_filter is not None:
        filtered = pre_filter(node_df)
        if not isinstance(filtered, (pd.Series, np.ndarray, list, tuple)):
            raise TypeError("`pre_filter` must return a boolean mask for the node table")
        filtered = pd.Series(filtered, index=node_df.index)
        if filtered.shape[0] != node_df.shape[0]:
            raise ValueError("`pre_filter` mask length does not match number of nodes")
        keep_mask &= filtered.astype(bool)
    # ----------------------------------------------------------------

    # --------- always require non-missing supervised fields ---------
    missing_required = [col for col in required_cols if col not in node_df.columns]
    if missing_required:
        raise KeyError(f"Required columns missing after transform: {missing_required}")

    keep_mask &= ~node_df.loc[:, required_cols].isna().any(axis=1)
    node_df = node_df[keep_mask].copy()
    _logger.info(
        "Shard %s kept %d/%d node row(s) after filters",
        features_path,
        int(node_df.shape[0]),
        int(index_df.shape[0]),
    )
    # ----------------------------------------------------------------

    if node_df.empty:
        _logger.info("Shard %s produced no usable rows after filtering", features_path)
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
        )

    # --- update old->new global node id map (in-place) --------------
    old_ids = node_df["_orig_node_id"].to_numpy(dtype=np.int64)
    if old_ids.min() < 0 or old_ids.max() >= global_id_map.shape[0]:
        raise ValueError("Encountered node ids outside `global_id_map` bounds")

    if np.any(global_id_map[old_ids] >= 0):
        dup = old_ids[global_id_map[old_ids] >= 0][:5].tolist()
        raise ValueError(
            "Some old node ids were already assigned in `global_id_map`. "
            f"Example duplicates: {dup}"
        )

    next_new_id = int(global_id_map.max()) + 1 if global_id_map.size else 0
    new_ids = np.arange(next_new_id, next_new_id + old_ids.shape[0], dtype=np.int64)
    global_id_map[old_ids] = new_ids
    _logger.debug(
        "Assigned %d global node id(s) for shard %s (new_id_range=[%d,%d])",
        int(old_ids.shape[0]),
        features_path,
        int(new_ids[0]),
        int(new_ids[-1]),
    )
    # ----------------------------------------------------------------

    # ------------------------ build X and Y -------------------------
    x_cols = [c for c in X_return_field_names if c != "Entry"]
    y_col = Y_return_field_name

    x_rows: list[np.ndarray] = []
    for vals in node_df[x_cols].itertuples(index=False, name=None):
        parts = [_to_numeric_vector(v, field_name=c, cast_float=True) for c, v in zip(x_cols, vals)]
        x_rows.append(np.concatenate(parts, axis=0))

    y_rows: list[np.ndarray] = []
    for v in node_df[y_col]:
        y_rows.append(_to_numeric_vector(v, field_name=y_col, cast_float=True))

    if x_rows:
        x_dim = x_rows[0].shape[0]
        if any(row.shape[0] != x_dim for row in x_rows):
            raise ValueError("X feature dimensionality is not consistent across rows")
        x = np.vstack(x_rows).astype(np.float32, copy=False)
    else:
        x = np.empty((0, 0), dtype=np.float32)

    if y_rows:
        y_dim = y_rows[0].shape[0]
        if any(row.shape[0] != y_dim for row in y_rows):
            raise ValueError("Y dimensionality is not consistent across rows")
        y = np.vstack(y_rows).astype(np.float32, copy=False)
    else:
        y = np.empty((0, 0), dtype=np.float32)

    _logger.info(
        "Built feature tensors for shard %s: x_shape=%s y_shape=%s",
        features_path,
        tuple(x.shape),
        tuple(y.shape),
    )
    return x, y
    # ----------------------------------------------------------------


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
        self.dataset_input.validate(require_graph=True)

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
        _logger.info("Preparing raw data for ProteinGraphInMemoryDataset at %s", raw_dir)
        if not features_path.exists():
            _logger.info(
                "Fetching UniProt features for %d accession(s) into %s",
                len(self.original_node_accessions),
                features_path,
            )
            fetched_features = fetch_uniprotkb_fields(
                        uniref_ids=self.original_node_accessions,
                        fields=[self.dataset_input.Y_query_field_name, *self.dataset_input.X_query_field_names],
                        request_size=self.dataset_input.request_size,
                        rps=self.dataset_input.rps,
                        max_retry=self.dataset_input.max_retry
                    )
            fetched_features.to_csv(features_path, index=False)
        else:
            _logger.debug("Reusing existing raw feature file at %s", features_path)

        # put index + edge files into raw/ so raw_file_names is satisfied
        self._materialize(
            self.dataset_input.path_to_accession_ids_csv_file,
            raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name,
        )
        edge_count = 0
        for edge_path in self.dataset_input.edge_csv_paths:
            self._materialize(edge_path, raw_dir / edge_path.name)
            edge_count += 1
        _logger.debug("Materialized accession index and %d edge shard(s) into raw dir", edge_count)

    @staticmethod
    def _to_tensor(value: Any, *, field_name: str, cast_float: bool = True) -> torch.Tensor:
        arr = _to_numeric_vector(value, field_name=field_name, cast_float=cast_float)
        return torch.from_numpy(arr)

    def process(self) -> None:
        # ------------------------- get the paths ------------------------
        raw_dir = Path(self.raw_dir)
        features_path = raw_dir / "features.csv"
        accessions_path = raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name
        _logger.info("Processing ProteinGraphInMemoryDataset from %s", raw_dir)
        if not features_path.exists():
            raise FileNotFoundError(f"Expected raw features at {features_path}")
        if not accessions_path.exists():
            raise FileNotFoundError(f"Expected accession index at {accessions_path}")
        # ----------------------------------------------------------------

        # ---------------------------- features --------------------------
        accessions_df = pd.read_csv(accessions_path)
        max_old_id = int(accessions_df["i"].max()) - 1
        id_map = np.full(max_old_id + 1, -1, dtype=np.int64)

        required_cols = [self.dataset_input.Y_return_field_name, *self.dataset_input.X_return_field_names]
        x_np, y_np = build_features_from_DatasetInput(
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
            accessions_path=accessions_path,
            features_path=features_path,
            required_cols=required_cols,
            global_id_map=id_map,
            X_return_field_names=self.dataset_input.X_return_field_names,
            Y_return_field_name=self.dataset_input.Y_return_field_name
        )

        if x_np.shape[0] == 0:
            raise ValueError("All nodes were filtered out; cannot build dataset")
        _logger.info(
            "Feature build complete: nodes=%d x_dim=%d y_dim=%d",
            int(x_np.shape[0]),
            int(x_np.shape[1]),
            int(y_np.shape[1]) if y_np.ndim > 1 else 1,
        )

        x = torch.from_numpy(x_np).float()
        y = torch.from_numpy(y_np).float()
        # ----------------------------------------------------------------

        # ------------------------------ edges ---------------------------
        chunk_name_pattern = self.dataset_input.edge_csv_file_name_pattern
        if chunk_name_pattern.groups < 1:
            chunk_name_pattern = re.compile(r"chunk_(\d+)\.csv$")

        edge_index_np, edge_attr_np, edge_attr_cols = build_topology_from_DatasetInput(
            id_map=id_map,
            csv_dir=raw_dir,
            edge_csv_file_name_pattern=self.dataset_input.edge_csv_file_name_pattern,
            edge_attr_columns=self.dataset_input.edge_attr_columns,
            chunk_name_pattern=chunk_name_pattern,
            edge_dst_column=self.dataset_input.edge_dst_column,
        )
        _logger.info(
            "Topology build complete: edges=%d edge_attr_dim=%d",
            int(edge_index_np.shape[1]),
            int(edge_attr_np.shape[1]),
        )
        edge_index = torch.from_numpy(edge_index_np).long()
        edge_attr = torch.from_numpy(edge_attr_np).float()
        # ----------------------------------------------------------------

        node_id_to_accession: dict[int, str] = {}
        for row in accessions_df.itertuples(index=False):
            old_id = int(row.i) - 1
            if old_id < 0 or old_id >= id_map.shape[0]:
                continue
            new_id = int(id_map[old_id])
            if new_id < 0:
                continue
            node_id_to_accession[new_id] = str(row.uniref).replace("UniRef90_", "", 1)

        if len(node_id_to_accession) != x.size(0):
            raise RuntimeError(
                "Node metadata mapping size mismatch after filtering/reindexing: "
                f"{len(node_id_to_accession)} mapped nodes vs {x.size(0)} feature rows."
            )

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.node_id_to_accession = dict(sorted(node_id_to_accession.items()))
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
        _logger.debug(
            "Split sizes: train=%d val=%d test=%d",
            int(split_data.train_mask.sum().item()),
            int(split_data.val_mask.sum().item()),
            int(split_data.test_mask.sum().item()),
        )
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
    
    def train_loader(self, num_neighbors: list[int], batch_size: int, shuffle: bool) -> NeighborLoader:
        return NeighborLoader(
            self[0],
            num_neighbors=num_neighbors,
            input_nodes=self[0].train_mask,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def val_loader(self, num_neighbors: list[int], batch_size: int) -> NeighborLoader:
        return NeighborLoader(
            self[0],
            num_neighbors=num_neighbors,
            input_nodes=self[0].val_mask,
            batch_size=batch_size,
            shuffle=False,
        )

    def test_loader(self, num_neighbors: list[int], batch_size: int) -> NeighborLoader:
        return NeighborLoader(
            self[0],
            num_neighbors=num_neighbors,
            input_nodes=self[0].test_mask,
            batch_size=batch_size,
            shuffle=False,
        )


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


class _ProteinFeatureStore(FeatureStore):
    def __init__(self,
                store_on_disk_location: Path | str,
                node_feature_dim: tuple[int, ...],
                edge_feature_dim: tuple[int, ...],
                target_feature_dim: tuple[int, ...],
                read_only: bool = False) -> None:
        super().__init__()
        self.store = util.ZarrFeatureStore(store_on_disk_location, read_only)
        self._read_only = read_only
        self._default_feature_shapes: dict[str, tuple[int, ...]] = {
            "x": (0, *node_feature_dim),
            "y": (0, *target_feature_dim),
            "edge_attr": (0, *edge_feature_dim),
        }
        _logger.debug(
            "Initialized _ProteinFeatureStore(path=%s, read_only=%s, default_shapes=%s)",
            store_on_disk_location,
            read_only,
            self._default_feature_shapes,
        )

        if not self._read_only:
            existing = self.store.which_tensors
            for name, shape in self._default_feature_shapes.items():
                if name in existing:
                    continue
                self.store.add_location(TensorAttr(None, name, None), shape)

    @staticmethod
    def _normalize(T: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(T, torch.Tensor):
            data = T.detach().cpu().numpy()
        else:
            data = np.asarray(T)
        return data

    def _is_full_index(self, index: Any) -> bool:
        return index is None or (isinstance(index, slice) and index == slice(None))

    def _require_attr_name(self, attr: TensorAttr) -> str:
        if attr.attr_name is None:
            raise ValueError("TensorAttr.attr_name cannot be None")
        return attr.attr_name

    def _ensure_location_for_replace(self, attr: TensorAttr, data: np.ndarray) -> None:
        name = self._require_attr_name(attr)
        overwrite = name in self.store.which_tensors
        self.store.add_location(attr, shape=tuple(data.shape), dtype=data.dtype, overwrite=overwrite)

    def _put_tensor(self, tensor: torch.Tensor | np.ndarray, attr: TensorAttr) -> bool:
        if self._read_only:
            raise PermissionError("Cannot write to _ProteinFeatureStore opened in read_only mode")

        data = self._normalize(tensor)
        if data.ndim == 0:
            data = data.reshape(1)

        name = self._require_attr_name(attr)
        index = attr.index

        if self._is_full_index(index):
            self._ensure_location_for_replace(attr, data)
            self.store.add_data_to_location(data, TensorAttr(attr.group_name, name, slice(None)))
            _logger.debug("Replaced tensor '%s' with shape=%s", name, tuple(data.shape))
            return True

        if name not in self.store.which_tensors:
            raise KeyError(
                f"Location '{name}' does not exist. Use a full put first, "
                "or `append_tensor(...)` for streaming growth."
            )
        self.store.add_data_to_location(data, attr)
        _logger.debug("Updated tensor '%s' at index=%s with shape=%s", name, attr.index, tuple(data.shape))
        return True

    def append_tensor(self, tensor: torch.Tensor | np.ndarray, *args, **kwargs) -> slice:
        """
        Append rows to an existing tensor location (or create it if absent).
        This is the streaming-friendly API for mining-time growth.
        """
        if self._read_only:
            raise PermissionError("Cannot append to _ProteinFeatureStore opened in read_only mode")

        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        name = self._require_attr_name(attr)

        data = self._normalize(tensor)
        if data.ndim == 0: # we don't work with scalars: convert () into (1,)
            data = data.reshape(1)

        if name not in self.store.which_tensors:
            if data.ndim == 1:
                init_shape = (0, data.shape[0])
            else:
                init_shape = (0, *data.shape[1:])
            self.store.add_location(TensorAttr(attr.group_name, name, None), init_shape, dtype=data.dtype)

        out = self.store.append(data, TensorAttr(attr.group_name, name, None))
        _logger.debug("Appended tensor '%s' with shape=%s at slice=%s", name, tuple(data.shape), out)
        return out

    def _get_tensor(self, attr: TensorAttr) -> Optional[torch.Tensor]:
        name = self._require_attr_name(attr)
        if name not in self.store.which_tensors:
            raise KeyError(f"Could not find tensor for '{attr}'")
        out = self.store.read_data_from_location(attr)
        return torch.from_numpy(out)

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        if self._read_only:
            raise PermissionError("Cannot remove from _ProteinFeatureStore opened in read_only mode")

        name = self._require_attr_name(attr)
        if name not in self.store.which_tensors:
            return False

        if self._is_full_index(attr.index):
            deleted = self.store.drop_location(attr)
            _logger.debug("Dropped full tensor '%s' (deleted=%s)", name, deleted)
            return deleted

        self.store.remove_data_from_location(attr)
        _logger.debug("Marked entries removed in tensor '%s' at index=%s", name, attr.index)
        return True

    def _get_tensor_size(self, attr: TensorAttr) -> Optional[tuple[int, ...]]:
        name = self._require_attr_name(attr)
        tensor_meta = self.store.which_tensors.get(name)
        if tensor_meta is None:
            return None

        shape, _ = tensor_meta
        if self._is_full_index(attr.index):
            return tuple(shape)

        indexed = self.store.read_data_from_location(attr)
        return tuple(indexed.shape)

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        # `edge_attr` is edge-level and must not be fetched with node indices in
        # PyG remote backend filtering; it is attached explicitly by loader transforms.
        return [
            TensorAttr(None, name, None)
            for name in self.store.which_tensors.keys()
            if name != "edge_attr"
        ]

    def close(self) -> None:
        _logger.debug("Closing _ProteinFeatureStore")
        self.store.close()


# OnDisk data interface for GNNs
class ProteinGraphOnDiskDataset:

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
        self.root = Path(root)
        self.dataset_input = dataset_input
        self.dataset_input.validate(require_graph=True)

        if ((val_set_size + test_set_size) >= 1.0
            or val_set_size < 0.0
            or test_set_size < 0.0):
            raise ValueError()

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self.force_reload = force_reload
        self.val_set_size = val_set_size
        self.test_set_size = test_set_size

        self.feature_store: _ProteinFeatureStore | None = None
        self.graph_store: _ProteinGraphStore | None = None
        self.id_map: np.ndarray | None = None
        self.train_node_ids: torch.Tensor | None = None
        self.val_node_ids: torch.Tensor | None = None
        self.test_node_ids: torch.Tensor | None = None
        self.meta: dict[str, Any] = {}
        if self.log:
            _logger.info(
                "Initializing ProteinGraphOnDiskDataset(root=%s, force_reload=%s, num_feature_batches=%d)",
                self.root,
                self.force_reload,
                self.num_feature_batches,
            )

        if self.force_reload:
            _logger.info("force_reload=True; clearing existing raw feature batches and processed artifacts")
            if self.features_batches_dir.exists():
                shutil.rmtree(self.features_batches_dir)
            if self.processed_dir.exists():
                shutil.rmtree(self.processed_dir)

        if not self._raw_ready():
            _logger.info("Raw artifacts are missing; starting download()")
            self.download()
        elif self.log:
            _logger.debug("Raw artifacts are already present; skipping download()")

        if self.force_reload or not self._processed_ready():
            _logger.info("Processed artifacts are missing/stale; starting process()")
            self.process()
        elif self.log:
            _logger.debug("Processed artifacts are already present; skipping process()")

        self._load_processed()

    @property
    def original_node_accessions(self):
        return [
            str(row.uniref).replace("UniRef90_", "", 1)
            for row in self.dataset_input.accession_ids.itertuples(index=False)
            if not str(row.uniref).startswith(("UniRef90_UNK", "UniRef90_UPI"))
        ]

    @property
    def raw_file_names(self) -> list[str]:
        feature_batch_files = [f"features_{i}.csv" for i in range(1, self.num_feature_batches + 1)]
        return [
            self.dataset_input.path_to_accession_ids_csv_file.name,
            *[path.name for path in self.dataset_input.edge_csv_paths],
            *[str(Path("features_batches") / name) for name in feature_batch_files],
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return [
            "feature_store/zarr.json",
            "edge_index.npy",
            "id_map.npy",
            "meta.pt"
        ]

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def features_batches_dir(self) -> Path:
        return self.raw_dir / "features_batches"

    @property
    def feature_store_dir(self) -> Path:
        return self.processed_dir / "feature_store"

    @property
    def edge_index_path(self) -> Path:
        return self.processed_dir / "edge_index.npy"

    @property
    def id_map_path(self) -> Path:
        return self.processed_dir / "id_map.npy"

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / "meta.pt"

    @property
    def num_feature_batches(self) -> int:
        requested = self.dataset_input.num_feature_batches or 1
        total_ids = max(1, len(self.original_node_accessions))
        return min(requested, total_ids)

    def _feature_batch_path(self, batch_id_1based: int) -> Path:
        return self.features_batches_dir / f"features_{batch_id_1based}.csv"

    def _raw_ready(self) -> bool:
        if not self.raw_dir.exists():
            return False

        expected = [self.raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name]
        expected.extend([self.raw_dir / path.name for path in self.dataset_input.edge_csv_paths])
        expected.extend([self._feature_batch_path(i) for i in range(1, self.num_feature_batches + 1)])
        return all(path.exists() for path in expected)

    def _processed_ready(self) -> bool:
        expected = [self.processed_dir / name for name in self.processed_file_names]
        return all(path.exists() for path in expected)

    def _load_processed(self) -> None:
        meta = torch.load(self.meta_path, weights_only=False)
        self.meta = meta

        x_dim = int(meta["x_dim"])
        y_dim = int(meta["y_dim"])
        edge_attr_dim = int(meta["edge_attr_dim"])

        self.feature_store = _ProteinFeatureStore(
            store_on_disk_location=self.feature_store_dir,
            node_feature_dim=(x_dim,),
            edge_feature_dim=(edge_attr_dim,),
            target_feature_dim=(y_dim,),
            read_only=True
        )

        edge_index_np = np.load(self.edge_index_path)
        row = torch.from_numpy(edge_index_np[0]).long()
        col = torch.from_numpy(edge_index_np[1]).long()
        num_nodes = int(meta["num_nodes"])
        self.graph_store = _ProteinGraphStore()
        self.graph_store.put_edge_index(
            (row, col),
            edge_type=None,
            layout="coo",
            is_sorted=False,
            size=(num_nodes, num_nodes),
        )

        self.id_map = np.load(self.id_map_path)
        self.train_node_ids = torch.as_tensor(meta["train_node_ids"]).long()
        self.val_node_ids = torch.as_tensor(meta["val_node_ids"]).long()
        self.test_node_ids = torch.as_tensor(meta["test_node_ids"]).long()
        if self.log:
            _logger.info(
                "Loaded on-disk graph dataset from %s (nodes=%d, edges=%d)",
                self.processed_dir,
                int(meta["num_nodes"]),
                int(meta["num_edges"]),
            )

    @staticmethod
    def _materialize(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    def download(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.features_batches_dir.mkdir(parents=True, exist_ok=True)

        # Materialize index + edge files into raw/
        self._materialize(
            self.dataset_input.path_to_accession_ids_csv_file,
            self.raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name,
        )
        for edge_path in self.dataset_input.edge_csv_paths:
            self._materialize(edge_path, self.raw_dir / edge_path.name)

        accessions = self.original_node_accessions
        if len(accessions) == 0:
            raise ValueError(
                "No valid UniRef accessions to query after removing UNK/UPI entries."
            )
        _logger.info(
            "Downloading feature shards for ProteinGraphOnDiskDataset: accessions=%d batches=%d",
            len(accessions),
            self.num_feature_batches,
        )
        batches = np.array_split(np.asarray(accessions, dtype=object), self.num_feature_batches)
        fields = [self.dataset_input.Y_query_field_name, *self.dataset_input.X_query_field_names]

        for i, batch in enumerate(batches, start=1):
            batch_path = self._feature_batch_path(i)
            if batch_path.exists():
                _logger.debug("Reusing existing feature shard %s", batch_path)
                continue

            batch_ids = [str(x) for x in batch.tolist()]
            if len(batch_ids) == 0:
                pd.DataFrame(columns=fields).to_csv(batch_path, index=False)
                _logger.debug("Wrote empty feature shard %s", batch_path)
                continue
            _logger.info(
                "Fetching feature shard %d/%d with %d accession(s)",
                i,
                self.num_feature_batches,
                len(batch_ids),
            )
            features_df = fetch_uniprotkb_fields(
                uniref_ids=batch_ids,
                fields=fields,
                request_size=self.dataset_input.request_size,
                rps=self.dataset_input.rps,
                max_retry=self.dataset_input.max_retry,
            )
            features_df.to_csv(batch_path, index=False)
            _logger.debug("Saved feature shard %s with %d row(s)", batch_path, int(features_df.shape[0]))

    @staticmethod
    def _to_tensor(value: Any, *, field_name: str, cast_float: bool = True) -> torch.Tensor:
        arr = _to_numeric_vector(value, field_name=field_name, cast_float=cast_float)
        return torch.from_numpy(arr)

    def _attach_edge_attr(self, batch: Data) -> Data:
        if self.feature_store is None:
            raise RuntimeError("Feature store is not initialized")
        if "e_id" in batch:
            edge_attr = self.feature_store.get_tensor(
                TensorAttr(None, "edge_attr", batch.e_id.detach().cpu())
            )
            batch.edge_attr = edge_attr.float()
        else:
            batch.edge_attr = torch.empty((batch.edge_index.size(1), 0), dtype=torch.float32)
        return batch

    def process(self) -> None:
        # ------------------------- get the paths ------------------------
        accessions_path = self.raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name
        _logger.info("Processing ProteinGraphOnDiskDataset into %s", self.processed_dir)
        if not accessions_path.exists():
            raise FileNotFoundError(f"Expected accession index at {accessions_path}")
        for i in range(1, self.num_feature_batches + 1):
            batch_path = self._feature_batch_path(i)
            if not batch_path.exists():
                raise FileNotFoundError(f"Expected raw feature batch at {batch_path}")
        # ----------------------------------------------------------------

        # If the dataset was already loaded in read mode, release the handle
        # before rebuilding the processed artifacts
        self.close()
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir) # ensures no stale state mixes with new processing
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)

        accessions_df = pd.read_csv(accessions_path)
        max_old_id = int(accessions_df["i"].max()) - 1
        id_map = np.full(max_old_id + 1, -1, dtype=np.int64)

        required_cols = [self.dataset_input.Y_return_field_name, *self.dataset_input.X_return_field_names]

        feature_store: _ProteinFeatureStore | None = None
        total_rows = 0
        try:
            # ---------------------- pass 1: node features -------------------
            for i in range(1, self.num_feature_batches + 1):
                batch_path = self._feature_batch_path(i)
                x_np, y_np = build_features_from_DatasetInput(
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter,
                    accessions_path=accessions_path,
                    features_path=batch_path,
                    required_cols=required_cols,
                    global_id_map=id_map,
                    X_return_field_names=self.dataset_input.X_return_field_names,
                    Y_return_field_name=self.dataset_input.Y_return_field_name,
                )

                if x_np.shape[0] == 0:
                    _logger.debug("Feature shard %s produced 0 kept rows; skipping", batch_path)
                    continue

                if feature_store is None:
                    feature_store = _ProteinFeatureStore(
                        store_on_disk_location=self.feature_store_dir,
                        node_feature_dim=(x_np.shape[1],),
                        edge_feature_dim=(0,),  # rewritten after topology construction
                        target_feature_dim=(y_np.shape[1],),
                        read_only=False,
                    )
                else:
                    if x_np.shape[1] != int(feature_store.store.which_tensors["x"][0][1]):
                        raise ValueError("Inconsistent X feature dimensionality across feature batches")
                    if y_np.shape[1] != int(feature_store.store.which_tensors["y"][0][1]):
                        raise ValueError("Inconsistent Y dimensionality across feature batches")

                feature_store.append_tensor(x_np, group_name=None, attr_name="x", index=None)
                feature_store.append_tensor(y_np, group_name=None, attr_name="y", index=None)
                total_rows += int(x_np.shape[0])
                _logger.info(
                    "Processed feature shard %d/%d -> rows=%d x_dim=%d y_dim=%d (total_rows=%d)",
                    i,
                    self.num_feature_batches,
                    int(x_np.shape[0]),
                    int(x_np.shape[1]),
                    int(y_np.shape[1]) if y_np.ndim > 1 else 1,
                    total_rows,
                )
            # ----------------------------------------------------------------

            if feature_store is None or total_rows == 0:
                raise ValueError("All nodes were filtered out; cannot build on-disk dataset")
            _logger.info("Feature pass complete: total_nodes=%d", total_rows)

            # ---------------------- pass 2: topology ------------------------
            chunk_name_pattern = self.dataset_input.edge_csv_file_name_pattern
            if chunk_name_pattern.groups < 1:
                chunk_name_pattern = re.compile(r"chunk_(\d+)\.csv$")

            edge_index_np, edge_attr_np, edge_attr_cols = build_topology_from_DatasetInput(
                id_map=id_map,
                csv_dir=self.raw_dir,
                edge_csv_file_name_pattern=self.dataset_input.edge_csv_file_name_pattern,
                edge_attr_columns=self.dataset_input.edge_attr_columns,
                chunk_name_pattern=chunk_name_pattern,
                edge_dst_column=self.dataset_input.edge_dst_column,
            )
            _logger.info(
                "Topology pass complete: edges=%d edge_attr_dim=%d",
                int(edge_index_np.shape[1]),
                int(edge_attr_np.shape[1]),
            )

            # Recreate edge_attr location with the inferred dimensionality and append values
            feature_store.remove_tensor(TensorAttr(None, "edge_attr", None))
            feature_store.append_tensor(edge_attr_np, group_name=None, attr_name="edge_attr", index=None)
            # ----------------------------------------------------------------

            # ---------------------------- splits ----------------------------
            y_full = feature_store.get_tensor(TensorAttr(None, "y", slice(None))).float()
            dummy_x = torch.zeros((total_rows, 1), dtype=torch.float32)
            split_seed = Data(x=dummy_x, y=y_full)
            val_set_size = floor(self.val_set_size * total_rows)
            test_set_size = floor(self.test_set_size * total_rows)
            split = RandomNodeSplit(num_val=val_set_size, num_test=test_set_size)(split_seed)
            train_node_ids = split.train_mask.nonzero(as_tuple=False).view(-1).long()
            val_node_ids = split.val_mask.nonzero(as_tuple=False).view(-1).long()
            test_node_ids = split.test_mask.nonzero(as_tuple=False).view(-1).long()
            _logger.debug(
                "Split sizes: train=%d val=%d test=%d",
                int(train_node_ids.numel()),
                int(val_node_ids.numel()),
                int(test_node_ids.numel()),
            )
            # ----------------------------------------------------------------

            # ---------------------- node metadata map -----------------------
            node_id_to_accession: dict[int, str] = {}
            for row in accessions_df.itertuples(index=False):
                old_id = int(row.i) - 1
                if old_id < 0 or old_id >= id_map.shape[0]:
                    continue
                new_id = int(id_map[old_id])
                if new_id < 0:
                    continue
                node_id_to_accession[new_id] = str(row.uniref).replace("UniRef90_", "", 1)

            if len(node_id_to_accession) != total_rows:
                raise RuntimeError(
                    "Node metadata mapping size mismatch after filtering/reindexing: "
                    f"{len(node_id_to_accession)} mapped nodes vs {total_rows} feature rows."
                )
            # ----------------------------------------------------------------

            np.save(self.edge_index_path, edge_index_np)
            np.save(self.id_map_path, id_map)

            meta = {
                "x_fields": [i for i in self.dataset_input.X_return_field_names if i != "Entry"],
                "y_field": self.dataset_input.Y_return_field_name,
                "edge_attr_fields": tuple(edge_attr_cols),
                "num_nodes": int(total_rows),
                "num_edges": int(edge_index_np.shape[1]),
                "x_dim": int(feature_store.store.which_tensors["x"][0][1]),
                "y_dim": int(feature_store.store.which_tensors["y"][0][1]),
                "edge_attr_dim": int(edge_attr_np.shape[1]),
                "train_node_ids": train_node_ids.cpu(),
                "val_node_ids": val_node_ids.cpu(),
                "test_node_ids": test_node_ids.cpu(),
                "node_id_to_accession": dict(sorted(node_id_to_accession.items())),
            }
            torch.save(meta, self.meta_path)

            if self.log:
                _logger.info(
                    "Processed on-disk graph at %s (nodes=%d, edges=%d, x_dim=%d, y_dim=%d, edge_attr_dim=%d)",
                    self.processed_dir,
                    meta["num_nodes"],
                    meta["num_edges"],
                    meta["x_dim"],
                    meta["y_dim"],
                    meta["edge_attr_dim"],
                )
        finally:
            if feature_store is not None:
                feature_store.close()

    def _loader_transform(self, batch: Data) -> Data:
        batch = self._attach_edge_attr(batch)
        if self.transform is not None:
            maybe_batch = self.transform(batch)
            if maybe_batch is not None:
                batch = maybe_batch
        return batch
    
    def train_loader(self, num_neighbors: list[int], batch_size: int, shuffle: bool) -> NeighborLoader:
        if self.feature_store is None or self.graph_store is None or self.train_node_ids is None:
            raise RuntimeError("ProteinGraphOnDiskDataset is not initialized")
        return NeighborLoader(
            (self.feature_store, self.graph_store),
            num_neighbors=num_neighbors,
            input_nodes=self.train_node_ids,
            batch_size=batch_size,
            shuffle=shuffle,
            transform=self._loader_transform,
        )

    def val_loader(self, num_neighbors: list[int], batch_size: int) -> NeighborLoader:
        if self.feature_store is None or self.graph_store is None or self.val_node_ids is None:
            raise RuntimeError("ProteinGraphOnDiskDataset is not initialized")
        return NeighborLoader(
            (self.feature_store, self.graph_store),
            num_neighbors=num_neighbors,
            input_nodes=self.val_node_ids,
            batch_size=batch_size,
            shuffle=False,
            transform=self._loader_transform,
        )

    def test_loader(self, num_neighbors: list[int], batch_size: int) -> NeighborLoader:
        if self.feature_store is None or self.graph_store is None or self.test_node_ids is None:
            raise RuntimeError("ProteinGraphOnDiskDataset is not initialized")
        return NeighborLoader(
            (self.feature_store, self.graph_store),
            num_neighbors=num_neighbors,
            input_nodes=self.test_node_ids,
            batch_size=batch_size,
            shuffle=False,
            transform=self._loader_transform,
        )

    def close(self) -> None:
        if self.feature_store is not None:
            _logger.debug("Closing ProteinGraphOnDiskDataset feature store")
            self.feature_store.close()
            self.feature_store = None


class _ProteinDatasetView(Dataset):
    def __init__(
        self,
        parent: ProteinDataset,
        node_ids: torch.Tensor,
        include_targets: bool,
    ) -> None:
        self.parent = parent
        self.node_ids = node_ids.detach().cpu().long()
        self.include_targets = include_targets

    def __len__(self) -> int:
        return int(self.node_ids.numel())

    def __getitem__(self, idx: int):
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Index out of range: {idx}")
        node_id = int(self.node_ids[idx].item())
        return self.parent._getitem_by_node_id(node_id, include_targets=self.include_targets)


# Data interface for FFNNs (loading batches from disk, so not the whole thing in RAM)
class ProteinDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        dataset_input: DatasetInput,
        transform=None,
        target_transform=None,
        pre_transform=None,
        pre_filter=None,
        log: bool = True,
        force_reload: bool = False,
        *,
        val_set_size: float = 0.1,
        test_set_size: float = 0.1,
        split: Literal["train", "val", "test", "all"] = "train",
        include_targets: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.dataset_input = dataset_input
        self.dataset_input.validate(require_graph=False)

        if ((val_set_size + test_set_size) >= 1.0
            or val_set_size < 0.0
            or test_set_size < 0.0):
            raise ValueError()

        self.transform = transform
        self.target_transform = target_transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self.force_reload = force_reload
        self.val_set_size = val_set_size
        self.test_set_size = test_set_size

        self.feature_store: _ProteinFeatureStore | None = None
        self.id_map: np.ndarray | None = None
        self.train_node_ids: torch.Tensor | None = None
        self.val_node_ids: torch.Tensor | None = None
        self.test_node_ids: torch.Tensor | None = None
        self.all_node_ids: torch.Tensor | None = None
        self.active_node_ids: torch.Tensor | None = None
        self.split: Literal["train", "val", "test", "all"] = split
        self.include_targets = bool(split == "train") if include_targets is None else bool(include_targets)
        self.meta: dict[str, Any] = {}
        if self.log:
            _logger.info(
                "Initializing ProteinDataset(root=%s, force_reload=%s, num_feature_batches=%d, split=%s)",
                self.root,
                self.force_reload,
                self.num_feature_batches,
                split,
            )

        if self.force_reload:
            _logger.info("force_reload=True; clearing existing raw feature batches and processed_ffnn artifacts")
            if self.features_batches_dir.exists():
                shutil.rmtree(self.features_batches_dir)
            if self.processed_dir.exists():
                shutil.rmtree(self.processed_dir)

        if not self._raw_ready():
            _logger.info("Raw artifacts are missing; starting download()")
            self.download()
        elif self.log:
            _logger.debug("Raw artifacts are already present; skipping download()")

        if self.force_reload or not self._processed_ready():
            _logger.info("Processed artifacts are missing/stale; starting process()")
            self.process()
        elif self.log:
            _logger.debug("Processed artifacts are already present; skipping process()")

        self._load_processed()
        self.set_split(split=split, include_targets=include_targets)

    @property
    def original_node_accessions(self):
        return [
            str(row.uniref).replace("UniRef90_", "", 1)
            for row in self.dataset_input.accession_ids.itertuples(index=False)
            if not str(row.uniref).startswith(("UniRef90_UNK", "UniRef90_UPI"))
        ]

    @property
    def raw_file_names(self) -> list[str]:
        feature_batch_files = [f"features_{i}.csv" for i in range(1, self.num_feature_batches + 1)]
        return [
            self.dataset_input.path_to_accession_ids_csv_file.name,
            *[str(Path("features_batches") / name) for name in feature_batch_files],
        ]

    @property
    def processed_file_names(self) -> list[str]:
        return [
            "feature_store/zarr.json",
            "id_map.npy",
            "meta.pt"
        ]

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed_ffnn"

    @property
    def features_batches_dir(self) -> Path:
        return self.raw_dir / "features_batches"

    @property
    def feature_store_dir(self) -> Path:
        return self.processed_dir / "feature_store"

    @property
    def id_map_path(self) -> Path:
        return self.processed_dir / "id_map.npy"

    @property
    def meta_path(self) -> Path:
        return self.processed_dir / "meta.pt"

    @property
    def num_feature_batches(self) -> int:
        requested = self.dataset_input.num_feature_batches or 1
        total_ids = max(1, len(self.original_node_accessions))
        return min(requested, total_ids)

    def _feature_batch_path(self, batch_id_1based: int) -> Path:
        return self.features_batches_dir / f"features_{batch_id_1based}.csv"

    def _raw_ready(self) -> bool:
        if not self.raw_dir.exists():
            return False

        expected = [self.raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name]
        expected.extend([self._feature_batch_path(i) for i in range(1, self.num_feature_batches + 1)])
        return all(path.exists() for path in expected)

    def _processed_ready(self) -> bool:
        expected = [self.processed_dir / name for name in self.processed_file_names]
        return all(path.exists() for path in expected)

    def _load_processed(self) -> None:
        meta = torch.load(self.meta_path, weights_only=False)
        self.meta = meta

        x_dim = int(meta["x_dim"])
        y_dim = int(meta["y_dim"])

        self.feature_store = _ProteinFeatureStore(
            store_on_disk_location=self.feature_store_dir,
            node_feature_dim=(x_dim,),
            edge_feature_dim=(0,),
            target_feature_dim=(y_dim,),
            read_only=True
        )

        self.id_map = np.load(self.id_map_path)
        self.train_node_ids = torch.as_tensor(meta["train_node_ids"]).long()
        self.val_node_ids = torch.as_tensor(meta["val_node_ids"]).long()
        self.test_node_ids = torch.as_tensor(meta["test_node_ids"]).long()
        self.all_node_ids = torch.arange(int(meta["num_nodes"]), dtype=torch.long)
        if self.log:
            _logger.info(
                "Loaded ProteinDataset from %s (nodes=%d, x_dim=%d, y_dim=%d)",
                self.processed_dir,
                int(meta["num_nodes"]),
                int(meta["x_dim"]),
                int(meta["y_dim"]),
            )

    @staticmethod
    def _materialize(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)

    def download(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.features_batches_dir.mkdir(parents=True, exist_ok=True)

        # Materialize index + raw/
        self._materialize(
            self.dataset_input.path_to_accession_ids_csv_file,
            self.raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name
        )

        accessions = self.original_node_accessions
        if len(accessions) == 0:
            raise ValueError(
                "No valid UniRef accessions to query after removing UNK/UPI entries."
            )
        _logger.info(
            "Downloading feature shards for ProteinDataset: accessions=%d batches=%d",
            len(accessions),
            self.num_feature_batches,
        )
        batches = np.array_split(np.asarray(accessions, dtype=object), self.num_feature_batches)
        fields = [self.dataset_input.Y_query_field_name, *self.dataset_input.X_query_field_names]

        for i, batch in enumerate(batches, start=1):
            batch_path = self._feature_batch_path(i)
            if batch_path.exists():
                _logger.debug("Reusing existing feature shard %s", batch_path)
                continue

            batch_ids = [str(x) for x in batch.tolist()]
            if len(batch_ids) == 0:
                pd.DataFrame(columns=fields).to_csv(batch_path, index=False)
                _logger.debug("Wrote empty feature shard %s", batch_path)
                continue
            _logger.info(
                "Fetching feature shard %d/%d with %d accession(s)",
                i,
                self.num_feature_batches,
                len(batch_ids),
            )
            features_df = fetch_uniprotkb_fields(
                uniref_ids=batch_ids,
                fields=fields,
                request_size=self.dataset_input.request_size,
                rps=self.dataset_input.rps,
                max_retry=self.dataset_input.max_retry,
            )
            features_df.to_csv(batch_path, index=False)
            _logger.debug("Saved feature shard %s with %d row(s)", batch_path, int(features_df.shape[0]))

    @staticmethod
    def _to_tensor(value: Any, *, field_name: str, cast_float: bool = True) -> torch.Tensor:
        arr = _to_numeric_vector(value, field_name=field_name, cast_float=cast_float)
        return torch.from_numpy(arr)

    def process(self) -> None:
        # ------------------------- get the paths ------------------------
        accessions_path = self.raw_dir / self.dataset_input.path_to_accession_ids_csv_file.name
        _logger.info("Processing ProteinDataset into %s", self.processed_dir)
        if not accessions_path.exists():
            raise FileNotFoundError(f"Expected accession index at {accessions_path}")
        for i in range(1, self.num_feature_batches + 1):
            batch_path = self._feature_batch_path(i)
            if not batch_path.exists():
                raise FileNotFoundError(f"Expected raw feature batch at {batch_path}")
        # ----------------------------------------------------------------

        # If the dataset was already loaded in read mode, release the handle
        # before rebuilding the processed artifacts
        self.close()
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir) # ensures no stale state mixes with new processing
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)

        accessions_df = pd.read_csv(accessions_path)
        max_old_id = int(accessions_df["i"].max()) - 1
        id_map = np.full(max_old_id + 1, -1, dtype=np.int64)

        required_cols = [self.dataset_input.Y_return_field_name, *self.dataset_input.X_return_field_names]

        feature_store: _ProteinFeatureStore | None = None
        total_rows = 0
        try:
            # -------------------------- node features -----------------------
            for i in range(1, self.num_feature_batches + 1):
                batch_path = self._feature_batch_path(i)
                x_np, y_np = build_features_from_DatasetInput(
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter,
                    accessions_path=accessions_path,
                    features_path=batch_path,
                    required_cols=required_cols,
                    global_id_map=id_map,
                    X_return_field_names=self.dataset_input.X_return_field_names,
                    Y_return_field_name=self.dataset_input.Y_return_field_name,
                )

                if x_np.shape[0] == 0:
                    _logger.debug("Feature shard %s produced 0 kept rows; skipping", batch_path)
                    continue

                if feature_store is None:
                    feature_store = _ProteinFeatureStore(
                        store_on_disk_location=self.feature_store_dir,
                        node_feature_dim=(x_np.shape[1],),
                        edge_feature_dim=(0,),
                        target_feature_dim=(y_np.shape[1],),
                        read_only=False,
                    )
                else:
                    if x_np.shape[1] != int(feature_store.store.which_tensors["x"][0][1]):
                        raise ValueError("Inconsistent X feature dimensionality across feature batches")
                    if y_np.shape[1] != int(feature_store.store.which_tensors["y"][0][1]):
                        raise ValueError("Inconsistent Y dimensionality across feature batches")

                feature_store.append_tensor(x_np, group_name=None, attr_name="x", index=None)
                feature_store.append_tensor(y_np, group_name=None, attr_name="y", index=None)
                total_rows += int(x_np.shape[0])
                _logger.info(
                    "Processed feature shard %d/%d -> rows=%d x_dim=%d y_dim=%d (total_rows=%d)",
                    i,
                    self.num_feature_batches,
                    int(x_np.shape[0]),
                    int(x_np.shape[1]),
                    int(y_np.shape[1]) if y_np.ndim > 1 else 1,
                    total_rows,
                )
            # ----------------------------------------------------------------

            if feature_store is None or total_rows == 0:
                raise ValueError("All nodes were filtered out; cannot build FFNN dataset")
            _logger.info("Feature pass complete: total_nodes=%d", total_rows)

            # ---------------------------- splits ----------------------------
            y_full = feature_store.get_tensor(TensorAttr(None, "y", slice(None))).float()
            dummy_x = torch.zeros((total_rows, 1), dtype=torch.float32)
            split_seed = Data(x=dummy_x, y=y_full)
            val_set_size = floor(self.val_set_size * total_rows)
            test_set_size = floor(self.test_set_size * total_rows)
            split = RandomNodeSplit(num_val=val_set_size, num_test=test_set_size)(split_seed)
            train_node_ids = split.train_mask.nonzero(as_tuple=False).view(-1).long()
            val_node_ids = split.val_mask.nonzero(as_tuple=False).view(-1).long()
            test_node_ids = split.test_mask.nonzero(as_tuple=False).view(-1).long()
            _logger.debug(
                "Split sizes: train=%d val=%d test=%d",
                int(train_node_ids.numel()),
                int(val_node_ids.numel()),
                int(test_node_ids.numel()),
            )
            # ----------------------------------------------------------------

            # ---------------------- node metadata map -----------------------
            node_id_to_accession: dict[int, str] = {}
            for row in accessions_df.itertuples(index=False):
                old_id = int(row.i) - 1
                if old_id < 0 or old_id >= id_map.shape[0]:
                    continue
                new_id = int(id_map[old_id])
                if new_id < 0:
                    continue
                node_id_to_accession[new_id] = str(row.uniref).replace("UniRef90_", "", 1)

            if len(node_id_to_accession) != total_rows:
                raise RuntimeError(
                    "Node metadata mapping size mismatch after filtering/reindexing: "
                    f"{len(node_id_to_accession)} mapped nodes vs {total_rows} feature rows."
                )
            # ----------------------------------------------------------------

            np.save(self.id_map_path, id_map)

            meta = {
                "x_fields": [i for i in self.dataset_input.X_return_field_names if i != "Entry"],
                "y_field": self.dataset_input.Y_return_field_name,
                "num_nodes": int(total_rows),
                "x_dim": int(feature_store.store.which_tensors["x"][0][1]),
                "y_dim": int(feature_store.store.which_tensors["y"][0][1]),
                "train_node_ids": train_node_ids.cpu(),
                "val_node_ids": val_node_ids.cpu(),
                "test_node_ids": test_node_ids.cpu(),
                "node_id_to_accession": dict(sorted(node_id_to_accession.items())),
            }
            torch.save(meta, self.meta_path)

            if self.log:
                _logger.info(
                    "Processed FFNN dataset at %s (nodes=%d, x_dim=%d, y_dim=%d)",
                    self.processed_dir,
                    meta["num_nodes"],
                    meta["x_dim"],
                    meta["y_dim"]
                )
        finally:
            if feature_store is not None:
                feature_store.close()
    
    def _node_ids_for_split(self, split: Literal["train", "val", "test", "all"]) -> torch.Tensor:
        if split == "train":
            if self.train_node_ids is None:
                raise RuntimeError("ProteinDataset is not initialized")
            return self.train_node_ids
        if split == "val":
            if self.val_node_ids is None:
                raise RuntimeError("ProteinDataset is not initialized")
            return self.val_node_ids
        if split == "test":
            if self.test_node_ids is None:
                raise RuntimeError("ProteinDataset is not initialized")
            return self.test_node_ids
        if split == "all":
            if self.all_node_ids is None:
                raise RuntimeError("ProteinDataset is not initialized")
            return self.all_node_ids
        raise ValueError(f"Unknown split: {split}")

    def set_split(
        self,
        split: Literal["train", "val", "test", "all"],
        include_targets: bool | None = None
    ) -> ProteinDataset:
        self.active_node_ids = self._node_ids_for_split(split)
        self.split = split
        self.include_targets = bool(split == "train") if include_targets is None else bool(include_targets)
        _logger.debug(
            "Activated ProteinDataset split='%s' include_targets=%s size=%d",
            split,
            self.include_targets,
            int(self.active_node_ids.numel()),
        )
        return self

    def _getitem_by_node_id(self, node_id: int, *, include_targets: bool):
        if self.feature_store is None:
            raise RuntimeError("ProteinDataset is not initialized")

        x = self.feature_store.get_tensor(TensorAttr(None, "x", node_id)).float().view(-1)
        if self.transform is not None:
            x = self.transform(x)

        if not include_targets:
            return x

        y = self.feature_store.get_tensor(TensorAttr(None, "y", node_id)).float().view(-1)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self) -> int:
        if self.active_node_ids is None:
            raise RuntimeError("ProteinDataset split is not initialized")
        return int(self.active_node_ids.numel())

    def __getitem__(self, idx: int):
        if self.active_node_ids is None:
            raise RuntimeError("ProteinDataset split is not initialized")
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"Index out of range: {idx}")
        node_id = int(self.active_node_ids[idx].item())
        return self._getitem_by_node_id(node_id, include_targets=self.include_targets)

    def view(
        self,
        split: Literal["train", "val", "test", "all"],
        *,
        include_targets: bool | None = None,
    ) -> Dataset:
        node_ids = self._node_ids_for_split(split)
        resolved_include_targets = bool(split == "train") if include_targets is None else bool(include_targets)
        return _ProteinDatasetView(
            parent=self,
            node_ids=node_ids,
            include_targets=resolved_include_targets,
        )

    def loader(
        self,
        batch_size: int,
        shuffle: bool = False,
        **kwargs,
    ) -> pt_DataLoader:
        return pt_DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def train_loader(self, batch_size: int, shuffle: bool = True, **kwargs) -> pt_DataLoader:
        return pt_DataLoader(
            self.view("train", include_targets=True),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def val_loader(self, batch_size: int, shuffle: bool = False, **kwargs) -> pt_DataLoader:
        return pt_DataLoader(
            self.view("val", include_targets=True),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def test_loader(self, batch_size: int, shuffle: bool = False, **kwargs) -> pt_DataLoader:
        return pt_DataLoader(
            self.view("test", include_targets=True),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def predict_loader(
        self,
        batch_size: int,
        split: Literal["train", "val", "test", "all"] = "all",
        shuffle: bool = False,
        **kwargs,
    ) -> pt_DataLoader:
        return pt_DataLoader(
            self.view(split, include_targets=False),
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def close(self) -> None:
        if self.feature_store is not None:
            _logger.debug("Closing ProteinDataset feature store")
            self.feature_store.close()
            self.feature_store = None


__all__ = [

]


if __name__ == "__main__":
    pass
