# third-party
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, OnDiskDataset, Data

# built-in
from dataclasses import dataclass, field
from typing import Iterator, Any
from pathlib import Path
import re
import shutil

# local
from . import util
from .mining_utils import fetch_uniprotkb_fields


@dataclass
class DatasetInput:
    """
    Input contract consumed by PyG dataset interfaces (InMemory / OnDisk).

    Expected raw format:
    - accession index CSV: columns ['uniref', 'i'] (1-based node ids)
    - edge chunk CSVs: file names like chunk_<i>.csv, columns ['j', 'v']
    - uniprot_features: list/tuple of UniProt return field names (e.g. 'sequence', 'go_f')
    """

    path_to_accession_ids_csv_file: Path
    path_to_edge_csv_dir: Path
    uniprot_features: list[str] | tuple[str, ...]
    request_size: int = 25
    rps: float = 1
    max_retry: int | float = 20
    edge_csv_file_name_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"chunk_\d+\.csv")
    )

    _validation_ctx: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _accession_ids_df: pd.DataFrame | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path_to_accession_ids_csv_file = Path(self.path_to_accession_ids_csv_file)
        self.path_to_edge_csv_dir = Path(self.path_to_edge_csv_dir)
        self._normalize_uniprot_features()
        self.validate()

    def validate(self) -> None:
        self._validate_uniprot_request_params()
        self._validate_uniprot_features()
        self._validate_accession_ids_csv_file()
        self._validate_edge_csv_files()

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

    def _normalize_uniprot_features(self) -> None:
        if not isinstance(self.uniprot_features, (list, tuple)):
            raise TypeError(
                "`uniprot_features` must be list[str] or tuple[str, ...]"
            )

        normalized: list[str] = []
        for feature in self.uniprot_features:
            if not isinstance(feature, str):
                raise TypeError(
                    f"All entries in `uniprot_features` must be strings, got {type(feature)}"
                )
            cleaned = feature.strip()
            if cleaned:
                normalized.append(cleaned)

        # deduplicate while preserving order
        normalized = list(dict.fromkeys(normalized))

        # ensure accession is always requested for stable joins/alignment
        if "accession" not in normalized:
            normalized.insert(0, "accession")

        self.uniprot_features = tuple(normalized)

    def _validate_uniprot_features(self) -> None:
        if len(self.uniprot_features) == 0:
            raise ValueError("`uniprot_features` cannot be empty")

        if any(not feature for feature in self.uniprot_features):
            raise ValueError("`uniprot_features` cannot contain empty strings")

        if "accession" not in self.uniprot_features:
            raise ValueError("`uniprot_features` must include 'accession'")

        self._validation_ctx["uniprot_features"] = self.uniprot_features
        self._validation_ctx["num_uniprot_features"] = len(self.uniprot_features)

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
            expected_cols = ["j", "v"]
            if df.columns.tolist() != expected_cols:
                raise ValueError(f"Expected edge CSV columns {expected_cols} in {path}")

            if not df.empty and not pd.api.types.is_integer_dtype(df["j"]):
                raise ValueError(f"Column 'j' must be integer dtype in {path}")

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
    def uniprot_query_fields(self) -> tuple[str, ...]:
        return tuple(self.uniprot_features)


class ProteinGraphInMemoryDataset(InMemoryDataset):
    """
    Skeleton for a single-graph PyG InMemoryDataset.

    TODO: Fill in `download()` and `process()` with project-specific logic.
    """

    def __init__(
        self,
        root: str | Path,
        dataset_input: DatasetInput,
        node_transform=None,
        node_pre_transform=None,
        node_pre_filter=None,
        log: bool = True,
        force_reload: bool = False
    ) -> None:
        self.dataset_input = dataset_input
        self.dataset_input.validate()
        super().__init__(
            root=str(root),
            log=log,
            force_reload=force_reload,
        )

        processed_path = Path(self.processed_paths[0])
        if processed_path.exists():
            self.data, self.slices = torch.load(processed_path, weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [
            self.dataset_input.path_to_accession_ids_csv_file.name,
            *[path.name for path in self.dataset_input.edge_csv_paths],
        ]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        """
        TODO:
        - Optionally copy/symlink raw files into `self.raw_dir`.
        - Optionally trigger remote download when files are missing.
        """
        pass

    def process(self) -> None:
        """
        TODO:
        - Parse accession index and chunk edge files.
        - Build Data(x, edge_index, edge_attr, y, ...).
        - Apply pre_filter/pre_transform as needed.
        - Save via `torch.save(self.collate([data]), self.processed_paths[0])`.
        """
        raise NotImplementedError("Implement `process()` in ProteinGraphInMemoryDataset")


class ProteinGraphOnDiskDataset(OnDiskDataset):
    """
    Skeleton for a single-graph PyG OnDiskDataset.

    TODO: Fill in `download()` and `process()` with project-specific logic.
    """

    def __init__(
        self,
        root: str | Path,
        dataset_input: DatasetInput,
        transform=None,
        pre_filter=None,
        backend: str = "sqlite",
        schema: object = object,
        log: bool = True
    ) -> None:
        self.dataset_input = dataset_input
        self.dataset_input.validate()
        super().__init__(
            root=str(root),
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
            schema=schema,
            log=log,
        )

    @property
    def raw_file_names(self) -> list[str]:
        return [
            self.dataset_input.path_to_accession_ids_csv_file.name,
            *[path.name for path in self.dataset_input.edge_csv_paths],
        ]

    def download(self) -> None:
        """
        TODO:
        - Optionally copy/symlink raw files into `self.raw_dir`.
        - Optionally trigger remote download when files are missing.
        """
        pass

    def process(self) -> None:
        """
        TODO:
        - Parse raw graph.
        - Convert each stored object to Data (or schema-compatible object).
        - Write with `self.append(data)` / `self.extend(data_list)`.
        """
        raise NotImplementedError("Implement `process()` in ProteinGraphOnDiskDataset")
