# third party
import zarr
from zarr.storage import LocalStore
from torch_geometric.data import TensorAttr
from zarr.core.array import Array
from zarr.core.group import Group
from numpy import ndarray
import numpy as np
import torch

# built in
import re
import os
import warnings
import functools
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Type

_logger = logging.getLogger(__name__)


def files_from(dir_path: str, pattern: re.Pattern = None):
    """
    Execute `files from`.

    Args:
        dir_path: Input value for `dir_path`.
        pattern: Input value for `pattern`.
    """
    pattern = pattern or re.compile(r".*")
    for file_name in sorted(os.listdir(dir_path)):
        if re.match(pattern, file_name):
            yield os.path.join(dir_path, file_name)

def compose(*funcs):
    """
    Execute `compose`.

    Args:
        funcs: Input value for `funcs`.
    """
    def inner(x: Any, **fun_args_map):
        """
        Execute `inner`.

        Args:
            x: Input value for `x`.
            fun_args_map: Input value for `fun_args_map`.
        """
        for f in funcs:
            if f in fun_args_map:
                args = fun_args_map[f]
            else:
                args = fun_args_map.get(f.__name__, ())
            if not isinstance(args, (tuple, list)):
                raise TypeError(
                    f"Arguments for function '{f.__name__}' must be a tuple/list, got {type(args)}"
                )
            x = f(x, *args)
        return x
    return inner

def suppress_warnings(*warning_types: Type[Warning]):
    """
    Execute `suppress warnings`.

    Args:
        warning_types: Input value for `warning_types`.
    """
    def decorator(func):
        """
        Execute `decorator`.

        Args:
            func: Input value for `func`.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            Execute `wrapper`.

            Args:
                args: Additional positional arguments.
                kwargs: Additional keyword arguments.
            """
            with warnings.catch_warnings():
                for w in warning_types:
                    warnings.simplefilter("ignore", w)
                return func(*args, **kwargs)
        return wrapper
    return decorator

def current_time() -> str:
    """Returns the current time in the Y-%m-%d_%H%M%S format"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


class ZarrFeatureStore:

    """
    Represent the `ZarrFeatureStore` type.
    """
    def __init__(self,
                store_on_disk_location: Path | str,
                read_only: bool = False):
        """
        Initialize a `ZarrFeatureStore` instance.

        Args:
            store_on_disk_location: Input value for `store_on_disk_location`.
            read_only: Input value for `read_only`.
        """
        self.read_only = read_only
        self.store_path = Path(store_on_disk_location)
        store_exists = self._zarr_is_store(store_on_disk_location)
        _logger.debug(
            "Initializing ZarrFeatureStore(path=%s, read_only=%s, existing_store=%s)",
            self.store_path,
            read_only,
            store_exists,
        )
        self.store, self.root = (
                self._zarr_load_store(store_on_disk_location, read_only=read_only) 
            if 
                store_exists
            else 
                self._zarr_create_store(store_on_disk_location, read_only=read_only)
        )

    @property
    def which_tensors(self) \
        -> dict[str,             # name
            tuple[
                tuple[int, ...], # shape
                str]]:           # dtype (zarr stores these as strings)
        """
        Execute `which tensors`.
        """
        out = dict()
        for name in self._zarr_group_subnode_names(self.root):
            if name.startswith("mask_of_"):
                continue
            node = self._zarr_node(named=name, from_group=self.root)
            if self._zarr_node_type(node) is not Array:
                continue
            out[name] = (self._zarr_arr_shape(node), self._zarr_arr_dtype(node))
        return out

    # -=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=- #
    #                                     ZARR HELPERS                                           #
    # -=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=- #

    @staticmethod
    def _zarr_is_store(store_on_disk_location: Path | str) -> bool:
        """
        Execute `zarr is store`.

        Args:
            store_on_disk_location: Input value for `store_on_disk_location`.
        """
        return Path(store_on_disk_location, "zarr.json").is_file()

    @staticmethod
    def _zarr_load_store(
        store_on_disk_location: Path | str, *,
        read_only: bool
    ) -> tuple[LocalStore, Group]:
        """
        Execute `zarr load store`.

        Args:
            store_on_disk_location: Input value for `store_on_disk_location`.
            read_only: Input value for `read_only`.
        """
        store = LocalStore(store_on_disk_location, read_only=read_only)
        mode = "r" if read_only else "a"
        root = zarr.open_group(store=store, mode=mode)
        return store, root

    @staticmethod
    def _zarr_create_store(
        store_on_disk_location: Path | str, *,
        read_only: bool
    ) -> tuple[LocalStore, Group]:
        """
        Execute `zarr create store`.

        Args:
            store_on_disk_location: Input value for `store_on_disk_location`.
            read_only: Input value for `read_only`.
        """
        if read_only:
            raise FileNotFoundError(
                f"There is no Zarr store at '{store_on_disk_location}' and "
                f"`read_only=True` forbids creating one."
            )
        os.makedirs(store_on_disk_location, exist_ok=True)
        store = LocalStore(store_on_disk_location, read_only=False)
        root = zarr.group(store, overwrite=True)
        return store, root
    
    @staticmethod
    def _zarr_node_type(node: Array | Group) -> Array | Group:
        """
        Execute `zarr node type`.

        Args:
            node: Input value for `node`.
        """
        node_metadata = node.metadata.to_dict()
        if node_metadata["node_type"] == "array":
            return Array
        elif node_metadata["node_type"] == "group":
            return Group
        else:
            raise RuntimeError(f"Unknown node type: {node_metadata['node_type']}")

    @staticmethod
    def _zarr_arr_shape(node: Array | Group) -> tuple[int, ...]:
        """
        Execute `zarr arr shape`.

        Args:
            node: Input value for `node`.
        """
        node_metadata = node.metadata.to_dict()
        return tuple(node_metadata["shape"])

    @staticmethod
    def _zarr_arr_dtype(node: Array | Group) -> str:
        """
        Execute `zarr arr dtype`.

        Args:
            node: Input value for `node`.
        """
        node_metadata = node.metadata.to_dict()
        return str(node_metadata["data_type"])

    @staticmethod
    def _zarr_group_subnode_names(G: Group) -> list[str]:
        """
        Execute `zarr group subnode names`.

        Args:
            G: Input value for `G`.
        """
        return list(G.keys())

    @staticmethod
    def _zarr_node(*, named: str, from_group: Group) -> Array | Group:
        """
        Execute `zarr node`.

        Args:
            named: Input value for `named`.
            from_group: Input value for `from_group`.
        """
        g = from_group
        try:
            return g[named]
        except KeyError:
            raise KeyError(f"There is no node named {named} in {g}") # <-- `g` might have an ugly repr 

    # -=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=--=-=-=-=-=-=-

    @staticmethod
    def _tensor_index(attr: TensorAttr) -> ndarray | slice | int:
        """
        Execute `tensor index`.

        Args:
            attr: Input value for `attr`.
        """
        index = attr.index if attr.index is not None else slice(None)

        if isinstance(index, torch.Tensor):
            return index.detach().cpu().numpy()
        if isinstance(index, (int, np.integer, slice, np.ndarray)):
            return index
        if isinstance(index, (list, tuple)):
            return np.asarray(index)

        raise TypeError(
            f"Unsupported index type '{type(index)}'. Expected int/slice/list/tuple/ndarray/Tensor."
        )

    @staticmethod
    def _tensor_name(attr: TensorAttr) -> str:
        """
        Execute `tensor name`.

        Args:
            attr: Input value for `attr`.
        """
        if attr.attr_name is None:
            raise RuntimeError("The passed tensor does not have a name")
        return attr.attr_name

    @staticmethod
    def _mask_named_after(name: str) -> str:
        """
        Execute `mask named after`.

        Args:
            name: Input value for `name`.
        """
        return f"mask_of_{name}"

    def _mask_of(self, name: str) -> Array:
        """
        Execute `mask of`.

        Args:
            name: Input value for `name`.
        """
        name = self._mask_named_after(name)
        try:
            node = self.root[name]
            if self._zarr_node_type(node) is not Array:
                raise KeyError
        except KeyError:
            raise KeyError(f"No array named {name}")
        return node

    def _data_of(self, name: str) -> Array:
        """
        Execute `data of`.

        Args:
            name: Input value for `name`.
        """
        try:
            node = self.root[name]
            if self._zarr_node_type(node) is not Array:
                raise KeyError
        except KeyError:
            raise KeyError(f"No array named {name}")
        return node

    def add_location(
        self,
        attr: TensorAttr,
        shape: tuple[int, ...],
        *,
        dtype: str | np.dtype = "float32",
        overwrite: bool = False
    ) -> None:
        """
        Execute `add location`.

        Args:
            attr: Input value for `attr`.
            shape: Input value for `shape`.
            dtype: Input value for `dtype`.
            overwrite: Input value for `overwrite`.
        """
        if len(shape) == 0:
            raise ValueError("`shape` must have at least one dimension")

        name = self._tensor_name(attr)
        mask_name = self._mask_named_after(name)

        if overwrite:
            for node_name in (name, mask_name):
                if node_name in self.root:
                    del self.root[node_name]
        else:
            if name in self.root or mask_name in self.root:
                raise ValueError(
                    f"Location '{name}' already exists. Pass overwrite=True to recreate it."
                )

        self.root.create_array(name, shape=shape, dtype=dtype)
        # Mask tracks logical liveness along the first axis only
        self.root.create_array(mask_name, data=np.zeros(shape[0], dtype=np.bool_))
        _logger.debug(
            "Created tensor location '%s' with shape=%s dtype=%s (overwrite=%s)",
            name,
            shape,
            str(dtype),
            overwrite,
        )

    def add_data_to_location(self, data: ndarray, attr: TensorAttr):
        """
        Execute `add data to location`.

        Args:
            data: Input value for `data`.
            attr: Input value for `attr`.
        """
        name = self._tensor_name(attr)
        index = self._tensor_index(attr) # indexing is expected along the leading axis only
        mask, arr = self._mask_of(name), self._data_of(name)
        data = np.asarray(data, dtype=arr.dtype)
        try:
            arr[index] = data
            mask[index] = True
        except Exception as e:
            raise ValueError(
                f"Could not write data to '{name}' at index {index}: {e}"
            ) from e
        _logger.debug(
            "Wrote data to '%s' at index=%s with data_shape=%s",
            name,
            index,
            data.shape,
        )
    
    def read_data_from_location(self, attr: TensorAttr) -> ndarray:
        """
        Execute `read data from location`.

        Args:
            attr: Input value for `attr`.
        """
        name = self._tensor_name(attr)
        index = self._tensor_index(attr) # indexing is expected along the leading axis only
        mask, arr = self._mask_of(name), self._data_of(name)
        alive = np.asarray(mask[index])
        if alive.size > 0 and not np.all(alive):
            raise IndexError(
                f"Requested entries in '{name}' include rows that are logically deleted or not written."
            )
        return np.asarray(arr[index])

    def remove_data_from_location(self, attr: TensorAttr):
        """
        Execute `remove data from location`.

        Args:
            attr: Input value for `attr`.
        """
        name = self._tensor_name(attr)
        index = self._tensor_index(attr) # indexing is expected along the leading axis only
        mask = self._mask_of(name)
        mask[index] = False
        _logger.debug("Marked entries as deleted in '%s' at index=%s", name, index)

    def clear_location(self, attr: TensorAttr):
        """
        Execute `clear location`.

        Args:
            attr: Input value for `attr`.
        """
        name = self._tensor_name(attr)
        mask = self._mask_of(name)
        mask[...] = False
        _logger.debug("Cleared liveness mask for '%s'", name)

    def append(self, data: ndarray, attr: TensorAttr) -> slice:
        """
        Execute `append`.

        Args:
            data: Input value for `data`.
            attr: Input value for `attr`.
        """
        name = self._tensor_name(attr)
        arr = self._data_of(name)
        mask = self._mask_of(name)

        data = np.asarray(data, dtype=arr.dtype)
        if data.ndim == arr.ndim - 1:
            # Single row append.
            data = data[None, ...]
        elif data.ndim != arr.ndim:
            raise ValueError(
                f"`data` rank mismatch for '{name}': expected {arr.ndim - 1} or "
                f"{arr.ndim}, got {data.ndim}"
            )

        if data.shape[1:] != arr.shape[1:]:
            raise ValueError(
                f"`data` trailing shape mismatch for '{name}': expected {arr.shape[1:]}, got {data.shape[1:]}"
            )

        old_n = int(arr.shape[0])
        batch_n = int(data.shape[0])
        new_n = old_n + batch_n

        arr.resize((new_n, *arr.shape[1:]))
        mask.resize((new_n,))

        arr[old_n:new_n] = data
        mask[old_n:new_n] = True

        out = slice(old_n, new_n) # the newly appended block
        _logger.debug(
            "Appended %d row(s) to '%s' (old_n=%d, new_n=%d)",
            batch_n,
            name,
            old_n,
            new_n,
        )
        return out

    def drop_location(self, attr: TensorAttr) -> bool:
        """
        Execute `drop location`.

        Args:
            attr: Input value for `attr`.
        """
        name = self._tensor_name(attr)
        mask_name = self._mask_named_after(name)
        deleted_any = False
        for node_name in (name, mask_name):
            if node_name in self.root:
                del self.root[node_name]
                deleted_any = True
        _logger.debug("Dropped tensor location '%s' (deleted_any=%s)", name, deleted_any)
        return deleted_any

    def close(self):
        """
        Close the current object.
        """
        _logger.debug("Closing ZarrFeatureStore at %s", self.store_path)
        self.store.close()
    
    def __enter__(self):
        """
        Enter the context manager.
        """
        return self
    
    def __exit__(self, exc_type, exc, tb):
        """
        Exit the context manager.

        Args:
            exc_type: Input value for `exc_type`.
            exc: Input value for `exc`.
            tb: Input value for `tb`.
        """
        self.close()


if __name__ == "__main__":
    pass
