# builtins:
import os
import time
import io
from typing import List, Optional, Iterable, Tuple
import re
import logging
from math import ceil

# third-party:
import requests
import pandas as pd

# local:
from . import util

# *-----------------------------------------------*
#                      GLOBALS
# *-----------------------------------------------*

_logger = logging.getLogger(__name__)
_UNIREF90_RE   = re.compile(r"UniRef90_([A-Z0-9]+)")
_UNICLUST90_RE = re.compile(r"UniClust90_([0-9]+)")

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def extract_accessions_from_humann(
    file_path: str, 
    out_type: type = list
) -> Tuple[Iterable, Iterable]:
    """Extract UniRef90 and UniClust90 accessions from a HUMAnN gene-families file.

    Parses the `READS_UNMAPPED` column, filters out accessions starting with
    'UNK' or 'UPI' (as UniProtkb queries fail on them),
    and returns two collections of 'out_type': (unirefs, uniclusts).
    """
    unirefs, uniclusts = set(), set()
    df = pd.read_csv(file_path, sep="\t", skiprows=[0])
    
    if "READS_UNMAPPED" not in df.columns:
        raise KeyError(
            f"Column 'READS_UNMAPPED' not found in {os.path.basename(file_path)}; "
            "check HUMAnN output format or parsing options."
        )

    _logger.info(f"Extracting UniRef90 and UniClust90 id(s) from {os.path.basename(file_path)}")
    for id_ in df["READS_UNMAPPED"]:
        uniref_match = _UNIREF90_RE.search(id_)

        if uniref_match:
            uniref = uniref_match.group(1)
            if not uniref.startswith(("UNK", "UPI")):
                unirefs.add(uniref)
            continue

        uniclust_match = _UNICLUST90_RE.search(id_)
        if uniclust_match:
            uniclusts.add(uniclust_match.group(1))

    _logger.info(f"Successfully extracted {len(unirefs)} UniRef90(s) and {len(uniclusts)} UniClust90(s)")
    return out_type(unirefs), out_type(uniclusts)


def extract_all_accessions_from_dir(
    dir_path: str, 
    pattern: Optional[re.Pattern] = None,
    out_type: type = list
) -> Tuple[Iterable, Iterable]:
    """Aggregate UniRef90 and UniClust90 accessions from all files in a directory.

    Iterates files from dir_path, extracts accessions
    per file if file name matches the 'pattern', unions them, and returns (unirefs, uniclusts) of 'out_type'.
    """
    all_unirefs, all_uniclusts = set(), set()
    for file in util.files_from(dir_path, pattern):
        unirefs, uniclusts = extract_accessions_from_humann(file, out_type=set)
        all_unirefs.update(unirefs)
        all_uniclusts.update(uniclusts)
    return out_type(all_unirefs), out_type(all_uniclusts)


def fetch_uniprotkb_fields(
    uniref_ids: List[str],
    fields: List[str],
    request_size: int = 100,
    rps: float = 10,
    max_retry: Optional[int | float] = float("inf"),
    subroutine_call_count: int = 0
) -> pd.DataFrame:
    """Fetch selected UniProtKB fields for a list of accessions with batched requests.

    Sends batched queries to the UniProtKB REST API, rate-limited to `rps`,
    halves the batch size and retries on HTTP errors up to `max_retry`, and
    concatenates results into a single DataFrame (or empty with given columns).
    """
    if request_size < 1:
        raise ValueError("request_size must be ≥ 1")

    if subroutine_call_count == 0:
        _logger.info(f"Started retrieving {fields} for {len(uniref_ids)} ID(s)")

    dfs: list[pd.DataFrame] = []
    total_ids       = len(uniref_ids)
    total_requests  = ceil(total_ids / request_size)

    # ---------Batched-data-retrieval---------
    for request_id, start in enumerate(range(0, total_ids, request_size), start=1):
        end   = start + request_size
        batch = uniref_ids[start:end]

        if subroutine_call_count == 0:
            _logger.info(f"Processed {(request_id - 1)}/{total_requests} requests; Querying {len(batch)} ID(s)…")

        params = {
            "format": "tsv",
            "size":   len(batch),
            "query":  " OR ".join(f"accession:{uid}" for uid in batch),
            "fields": ",".join(fields),
        }

        start_time = time.perf_counter()
        try:
            resp = requests.get("https://rest.uniprot.org/uniprotkb/search",
                                params=params, timeout=30)
            resp.raise_for_status()

        except requests.HTTPError as e:
            _logger.warning(f"HTTP error on request number {request_id}: {e}")
            if request_size <= 1 or subroutine_call_count >= max_retry:
                _logger.warning(f"Couldn't retrieve data for {batch}\nDropping and moving on")
            else:
                new_size = max(1, request_size // 2)
                sub_df   = fetch_uniprotkb_fields(
                    uniref_ids            = batch,
                    fields                = fields,
                    request_size          = new_size,
                    rps                   = rps,
                    max_retry             = max_retry,
                    subroutine_call_count = subroutine_call_count + 1,
                )
                if not sub_df.empty:
                    dfs.append(sub_df)
            continue

        except Exception as e:
            _logger.error(f"Unexpected error on batch {request_id}: {e}")
            continue

        # ---------to-df-conversion-and-minor-cleaning---------
        file_view = io.StringIO(resp.text)
        df = pd.read_csv(file_view, sep="\t", na_values=[""], keep_default_na=True)
        df.dropna(how="all", inplace=True)
        if not df.empty:
            dfs.append(df)

        elapsed = time.perf_counter() - start_time
        time.sleep(max(0, 1.0 / rps - elapsed))

    # ---------after-data-retrieval---------
    if subroutine_call_count == 0:
        _logger.info(f"Processed {request_id}/{total_requests} requests.")
        _logger.info("Finished fetching the data")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=fields)


def fetch_save_uniprotkb_batches(
    uniref_ids: List[str],
    fields: List[str],
    batch_size: int,
    single_api_request_size: int = 100,
    rps: float = 10,
    save_to_dir: Optional[str] = None
) -> str:
    """Fetch UniProtKB data in large batches and save each batch to disk.

    Splits `uniref_ids` into `batch_size` groups; for each group, calls
    `fetch_uniprotkb_fields` (with `single_api_request_size` per API call),
    then saves to Parquet (falling back to CSV). Returns the output directory.
    """
    if save_to_dir is None:
        save_to_dir = os.getcwd()
    os.makedirs(save_to_dir, exist_ok=True)

    total_ids           = len(uniref_ids)
    batches_to_process  = ceil(total_ids / batch_size)
    processed_batches   = 0

    for request_id, start in enumerate(range(0, total_ids, batch_size), start=1):
        end   = start + batch_size
        batch = uniref_ids[start:end]

        _logger.info(f"Submitting {request_id}/{batches_to_process} batch of API requests with {len(batch)} entry(s)")

        data = fetch_uniprotkb_fields(batch, fields,
                                      request_size=single_api_request_size,
                                      rps=rps)

        _logger.info(f"Received {len(data)} non-empty rows of data")

        ts   = int(time.time())
        file = os.path.join(save_to_dir, f"batch_{request_id}_{ts}.parquet")

        try:
            data.to_parquet(path=file, index=True)
        except Exception as e:
            _logger.warning(f"to_parquet failed ({e}); falling back to CSV")
            file = os.path.join(save_to_dir, f"batch_{request_id}_{ts}.csv")
            data.to_csv(file, index=True)

        _logger.info(f"The data were saved at {file}")
        del data  # explicitly free the RAM

        processed_batches += 1

    if processed_batches < batches_to_process:
        _logger.error(f"Did not manage to process all the batches of UniProt requests: only {processed_batches}/{batches_to_process} were processed")

    return save_to_dir


__all__ = [
    "extract_accessions_from_humann",
    "extract_all_accessions_from_dir",
    "fetch_uniprotkb_fields",
    "fetch_save_uniprotkb_batches"
]

if __name__ == "__main__":
    pass
