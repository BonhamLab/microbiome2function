# external:
import requests
import pandas as pd
from numpy import nan

# builtins:
import os
import time
import io
from typing import List
import re
import logging
from math import ceil

# *-----------------------------------------------*
#                      GLOBALS
# *-----------------------------------------------*

_logger = logging.getLogger(__name__)

recommended_fields_example1 = [
    "accession", "ft_domain", "cc_domain", "protein_families", "go_f", "go_p",
    "cc_interaction", "cc_function", "cc_catalytic_activity",
    "ec", "cc_pathway", "rhea", "cc_cofactor", "cc_activity_regulation"
]

recommended_fields_example2 = [
    "accession", "ft_domain", "cc_domain", "protein_families", "go_f", "go_p",
    "cc_function", "cc_catalytic_activity",
    "ec", "cc_pathway", "rhea", "cc_cofactor", "sequence"
]

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def ids_from_tsv(path_: str, out: type = list) -> list:

    unirefs = set()
    uniclusts = set()

    df = pd.read_csv(path_, sep='\t', skiprows=[0])

    _logger.info(f"Extracting UniRef90 and UniClust90 id(s) from {os.path.basename(path_)}")
    for id in df["READS_UNMAPPED"]:
        uniref_match = re.search(r"UniRef90_([A-Z0-9]+)", id)
        if uniref_match:
            unirefs.add(uniref_match.group(1))
            continue
        uniclust_match = re.search(r"UniClust90_([0-9]+)", id)
        if uniclust_match:
            uniclusts.add(uniclust_match.group(1))
    _logger.info(f"Successfully extracted {len(unirefs)} UniRef90(s) and {len(uniclusts)} UniClust90(s)")

    return out(unirefs), out(uniclusts)

def unirefs_from_multiple_files(dir_path: str):

    file_paths = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]
    all_unirefs = set()
    for path_ in file_paths:
        unirefs, _ = ids_from_tsv(path_, out=set)
        all_unirefs.update(unirefs)
    
    return list(all_unirefs)

def retrieve_fields_for_unirefs(uniref_ids: List[str], fields: List[str], request_size: int = 100,
                  rps: float = 10, filter_out_bad_ids: bool = True, subroutine: bool = False) -> pd.DataFrame:

    if filter_out_bad_ids and not subroutine:
        p1 = r"^UNK"; p2 = r"^UPI"
        valid_ids = [id_ for id_ in uniref_ids if not (re.match(p1, id_) or re.match(p2, id_))]
        _logger.info(f"Filtered out {len(uniref_ids) - len(valid_ids)} id(s) -- every one prefixed with either 'UNK' or 'UPI'")
        _logger.info(f"Filtered out {len(uniref_ids) - len(valid_ids)} corrupt id(s)")
        uniref_ids = valid_ids

    _logger.info(
        f"Started retrieving {fields} for {len(uniref_ids)} IDs"
    )
    
    dfs: list[pd.DataFrame] = []
    total_ids = len(uniref_ids)
    
    total_requests= ceil(total_ids / request_size)
    # process in batches
    for request_id, start in enumerate(range(0, total_ids, request_size), start=1):
        
        end = start + request_size
        batch = uniref_ids[start:end]

        if not subroutine:
            _logger.info(f"Processed {(request_id-1)}/{total_requests} requests; Querying {len(batch)} ID(s)…")

        params = {
            "format": "tsv",
            "size": len(batch),
            "query": " OR ".join(f"accession:{uid}" for uid in batch)
        }

        if fields:
            params["fields"] = ",".join(fields)
        
        _logger.info(f"Query params: {params}")
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            _logger.warning(f"HTTP error on request number {request_id}: {e}")
            if request_size <= 1:
                _logger.warning(f"❌ Couldn't retrieve data for {batch} \nDropping and moving on")
                _logger.warning(f"Dropping ID(s): {batch} and moving on")
                continue
            # split the batch and retry
            sub_df = retrieve_fields_for_unirefs(
                uniref_ids=batch,
                fields=fields,
                request_size=request_size // 2,
                rps=rps, filter_out_bad_ids=filter_out_bad_ids,
                subroutine=True)
            if not sub_df.empty:
                dfs.append(sub_df)
            continue
        except Exception as e:
            _logger.error(f"Unexpected error on batch {request_id}: {e}")
            continue

        file_view = io.StringIO(resp.text)
        df = pd.read_csv(file_view, sep="\t", na_filter=False)
        if not df.empty:
            dfs.append(df)

        time.sleep(1.0 / rps)
    
    if not subroutine:
        _logger.info(f"Processed {(request_id)}/{total_requests} requests.")
        _logger.info("Finished fetching the data")

    # stitch together or return an empty frame with correct columns
    if dfs:
        DF = pd.concat(dfs, ignore_index=True)
    else:
        DF = pd.DataFrame(columns=fields or [])
        DF.set_index("accession", inplace=True, drop=True)
        return DF
    
    DF = DF.replace("", nan)
    DF.set_index("Entry", inplace=True, drop=True)
    
    original_DF_length = len(DF)

    subset = list(DF.columns)
    subset.remove("Sequence")
    DF.dropna(how="all", subset=subset, inplace=True)

    rows_dropped = original_DF_length - len(DF)
    if rows_dropped > 0:
        _logger.info(f"Dropped {rows_dropped} UniRef90s that did not have the requested data associated with them"
                    if rows_dropped >1 else f"Dropped a UniRef90 that did not have the requested data associated with it")

    return DF

def process_uniref_batches(uniref_ids: List[str], fields: List[str], batch_size=40_000, 
                single_api_request_size: int = 100, rps: float = 10, save_to_dir: str = None,
                filter_out_bad_ids: bool = True):

    total_ids = len(uniref_ids)
    batches_to_process= ceil(total_ids / batch_size)

    processed_batches = 0
    # process in batches
    for request_id, start in enumerate(range(0, total_ids, batch_size), start=1):
        end = start + batch_size
        batch = uniref_ids[start:end]

        _logger.info(f"Submitting {request_id}/{batches_to_process} batch of API requests with {len(batch)} entries")

        data = retrieve_fields_for_unirefs(uniref_ids=batch,
                                    fields=fields,
                                    request_size=single_api_request_size,
                                    rps=rps,
                                    filter_out_bad_ids=filter_out_bad_ids)
        
        _logger.info(f"Received {len(data)} non-empty rows of data")

        if save_to_dir is None:
             _logger.warning(f"save_to_dir is set to None! The data will discarded")
             file = None
        else:
            file = os.path.join(save_to_dir, f'batch_{request_id}.csv')
            _logger.info(f"The data will be saved at {file}")

        data.to_csv(path_or_buf=file, index=True)
        processed_batches += 1
    
    if processed_batches < batches_to_process:
        _logger.error(f"Did not manage to process all the batches of UniProt requests: only {processed_batches}/{batches_to_process} were processed")
        
    return save_to_dir
        
def merge_dfs(dir_path: str) -> pd.DataFrame:
    file_paths = sorted([os.path.join(dir_path, file_name) 
                        for file_name in os.listdir(dir_path)
                        if file_name.endswith(".csv")])

    if not file_paths:
        raise FileNotFoundError(f"No files found in {dir_path}")

    dfs = [pd.read_csv(fp) for fp in file_paths]
    merged = pd.concat(dfs, ignore_index=True)
    merged.set_index("Entry", inplace=True, drop=True)

    return merged


__all__ = [
    "unirefs_from_multiple_files",
    "recommended_fields_example1",
    "recommended_fields_example2",
    "ids_from_tsv",
    "retrieve_fields_for_unirefs",
    "process_uniref_batches",
    "merge_dfs"
]

if __name__ == "__main__":
    pass
