# external:
import requests
import pandas as pd
from tqdm import tqdm
from numpy import nan
# builtins:
import os
import time
import io
from typing import List
import re
import logging
from datetime import datetime
from math import ceil
# pkg:
from .data_preparation_utils import preprocess_col
# env:
from dotenv import load_dotenv
load_dotenv()
LOGS_DIR = os.getenv("LOGS_DIR")

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, f"uniprot_parsing_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"),
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

def unirefs_from_tsv(path_, uniclust_to_uniref_tsv=None) -> list:

    unirefs = set()
    uniclusts = set()

    df = pd.read_csv(path_, sep='\t', skiprows=[0])

    print("Extracting UniRef90 and UniClust90 id(s) from ", os.path.basename(path_))
    for id in df["READS_UNMAPPED"]:
        uniref_match = re.search(r"UniRef90_([A-Z0-9]+)", id)
        if uniref_match:
            unirefs.add(uniref_match.group(1))
            continue
        uniclust_match = re.search(r"UniClust90_([0-9]+)", id)
        if uniclust_match:
            uniclusts.add(uniclust_match.group(1))
    print(f"Successfully extracted {len(unirefs)} UniRef90(s) and {len(uniclusts)} UniClust90(s)")
    unmatched_uniclusts_count = 0
    if uniclust_to_uniref_tsv is not None:
        print(f"Now attempting to map the {len(uniclusts)} UniClust90(s) to UniRef90 id(s)",
              f"using the {os.path.basename(uniclust_to_uniref_tsv)} mapping")
        map_df = pd.read_csv(uniclust_to_uniref_tsv, sep="\t", header=None, names=["uniclust_id", "uniref_id"])
        map_df["uniclust_id"] = map_df["uniclust_id"].astype(str)
        id_to_uniref = dict(zip(map_df["uniclust_id"], map_df["uniref_id"]))
        for id in tqdm(uniclusts, desc=f"Attempting to map {len(uniclusts)} UniClusts to UniRefs"):
            try:
                uniref = id_to_uniref[id]
            except KeyError:
                unmatched_uniclusts_count += 1
                continue
            unirefs.add(uniref)
    else:
        unmatched_uniclusts_count = len(uniclusts)
    
    print(f"{unmatched_uniclusts_count} UniClust90(s) was/were not mapped to UniRef90s and were dropped")

    return list(unirefs)

def retrieve_fields_for_unirefs(uniref_ids: List[str], fields: List[str] = recommended_fields_example2, batch_size: int = 100,
                  rps: float = 10, filter_out_bad_ids: bool = True, subroutine: bool = False) -> pd.DataFrame:

    if filter_out_bad_ids and not subroutine:
        p1 = r"^UNK"; p2 = r"^UPI"
        valid_ids = [id_ for id_ in uniref_ids if not (re.match(p1, id_) or re.match(p2, id_))]
        logging.info(f"Filtered out {len(uniref_ids) - len(valid_ids)} id(s) -- every one prefixed with either 'UNK' or 'UPI'")
        print(f"Filtered out {len(uniref_ids) - len(valid_ids)} corrupt id(s)")
        uniref_ids = valid_ids

    logging.info(
        f"Started retrieving {fields} for {len(uniref_ids)} IDs"
    )
    
    dfs: list[pd.DataFrame] = []
    total_ids = len(uniref_ids)
    
    total_batches = ceil(total_ids / batch_size)
    # process in batches
    for batch_idx, start in enumerate(range(0, total_ids, batch_size), start=1):
        
        end = start + batch_size
        batch = uniref_ids[start:end]

        if not subroutine:
            print(f"Processed {(batch_idx-1)}/{total_batches} batches; Querying {len(batch)} IDs…")

        params = {
            "format": "tsv",
            "size": len(batch),
            "query": " OR ".join(f"accession:{uid}" for uid in batch)
        }

        if fields:
            params["fields"] = ",".join(fields)
        
        logging.info(f"Query params: {params}")
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
        except requests.HTTPError as e:
            logging.warning(f"HTTP error on batch {batch_idx}: {e}")
            if batch_size <= 1:
                print(f"❌ Couldn't retrieve data for {batch} \nDropping and moving on")
                logging.warning(f"Dropping ID(s): {batch} and moving on")
                continue
            # split the batch and retry
            sub_df = retrieve_fields_for_unirefs(
                uniref_ids=batch,
                fields=fields,
                batch_size=batch_size // 2,
                rps=rps, filter_out_bad_ids=filter_out_bad_ids,
                subroutine=True)
            if not sub_df.empty:
                dfs.append(sub_df)
            continue
        except Exception as e:
            logging.error(f"Unexpected error on batch {batch_idx}: {e}")
            continue

        file_view = io.StringIO(resp.text)
        df = pd.read_csv(file_view, sep="\t", na_filter=False)
        if not df.empty:
            dfs.append(df)

        time.sleep(1.0 / rps)
    
    if not subroutine:
        print("Finished fetching the data")

    # stitch together or return an empty frame with correct columns
    if dfs:
        DF = pd.concat(dfs, ignore_index=True)
    else:
        DF = pd.DataFrame(columns=fields or [])
        DF.set_index("accession", inplace=True, drop=True)
        return DF
    
    DF.replace("", nan, inplace=True)
    DF.set_index("Entry", inplace=True, drop=True)
    DF.dropna(how="all", inplace=True)

    return DF

def process_entries(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy(deep=True)
    for col_name in new_df.columns:
        preprocess_col(new_df, col_name)
    return new_df

def tsv2df(tsv_path: str, uniclust_map_path: str, save: bool = True, save2dir: str = None):
    unirefs: List[str] = unirefs_from_tsv(tsv_path, uniclust_map_path)
    df: pd.DataFrame = process_entries(retrieve_fields_for_unirefs(unirefs, rps=15))
    if save:
        if save2dir is None:
            raise RuntimeError("if 'save' is True, 'save2dir' argument must be provided")
        save2path = os.path.join(save2dir, os.path.basename(tsv_path).replace("tsv", "csv"))
        df.to_csv(save2path)
    return df

def process_all_tsvs(tsv_files_dir: str, uniclust_map_path: str, save2dir: str = None):
    if save2dir is None:
        save2dir = os.getcwd()
    files = [os.path.join(tsv_files_dir, file) for file in os.listdir(tsv_files_dir)]
    for file in files:
        if file.endswith("_genefamilies.tsv"):
            tsv2df(file, uniclust_map_path, save=True)
            print("Successfully processed: ", os.path.basename(file))


__all__ = [
    "unirefs_from_tsv",
    "retrieve_fields_for_unirefs",
    "process_entries",
    "tsv2df",
    "process_all_tsvs"
]
