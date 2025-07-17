import requests
import time
import io
import pandas as pd
from typing import List
import re
import logging
from datetime import datetime
from math import ceil

logging.basicConfig(
    filename=f"uniprot_parsing_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

recommended_fields = [
    "accession", "ft_domain", "cc_domain", "protein_families", "go_f", "go_p",
    "cc_interaction", "cc_function", "cc_catalytic_activity",
    "ec", "cc_pathway", "rhea", "cc_cofactor", "cc_activity_regulation"
]

# I have empirically determined that requests for accession IDs that start with 'UNK' and 'UPI' are never found
# in the UniProt database and lead to HTTP errors (Bad request error), so what we will do is filter them out
def parse_unirefs(uniref_ids: List[str], fields: List[str] = recommended_fields, batch_size: int = 100,
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
            sub_df = parse_unirefs(
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
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=fields or [])
