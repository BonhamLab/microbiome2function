from tqdm import tqdm
import requests
import time
import pandas as pd
from util import flatten

FIELDS = [
    "ft_domain", "cc_domain", "protein_families", "go_f", "go_p",
    "cc_interaction", "cc_function", "cc_catalytic_activity",
    "ec", "cc_pathway", "rhea", "cc_cofactor", "cc_activity_regulation"
]

def data_from_Uniref90ID(uniref_id: str, *fields) -> dict:
    base = "https://rest.uniprot.org/uniprotkb/search?query={ID}"
    url = base.format(ID=uniref_id)
    if fields:
        url += "&fields=" + ",".join(fields)

    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"UniProt API error {resp.status_code}: {resp.text}")

    j = resp.json()
    return flatten(j.get("results", []))

def parse_unirefs(uniref_ids: list, fields: list = FIELDS) -> pd.DataFrame:
    records = []
    for uniref_id in tqdm(uniref_ids, desc="Fetching UniProt data"):
        time.sleep(1)
        print("Fetching ", uniref_id)
        try:
            data = data_from_Uniref90ID(uniref_id, *fields)
        except Exception as e:
            print(f"❌ failed for {uniref_id}: {e}")
            continue
        records.append(data)

    if not records:
        return pd.DataFrame()  # empty

    print("Finished parsing")

    return pd.DataFrame(records)
