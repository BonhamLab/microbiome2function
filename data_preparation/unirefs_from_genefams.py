import pandas as pd
import re

def unirefs_from_tsv(path_, uniclust_to_uniref_tsv=None) -> list:

    unirefs = set()
    uniclusts = set()

    df = pd.read_csv(path_, sep='\t', skiprows=[0])

    for id in df["READS_UNMAPPED"]:
        uniref_match = re.search(r"UniRef90_([A-Z0-9]+)", id)
        if uniref_match:
            unirefs.add(uniref_match.group(1))
            continue
        uniclust_match = re.search(r"UniClust90_([0-9]+)", id)
        if uniclust_match:
            uniclusts.add(uniclust_match.group(1))
    
    unmatched_uniclusts_count = 0
    if uniclust_to_uniref_tsv is not None:
        map_df = pd.read_csv(uniclust_to_uniref_tsv, sep="\t", header=None, names=["uniclust_id", "uniref_id"])
        map_df["uniclust_id"] = map_df["uniclust_id"].astype(str)
        id_to_uniref = dict(zip(map_df["uniclust_id"], map_df["uniref_id"]))
        for id in uniclusts:
            try:
                uniref = id_to_uniref[id]
            except KeyError:
                unmatched_uniclusts_count += 1
                continue
            unirefs.add(uniref)
    else:
        unmatched_uniclusts_count = len(uniclusts)
    
    print(f"{unmatched_uniclusts_count} UniClust90(s) was/were not matched")

    return list(unirefs)
