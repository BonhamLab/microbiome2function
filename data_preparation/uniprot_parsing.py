from tqdm import tqdm
from typing import List
import requests
import time
import pandas as pd
from util import flatten, FIELDS
import json


def data_from_Uniref90ID(ID: str, *fields) -> dict:
    url = "https://rest.uniprot.org/uniprotkb/search?query={Uniref90_ID}"

    if fields:
        url = url.format(Uniref90_ID=ID) + "&fields=" + ",".join(fields)
    else:
        url = url.format(Uniref90_ID=ID)

    response = requests.get(url).text

    try:
        data = json.loads(response)["results"]
    except KeyError:
        print(f"Invalid query")
        data = {}

    return flatten(data)


def uniprot_resp2pandas():
    pass
