import requests
import pandas as pd
import json

"""
 FEATURE NAMES:
 https://www.uniprot.org/help/return_fields
"""

def features_from_Uniref90ID(ID: str, *fields) -> pd.DataFrame:
    url = "https://rest.uniprot.org/uniprotkb/search?query={Uniref90_ID}"
    url_with_fields = url
    if fields:
        url_with_fields += "&fields=" + ",".join(fields)
    response = requests.get(url_with_fields.format(Uniref90_ID=ID)).text
    data = json.loads(response)["results"]
    df = pd.DataFrame(data)
    
    return df
