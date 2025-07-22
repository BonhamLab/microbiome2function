import pandas as pd
import re

_identity_regex = re.compile(r'^(.*)$')

_info_extr_patterns = {
    "Domain [FT]" : re.compile(r'/note="([^"]+)"'),
    "Domain [CC]" : re.compile(r"DOMAIN:\s(.*?)(?=\s\{|$)"),
    "Protein families": _identity_regex, # no pre-processing is needed just yet
    "Gene Ontology (molecular function)" : _identity_regex, # no pre-processing is needed just yet
    "Gene Ontology (biological process)" : _identity_regex, # no pre-processing is needed just yet
    "Interacts with" : _identity_regex, # no pre-processing is needed just yet
    "Function [CC]" : re.compile(r"FUNCTION:\s(.*?)(?=\{|$)"),
    "Catalytic activity" : re.compile(r"Reaction=(.*?)(?=;|\.|$)"),
    "EC number" : _identity_regex, # no pre-processing is needed just yet
    "Pathway" : re.compile(r"PATHWAY:\s(.*?)(?=\{|$)"),
    "Rhea ID" : _identity_regex, # no pre-processing is needed just yet
    "Cofactor" : re.compile(r"COFACTOR:\s(.*?)(?=\{|$)"),
    "Activity regulation" : re.compile(r"ACTIVITY REGULATION:\s(.*?)(?=\{|$)")
}

def _preprocess_col_helper(col_name: str):
    unavailable = set()

    def inner(s: str):
        if not isinstance(s, str):
            return s

        try:
            matches = re.findall(_info_extr_patterns[col_name], s)
        except KeyError:
            if col_name not in unavailable:
                print(f"No preprocessing for '{col_name}' — skipping.")
                unavailable.add(col_name)
            return s

        if not matches:
            return s

        return matches[0] if len(matches) == 1 else ",".join(matches) # NOTE, need to make it a list! -- not a string!

    return inner

def preprocess_col(df: pd.DataFrame, col_name: str) -> None:
    df[col_name] = df[col_name].apply(_preprocess_col_helper(col_name))
