import pandas as pd
import re

# *-----------------------------------------------*
#                      GLOBALS
# *-----------------------------------------------*

_info_extr_patterns = {
    "Domain [FT]" : re.compile(r'DOMAIN\s(\d+..\d+)'), # <-- AA-seq fragment;
    # ^^^ use 're.compile(r'/note="([^"]+)"')' to extract human-readable description of the domain
    "Domain [CC]" : re.compile(r"DOMAIN:\s(.*?)(?=\s\{|$)"),
    "Protein families": re.compile(r"(\w.*?)(?=;|$)"),
    "Gene Ontology (molecular function)" : re.compile(r"\[(GO:\d{7})\]"),
    "Gene Ontology (biological process)" : re.compile(r"\[(GO:\d{7})\]"),
    "Interacts with" : re.compile(r"([A-Z0-9]+?)(?=\s|;|$)"),
    "Function [CC]" : re.compile(r"FUNCTION:\s*(.+?\S)(?=\s*\{|$)"),
    "Catalytic activity" : re.compile(r"Reaction=(.*?)(?=;|\.|$)"),
    "EC number" : re.compile(r"([\d\.-]+?)(?=;|$)"),
    "Pathway" : re.compile(r"PATHWAY:\s*(.+?\S)(?=\s*;\s*PATHWAY:|\s*\{|\s*$)"),
    "Rhea ID" : re.compile(r"RHEA:(\d*?)(?=\s|$)"),
    "Cofactor" : re.compile(r"Name=(.*?)(?=;|$)"),
    "Activity regulation" : re.compile(r"ACTIVITY REGULATION:\s*(.+?\S)(?=\s*\{|$)")
} # note, I our set we also have the AA sequences of proteins, but no extraction is needed there

_inline_re = re.compile(r"\s*\(PubMed:\d+(?:\s*,\s*PubMed:\d+)*\)")

_brace_re  = re.compile(r"\s*\{[^}]*PubMed:[^}]*\}")

_2normalizeORnot = {
    "Domain [FT]" : False,
    "Domain [CC]" : True,
    "Protein families": False,
    "Gene Ontology (molecular function)" : False,
    "Gene Ontology (biological process)" : False,
    "Interacts with" : False,
    "Function [CC]" : True,
    "Catalytic activity" : False,
    "EC number" : False,
    "Pathway" : True,
    "Rhea ID" : False,
    "Cofactor" : False,
    "Activity regulation" : True
}

_TRIM_PUNCT = re.compile(r'(^[^\w]+|[^\w]+$)')
_CLEAN_PUNCT = re.compile(r'[^A-Za-z0-9\s\-/]')
_MULTI_WS = re.compile(r'\s+')

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def _strip_pubmed(text: str) -> str:
    if not isinstance(text, str):
        return text
    # in parens
    text = _inline_re.sub("", text)
    # in braces
    text = _brace_re.sub("", text)
    # collapse any accidental double‑spaces
    return re.sub(r"\s{2,}", " ", text).strip()

def _normalize(s: str) -> str:
    # remove whitespace at the ends
    s = s.strip()
    # remove leading/trailing non-word characters
    s = _TRIM_PUNCT.sub('', s)
    # remove all other punctuation except hyphens/slashes
    s = _CLEAN_PUNCT.sub('', s)
    # collapse runs of spaces
    s = _MULTI_WS.sub(' ', s)
    # make lowercase
    return s.lower()

def _clean_col_helper(col_name: str, apply_norm: bool = True, apply_strip_pubmed: bool = True):
    unknown = set()

    def _inner(value: str):
        
        if not isinstance(value, str):
            return value
        
        text = _strip_pubmed(value) if apply_strip_pubmed else value

        try:
            _2normlize = _2normalizeORnot[col_name] and apply_norm
        except KeyError:
            _2normlize = apply_norm

        try:
            regex = _info_extr_patterns[col_name]
            matches = re.findall(regex, text)
        except KeyError:
            if col_name not in unknown:
                print(f"No preprocessing rule for '{col_name}' — leaving as is.")
                unknown.add(col_name)

            return _normalize(text) if _2normlize else text

        if not matches:
            return _normalize(text) if _2normlize else text

        if len(matches) == 1:
            res = matches[0]
            return _normalize(res) if _2normlize else res

        cleaned = [_normalize(m) for m in matches] if _2normlize else matches
        return tuple(dict.fromkeys(cleaned))

    return _inner

def clean_col(df: pd.DataFrame, col_name: str) -> None:
    df[col_name] = df[col_name].apply(_clean_col_helper(col_name))

def clean_all_entries(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy(deep=True)
    for col_name in new_df.columns:
        clean_col(new_df, col_name)
    return new_df


__all__ = [
    "clean_col",
    "clean_all_entries",
]

if __name__ == "__main__":
    pass
