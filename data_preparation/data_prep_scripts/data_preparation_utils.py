import pandas as pd
import re

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

def strip_pubmed(text: str) -> str:
    if not isinstance(text, str):
        return text
    # in parens
    text = _inline_re.sub("", text)
    # in braces
    text = _brace_re.sub("", text)
    # collapse any accidental double‑spaces
    return re.sub(r"\s{2,}", " ", text).strip()

_ws_re = re.compile(r"\s+")

def normalize(s: str) -> str:
    s = _ws_re.sub(" ", s.strip().casefold())
    s = s.rstrip(" .;,")
    return s

def _preprocess_col_helper(col_name: str, apply_norm: bool = True, apply_strip_pubmed: bool = True):
    unknown = set()

    def _inner(value: str):
        
        if not isinstance(value, str):
            return value
        
        text = strip_pubmed(value) if apply_strip_pubmed else value

        try:
            regex = _info_extr_patterns[col_name]
            matches = re.findall(regex, text)
        except KeyError:
            if col_name not in unknown:
                print(f"No preprocessing rule for '{col_name}' — leaving as is.")
                unknown.add(col_name)
            return normalize(text) if apply_norm else text

        if not matches:
            return normalize(text) if apply_norm else text

        if len(matches) == 1:
            res = matches[0]
            return normalize(res) if apply_norm else res

        cleaned = [normalize(m) for m in matches] if apply_norm else matches
        return tuple(dict.fromkeys(cleaned))

    return _inner

def preprocess_col(df: pd.DataFrame, col_name: str) -> None:
    df[col_name] = df[col_name].apply(_preprocess_col_helper(col_name))


if __name__ == "__main__":
    s = "DOMAIN 229..339; /note=\"SpaA-like prealbumin fold\"; /evidence=\"ECO:0000259|Pfam:PF17802\"; DOMAIN 371..406; /note=\"Gram-positive cocci surface proteins LPxTG\"; /evidence=\"ECO:0000259|Pfam:PF00746\""
    m = re.findall(_info_extr_patterns["Domain [FT]"], s)
    print(m)