# builtins:
from typing import Union, Tuple, Any, Optional, Dict, List
import logging
import re

# third-party:
import pandas as pd

# *-----------------------------------------------*
#                      GLOBALS
# *-----------------------------------------------*

_logger = logging.getLogger(__name__)

# ---------regexes---------
_inline_pubmed_re = re.compile(r"\s*\(PubMed:\d+(?:\s*,\s*PubMed:\d+)*\)")
_brace_pubmed_re  = re.compile(r"\s*\{[^}]*PubMed:[^}]*\}")
_TRIM_PUNCT       = re.compile(r'(^[^\w]+|[^\w]+$)')
_CLEAN_PUNCT      = re.compile(r'[^A-Za-z0-9\s\-/]')
_MULTI_WS         = re.compile(r'\s+')

AVAILABLE_EXTRACTION_PATTERNS: Dict[str, re.Pattern] = {
    "Domain [FT]"                       : re.compile(r"DOMAIN\s(\d+\.\.\d+)"),
    "Domain [CC]"                       : re.compile(r"DOMAIN:\s(.*?)(?=\s\{|$)"),
    "Protein families"                  : re.compile(r"(\w.*?)(?=;|$)"),
    "Gene Ontology (molecular function)": re.compile(r"\[(GO:\d{7})\]"),
    "Gene Ontology (biological process)": re.compile(r"\[(GO:\d{7})\]"),
    "Interacts with"                    : re.compile(r"([A-Z0-9]+?)(?=\s|;|$)"),
    "Function [CC]"                     : re.compile(r"FUNCTION:\s*(.+?\S)(?=\s*\{|$)"),
    "Catalytic activity"                : re.compile(r"Reaction=(.*?)(?=;|\.|$)"),
    "EC number"                         : re.compile(r"([\d\.-]+?)(?=;|$)"),
    "Pathway"                           : re.compile(r"PATHWAY:\s*(.+?\S)(?=\s*;\s*PATHWAY:|\s*\{|\s*$)"),
    "Rhea ID"                           : re.compile(r"RHEA:(\d*?)(?=\s|$)"),
    "Cofactor"                          : re.compile(r"Name=(.*?)(?=;|$)"),
    "Activity regulation"               : re.compile(r"ACTIVITY REGULATION:\s*(.+?\S)(?=\s*\{|$)")
}

# cache to avoid spamming logs when a pattern is missing
_NO_PATTERN_NOTIFIED: set[str] = set()

# *-----------------------------------------------*
#                      UTILS
# *-----------------------------------------------*

def strip_pubmed(text: str) -> str:
    """Remove inline or braced PubMed references from a string."""
    if not isinstance(text, str):
        return text
    text = _inline_pubmed_re.sub("", text)
    text = _brace_pubmed_re.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def normalize(s: str) -> str:
    """Lower-case, strip punctuation (except - and /) and collapse whitespace."""
    s = s.strip()
    s = _TRIM_PUNCT.sub("", s)
    s = _CLEAN_PUNCT.sub("", s)
    s = _MULTI_WS.sub(" ", s)
    return s.lower()


def _clean_col_helper(
    col_name: str,
    apply_norm: bool = True,
    apply_strip_pubmed: bool = True
):
    """
    Return a function that cleans a single cell in column `col_name`.
    Duplicates are removed, result is always a tuple (possibly empty).
    """

    def _inner(value: str) -> Union[Tuple[str, ...], Any]:
        # pass through non-strings (will become empty tuple later if it's a NaN)
        if not isinstance(value, str):
            return value

        text = strip_pubmed(value) if apply_strip_pubmed else value

        # try pattern extraction
        matches: List[str]
        pattern = AVAILABLE_EXTRACTION_PATTERNS.get(col_name)
        if pattern is not None:
            matches = re.findall(pattern, text)
        else:
            if col_name not in _NO_PATTERN_NOTIFIED:
                _logger.info(f"No extraction rule for column '{col_name}'.")
                _NO_PATTERN_NOTIFIED.add(col_name)
            matches = []

        matches = matches or [text] # fallback to raw string
        if apply_norm:
            matches = [normalize(m) for m in matches]

        # deduplicate while preserving order
        return tuple(dict.fromkeys(matches))

    return _inner


def clean_col(
    df: pd.DataFrame,
    col_name: str,
    apply_norm: bool = True,
    apply_strip_pubmed: bool = True,
    inplace: bool = True
) -> pd.DataFrame:
    """
    Clean a single column in *df*.
    • Extracts structured pieces via regex (if available).
    • Optionally strips PubMed refs and normalises tokens.
    • Always returns tuples; NaNs become empty tuples.
    """
    _logger.info(
        f"Cleaning column '{col_name}' (apply_norm={apply_norm}, "
        f"apply_strip_pubmed={apply_strip_pubmed}, inplace={inplace})"
    )

    if col_name not in df.columns:
        raise KeyError(f"Column '{col_name}' not found in DataFrame.")

    if not inplace:
        df = df.copy(deep=True)

    cleaner = _clean_col_helper(col_name, apply_norm, apply_strip_pubmed)
    df.loc[:, col_name] = df[col_name].map(lambda entry: () if pd.isna(entry) else cleaner(entry))

    _logger.info(f"Finished processing '{col_name}'.")
    return df


def clean_cols(
    df: pd.DataFrame,
    col_names: List[str],
    apply_norms: Optional[Dict[str, bool]] = None,
    apply_strip_pubmeds: Optional[Dict[str, bool]] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Clean multiple columns.  
    • `col_names` - list of columns to process.  
    • `apply_norms` / `apply_strip_pubmeds` - per-column boolean maps
      (default True for all).  
    """
    _logger.info(f"Cleaning columns: {col_names}")

    missing = [c for c in col_names if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not in DataFrame: {missing}")

    if not inplace:
        df = df.copy(deep=True)

    default_flags = {c: True for c in col_names}
    apply_norms = {**default_flags, **(apply_norms or {})}
    apply_strip_pubmeds = {**default_flags, **(apply_strip_pubmeds or {})}

    for col in col_names:
        df = clean_col(
            df,
            col,
            apply_norm=apply_norms[col],
            apply_strip_pubmed=apply_strip_pubmeds[col],
            inplace=True # prevent repeated deep copies
        )

    _logger.info("Successfully cleaned requested columns.")
    return df


__all__ = ["clean_col", "clean_cols"]

if __name__ == "__main__":
    pass
