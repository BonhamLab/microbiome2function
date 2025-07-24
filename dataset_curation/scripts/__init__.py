# pkg:
from .mining_utils import *
# env:
from os import getenv
from dotenv import load_dotenv
load_dotenv()

FETCHED_DATA_EXAMPLE = getenv("FETCHED_DATA")
RAW_DATA_EXAMPLE = getenv("RAW_DATA")

__all__ = [
    # data examples
    "FETCHED_DATA_EXAMPLE",
    "RAW_DATA_EXAMPLE",
    # mining utils
    "unirefs_from_tsv",
    "retrieve_fields_for_unirefs",
    # cleaning utils
    "clean_col",
    "clean_all_entries",
]
