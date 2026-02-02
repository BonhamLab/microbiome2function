import os
import pandas as pd
import M2F
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv("/cluster/home/myehor01/data_processing/microbiome2function/.env")

raw_data   = os.getenv("RAW_DATA")
output_dir = os.getenv("SAVE_PROCESSED_TO_DIR")
logs_dir   = os.getenv("LOGS_DIR")
job_name   = os.getenv("JOB_NAME")
db_path    = os.getenv("DB")
api_key    = os.getenv("OPENAI_API_KEY")

assert raw_data,   "RAW_DATA env var was not set!"
assert output_dir, "SAVE_PROCESSED_TO_DIR env var was not set!"
assert logs_dir,   "LOGS_DIR env var was not set!"
assert job_name,   "JOB_NAME env var was not set!"
assert api_key,    "OPENAI_API_KEY env var was not set!"
assert db_path,    "DB env var (SQLite cache path) was not set!"

out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

M2F.configure_logging(logs_dir)

col_names=["Domain [FT]",
        "Domain [CC]",
        "Gene Ontology (molecular function)",
        "Gene Ontology (biological process)",
        "Function [CC]",
        "Catalytic activity",
        "EC number",
        "Pathway",
        "Cofactor",
        "Sequence"
]

apply_norms={"Domain [FT]" : False,
        "Domain [CC]" : True,
        "Gene Ontology (molecular function)" : False,
        "Gene Ontology (biological process)" : False,
        "Function [CC]" : True,
        "Catalytic activity" : False,
        "EC number" : False,
        "Pathway" : True,
        "Cofactor" : False,
        "Sequence" : False
}

try:
    aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t6_8M_UR50D", device="cuda:0")
    logger.info("AAChainEmbedder will use a CUDA device")
except Exception as e:
    logger.info(f"AAChainEmbedder will use a CPU, because: {e}")
    aa_embedder = M2F.AAChainEmbedder(model_key="esm2_t6_8M_UR50D", device="cpu")

if os.path.exists(db_path):
    caching_mode = "APPEND"
else:
    caching_mode = "CREATE/OVERRIDE"

txt_embedder = M2F.FreeTXTEmbedder(api_key,
                                model="LARGE_OPENAI_MODEL",
                                cache_file_path=db_path,
                                caching_mode=caching_mode,
                                max_cache_size_kb=20_000)

# total number of rows to process is: 859,660
def process_df_inplace(df: pd.DataFrame, *, col_names: list, apply_norms: dict) -> dict:
    # Note: original columns are:
    # Entry,Domain [FT],Domain [CC],Protein families,Gene Ontology (molecular function),Gene Ontology (biological process),Function [CC],Catalytic activity,EC number,Pathway,Rhea ID,Cofactor,Sequence
    # But we want to keep only a specific subset of them
    df.drop(columns=["Protein families", "Rhea ID"], inplace=True)

    # clean
    M2F.clean_cols(df, col_names=col_names, apply_norms=apply_norms, inplace=True)
    # encode
    M2F.embed_ft_domains(df, aa_embedder, inplace=True)
    M2F.embed_AAsequences(df, aa_embedder, inplace=True)
    M2F.embed_freetxt_cols(df, ["Domain [CC]", "Function [CC]", "Catalytic activity", "Pathway"], txt_embedder, inplace=True)
    _, gomf_meta = M2F.encode_go(df, "Gene Ontology (molecular function)", coverage_target=0.8, inplace=True)
    _, gobp_meta = M2F.encode_go(df, "Gene Ontology (biological process)", coverage_target=0.8, inplace=True)
    _, ec_meta = M2F.encode_ec(df, "EC number", inplace=True)
    _, cofactor_meta = M2F.encode_multihot(df, "Cofactor", inplace=True)

    return {"gomf_meta": gomf_meta, "gobp_meta": gobp_meta, "ec_meta": ec_meta, "cofactor_meta": cofactor_meta}

files = M2F.util.files_from(raw_data)
logger.info(f"Processing all the files from {raw_data}")
for i, file in enumerate(files, start=1):
    file_name = os.path.basename(file)
    out_pth = os.path.join(out, file_name.replace(".csv", ".zip"))
    # Note: this is needed in case we rerun the job that was stopped in the middle of execution
    # so that we don't duplicate files
    if os.path.exists(out_pth):
        logger.info(f"File number {i} ({file_name}) already exists; Skipping")
        continue
    logger.info(f"Processing file number {i}: {file_name}")
    # load
    df = pd.read_csv(file)
    # process
    meta = process_df_inplace(df, col_names=col_names, apply_norms=apply_norms)
    # save
    M2F.save_df(df, out_pth, metadata=meta)

print(f"Processed data is available at {out}")
