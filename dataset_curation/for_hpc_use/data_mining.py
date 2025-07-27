from scripts import (recommended_fields_example2, 
                     unirefs_from_multiple_files,
                     process_uniref_batches)
import re
import os


gene_fam_files_dir = os.getenv("SAMPLE_FILES")
output_dir = os.getenv("SAVE_DATA_TO_DIR")
job_name = os.getenv("JOB_NAME")

assert gene_fam_files_dir, "SAMPLE_FILES env var was not set!"
assert output_dir, "SAVE_DATA_TO_DIR env var was not set!"
assert job_name, "JOB_NAME env var was not set!"

accession_nums = unirefs_from_multiple_files(gene_fam_files_dir, 
                                             pattern=re.compile(r".*_genefamilies\.tsv$"))


out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

process_uniref_batches(
    uniref_ids=accession_nums,
    fields=recommended_fields_example2,
    batch_size=40_000,
    single_api_request_size=100,
    rps=10,
    save_to_dir=out,
    filter_out_bad_ids=True
)

print(f"Mined data is available at {out}")
