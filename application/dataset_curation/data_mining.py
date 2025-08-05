from M2F import (extract_all_accessions_from_dir,
                fetch_save_uniprotkb_batches,
                configure_logging)
import re
import os

gene_fam_files_dir = os.getenv("SAMPLE_FILES")
output_dir = os.getenv("SAVE_DATA_TO_DIR")
job_name = os.getenv("JOB_NAME")
logs_dir = os.getenv("LOGS_DIR")

assert gene_fam_files_dir, "SAMPLE_FILES env var was not set!"
assert output_dir, "SAVE_DATA_TO_DIR env var was not set!"
assert job_name, "JOB_NAME env var was not set!"
assert logs_dir, "LOGS_DIR env var was not set!"

configure_logging(logs_dir)

accession_nums = extract_all_accessions_from_dir(gene_fam_files_dir, 
                                             pattern=re.compile(r".*_genefamilies\.tsv$"))


out = os.path.join(output_dir, job_name + "_output_dir")
os.makedirs(out, exist_ok=True)

fetch_save_uniprotkb_batches(
    uniref_ids=accession_nums,
    fields=["accession", "ft_domain", "cc_domain",
            "protein_families", "go_f", "go_p",
            "cc_function", "cc_catalytic_activity",
            "ec", "cc_pathway", "rhea", "cc_cofactor", "sequence"],
    batch_size=40_000,
    single_api_request_size=100,
    rps=10,
    save_to_dir=out,
)

print(f"Mined data is available at {out}")
