#!/bin/bash -l
#SBATCH -J uniprot_mine
#SBATCH --time=3-10:00:00  #requested time (DD-HH:MM:SS)
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8g  #requesting 8GB of RAM total
#SBATCH --output=uprotmine.%j.%N.out  #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=uprotmine.%j.%N.err   #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL    #email options
#SBATCH --mail-user=ymishchyriak@wesleyan.edu

# have a clean start. purge all loaded modules in current environment
module purge

module load anaconda/2021.05

source activate gnn_func_annotation

# env vars
export SAMPLE_FILES=/cluster/tufts/bonhamlab/shared/sequencing/processed/humann/main
export SAVE_DATA_TO_DIR=/cluster/home/myehor01/data_processing
export JOB_NAME=$SLURM_JOB_NAME

cd /cluster/home/myehor01/data_processing/microbiome2function/dataset_curation/for_hpc_use
python data_mining.py

conda deactivate
