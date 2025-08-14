#!/bin/bash -l
#SBATCH -J data_prep
#SBATCH --time=3-00:00:00  #requested time (DD-HH:MM:SS)
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48g  #requesting 48GB of RAM total
#SBATCH --output=data_prep.%j.%N.out  #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=data_prep.%j.%N.err   #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL    #email options
#SBATCH --mail-user=ymishchyriak@wesleyan.edu

# have a clean start. purge all loaded modules in current environment
module purge

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate m2fvenv

export PYTHONPATH=/cluster/home/myehor01/data_processing/microbiome2function/M2F:$PYTHONPATH

# env vars
export RAW_DATA=/cluster/home/myehor01/data_processing/uniprot_mine_output_dir
export SAVE_PROCESSED_TO_DIR=/cluster/home/myehor01/data_processing
export LOGS_DIR=/cluster/home/myehor01/data_processing
export DB=/cluster/home/myehor01/data_processing/data_prep.db
export JOB_NAME=$SLURM_JOB_NAME

cd /cluster/home/myehor01/data_processing/microbiome2function/
python examples/gatnn_model_end2end/dataset_curation/data_preparation.py

conda deactivate
