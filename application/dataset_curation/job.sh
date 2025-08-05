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

source $HOME/miniconda3/etc/profile.d/conda.sh

conda activate gnn_func_annotation

export PYTHONPATH=/cluster/home/myehor01/data_processing/microbiome2function/M2F:$PYTHONPATH

# env vars
export SAMPLE_FILES=/cluster/tufts/bonhamlab/shared/sequencing/processed/humann/main
export SAVE_DATA_TO_DIR=/cluster/home/myehor01/data_processing
export LOGS_DIR=/cluster/home/myehor01/data_processing
export JOB_NAME=$SLURM_JOB_NAME

cd /cluster/home/myehor01/data_processing/microbiome2function/
python application/dataset_curation/data_mining.py

conda deactivate
