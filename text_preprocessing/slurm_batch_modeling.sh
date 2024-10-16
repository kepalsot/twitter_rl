#!/bin/bash 

BATCH_NUM=$1
scontrol update JobName="roberta_${BATCH_NUM}" JobID=$SLURM_JOB_ID

#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 60
#SBATCH -n 50

# python module (latest version available on the cluster)
module load anacondapy/2023.03

# execute the modeling script 
python score_sentiment.py "$1"


