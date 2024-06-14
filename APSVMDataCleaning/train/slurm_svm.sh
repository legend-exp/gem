#!/bin/bash
#SBATCH --job-name=svm_search
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --time=1:30:00
#SBATCH --mem=64g
#SBATCH --constraint=cpu
#SBATCH --account=m2676
#SBATCH --image=docker:legendexp/legend-software:latest
#SBATCH --chdir=<path-to-train-directory>  
#SBATCH --output=<path-to-logs-dir>/svm_search_%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=<your-email>


echo "Job Start:"
date
echo "Node(s):  "$SLURM_JOB_NODELIST
echo "Job ID:  "$SLURM_JOB_ID

export HDF5_USE_FILE_LOCKING='FALSE'

# run script
shifter python svm_search.py --file ../data/l200-*-ml_train_dsp.lh5

echo "Job Complete:"
date