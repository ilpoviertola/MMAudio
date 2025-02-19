#!/bin/bash

#SBATCH --job-name=mma-cn_test-train
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gputest
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2,nvme:100
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=00:15:00

module load git
export PATH="/projappl/project_2000936/viertoli/MMAudio/env/bin:$PATH"
set -e

# Define the YAML file path as the first argument provided to the script
yaml_file="$1"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun torchrun --standalone --nproc_per_node=2 train.py --config-name $yaml_file debug=True
