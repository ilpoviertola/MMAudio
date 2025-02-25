#!/bin/bash

#SBATCH --job-name=mma-cn_train
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpusmall
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2,nvme:400
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=122500M
#SBATCH --time=36:00:00

module load git
export PATH="/projappl/project_2000936/viertoli/MMAudio/env/bin:$PATH"
set -e

# Define the YAML file path as the first argument provided to the script
yaml_file="$1"

# Copy data to LOCAL_SCRATCH
mkdir -p $LOCAL_SCRATCH/avssemantic-single-source
cp -r /scratch/project_2000936/viertoli/datasets/avssemantic-single-source/avs-* $LOCAL_SCRATCH/avssemantic-single-source
cp -r /scratch/project_2000936/viertoli/datasets/avssemantic-single-source/*_eval_cache $LOCAL_SCRATCH/avssemantic-single-source

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun torchrun --standalone --nproc_per_node=2 train.py --config-name $yaml_file num_workers=10
