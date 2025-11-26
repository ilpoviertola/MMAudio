#!/bin/bash

#SBATCH --job-name=mma-extr-vid-feats
#SBATCH --account=project_2000936
#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --partition=gpusmall
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2,nvme:400
#SBATCH --cpus-per-task=20
# #SBATCH --mem-per-gpu=122500M
#SBATCH --mem-per-gpu=245000M
#SBATCH --time=36:00:00

export PATH="/projappl/project_2000936/viertoli/MMAudio/env/bin:$PATH"
set -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun torchrun --standalone --nproc_per_node=2 training/extract_video_training_latents.py --output_dir /scratch/project_2000936/viertoli/datasets/avssemantic-single-source-unagg-full-sync-map --latent_dir /scratch/project_2000936/viertoli/datasets/tmp
