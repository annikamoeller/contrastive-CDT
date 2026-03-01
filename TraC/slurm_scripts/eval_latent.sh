#!/bin/bash

#SBATCH --job-name=trac_job
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=tue.gpu2.q
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis/TraC

# module purge
# module load gcc/10.2
# source ~/.bashrc
# conda activate TraC

python evaluate_latent.py --exp_path exp_safetygym/OfflineDroneRun-v0/SafetyDroneRun-v0-140-1990/TraC/cost-40.0/seed-0-seg