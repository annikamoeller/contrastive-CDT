#!/bin/bash
#SBATCH --job-name=condt_ablation       # Name of the job
#SBATCH --output=logs/condt_%j.out      # Standard output log (%j = Job ID)
#SBATCH --error=logs/condt_%j.err       # Standard error log
#SBATCH --time=24:00:00                 # Maximum wall time (HH:MM:SS)
#SBATCH --partition=tue.gpu2.q
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=8               # Number of CPU cores for data loading
#SBATCH --mem=32G                       # RAM allocation
#SBATCH --nodes=1                       # All hardware on a single node
#SBATCH --chdir=/home/20234949/thesis/OSRL

# --- Environment Setup ---
# (Uncomment and modify these lines based on your cluster's setup)
# module load miniconda3
# module load cuda/11.8
eval "$(conda shell.bash hook)"
conda activate CDT_env

# --- Variables ---
# You can change these variables here or pass them when submitting the job
# For the circular boundary task:
# TASK="OfflinePointCircle1-v0"

# OR for the speed limit task:
TASK="OfflineCarRun-v0"

# TASK="OfflinePointGoal1-v0"

SEED=42

echo "Starting training job on $HOSTNAME"
echo "Task: $TASK | Seed: $SEED"

export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL:$PYTHONPATH"
# Force MuJoCo to render without a physical display
export MUJOCO_GL="egl"

# --- Execute the Training Script ---
# This example runs Experiment 1: Joint Training + In-Batch Masking + Granular Bins
python osrl_contrastive/train.py \
    --task $TASK \
    --seed $SEED \
    --contrastive_weight 0.5 \
    --temperature 0.1 \
    --discretization_type "granular" \
    --use_explicit_pairs False \
    --pretrain_steps 0 \
    --probe_every 10000 \
    --eval_every 5000 \
    --project "My_Thesis_ConDT" \
    --device "cuda"

echo "Job finished successfully."