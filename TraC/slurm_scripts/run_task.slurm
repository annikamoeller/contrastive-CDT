#!/bin/bash

#SBATCH --job-name=trac_job
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=tue.gpu2.q
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis/TraC

module purge
module load gcc/10.2
source ~/.bashrc
conda activate TraC

echo "Dataset Path: $DATASET_PATH"

# ---------------------------------------------------------
# Dynamic Eval Env Extraction
# ---------------------------------------------------------
# 1. Get filename (e.g., "SafetyPointGoal1Gymnasium-v0-100-2022")
FILENAME=$(basename "$DATASET_PATH" .hdf5)

# 2. Strip metadata (remove the last two hyphenated parts: cost and seed)
# e.g. "SafetyPointGoal1Gymnasium-v0-100-2022" -> "SafetyPointGoal1Gymnasium-v0"
ENV_ID=${FILENAME%-*-*}

# 3. Convert "Safety" to "Offline" for the eval argument
# e.g. "SafetyPointGoal1Gymnasium-v0" -> "OfflinePointGoal1Gymnasium-v0"
EVAL_ENV=${ENV_ID//Safety/Offline}

echo "Extracted Env ID: $ENV_ID"
echo "Setting Eval Env: $EVAL_ENV"
# ---------------------------------------------------------

python train.py \
    --config=./configs/trac_safetygym.yaml \
    --path="exp_safetygym/" \
    --seed=0 \
    --eval_env="$EVAL_ENV" \
    --dataset_path="$DATASET_PATH" \
    --target_cost=40 \
    --alpha=0.1 \
    --gamma=0.99 \
    --eta=0.5 \
    --seg_ratio=0.75 \
    --kappa=0.7 \
    --safe_top_perc=0.5 \
    --safe_bottom_perc=0.5