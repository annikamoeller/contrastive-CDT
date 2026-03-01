#!/bin/bash
#SBATCH --job-name=gen_pointgoal_variants
#SBATCH --output=logs/gen_pointgoal_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=tue.default.q
#SBATCH --mem=16G

echo "========== Variant Generation Job Started =========="
echo "Running on node: $(hostname)"
date
echo "===================================================="

module load anaconda
conda activate trac   # adjust if needed

# ---- INPUT BASE DATASET ----
BASE_DATASET="datasets/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"

# ---- OUTPUT DIRECTORY ----
OUTDIR="datasets_variants/PointGoal"
mkdir -p $OUTDIR

echo "[INFO] Base dataset: $BASE_DATASET"
echo "[INFO] Output directory: $OUTDIR"

# ---- GENERATE VARIANTS ----
python research/datasets/make_dsrl_variants.py \
    --input "$BASE_DATASET" \
    --outdir "$OUTDIR" \
    --clusters 20:60 30:50 35:45 \
    --seed 42

echo "========== Variant Generation Completed =========="
date
echo "=================================================="
