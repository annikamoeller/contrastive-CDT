#!/bin/bash
#SBATCH --job-name=ccdt_verify
#SBATCH --output=OSRL/logs/verify_%j.log
#SBATCH --time=00:30:00
#SBATCH --partition=tue.gpu2.q
#SBATCH --gpus=1                # Slurm assigns 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis

# 1. Load your HPC environment
eval "$(conda shell.bash hook)"
conda activate CDT_env 

export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL:$PYTHONPATH"
export MUJOCO_GL="egl"

# 2. Define the verification grid
TASKS=("OfflineCarCircle-v0" "OfflineCarGoal1Gymnasium-v0" "OfflineAntRun-v0")
TEST_BUCKETS=5

for TASK in "${TASKS[@]}"; do
    echo "------------------------------------------------"
    echo "🚀 TESTING: $TASK | BUCKETS: $TEST_BUCKETS"
    echo "------------------------------------------------"
    
    # Use --device cuda (without a number) to let PyTorch pick the Slurm-assigned GPU
    # If the script requires a number, use 'cuda:0'
    # Update these flags in your Slurm script loop:
    python OSRL/osrl_contrastive/train.py \
        --task "$TASK" \
        --num_buckets "$TEST_BUCKETS" \
        --pretrain_steps 10 \
        --update_steps 1000 \
        --eval_every 100 \
        --eval_episodes 2 \
        --batch_size 16 \
        --device "cuda:0" \
        --project "Thesis_Verify" \
        --group "Line_Graph_Test"
done

echo "✅ Pipeline verification complete."