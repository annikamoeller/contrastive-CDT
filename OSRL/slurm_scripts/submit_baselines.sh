#!/bin/bash

# 1. Define the exact 5 tasks from the CDT paper
TASKS=( 
    "OfflineAntRun-v0" 
    "OfflineCarCircle-v0" 
    "OfflineCarRun-v0" 
    "OfflineDroneCircle-v0" 
    "OfflineDroneRun-v0" 
)

# Define the seeds
SEEDS=(0 1 2)

# 2. Define the submission template function
submit_job() {
    local ENV=$1
    local SEED=$2
    
    # Determine Batch Size: Drone tasks use 4096 in the paper; others use 2048.
    local BATCH_SIZE=2048
    if [[ "$ENV" == *"Drone"* ]]; then
        BATCH_SIZE=4096
    fi
    
    local JOB_NAME="CDT_Base_${ENV}_S${SEED}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=OSRL/logs/${JOB_NAME}_%j.log
#SBATCH --error=OSRL/logs/${JOB_NAME}_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=tue.gpu2.q
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis

# Load HPC environment
eval "\$(conda shell.bash hook)"
conda activate CDT_env 

# Set paths and environment variables
export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL:\$PYTHONPATH"
export MUJOCO_GL="egl"

# Run the training script 
# baseline mode: weight 0, 1 bucket, 0 pretrain
python OSRL/osrl_contrastive/train.py \
    --task "$ENV" \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --device "cuda:0" \
    --project "OSRL-baselines" \
    --contrastive_weight 0.0 \
    --pretrain_steps 0 \
    --num_buckets 1 \
    --eval_episodes 20 \
    --probe_every 999999
EOF
    
    echo "Queued: $JOB_NAME (Batch Size: $BATCH_SIZE)"
}

# 3. Loop through environments and seeds
for env in "${TASKS[@]}"; do
    echo "------------------------------------------------"
    echo "🚀 Submitting baseline jobs for $env..."
    echo "------------------------------------------------"

    for seed in "${SEEDS[@]}"; do
        submit_job "$env" "$seed"
    done
done

echo ""
echo "✅ All 15 baseline jobs submitted successfully to tue.gpu2.q!"