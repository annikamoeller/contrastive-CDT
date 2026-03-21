#!/bin/bash

# 1. Define the environments
TASKS=( 
    "OfflineAntRun-v0" 
    "OfflineCarCircle-v0" 
    "OfflineCarGoal1Gymnasium-v0" 
)

# 2. Define a submission template function
submit_job() {
    local ENV=$1
    local EXPERIMENT_NAME=$2
    local EXTRA_ARGS=$3
    
    local JOB_NAME="${ENV}_${EXPERIMENT_NAME}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=OSRL/logs/${JOB_NAME}_%j.log
#SBATCH --time=12:00:00
#SBATCH --partition=tue.gpu2.q
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis

# Load HPC environment
eval "\$(conda shell.bash hook)"
conda activate CDT_env 

export DSRL_DATASET_DIR="/home/20234949/thesis/datasets"
export PYTHONPATH="/home/20234949/thesis/OSRL:\$PYTHONPATH"
export MUJOCO_GL="egl"

# Run the training script 
python OSRL/osrl_contrastive/train.py \
    --task "$ENV" \
    --device "cuda:0" \
    --batch_size 256 \
    --project "CCDT_experiments_11_03" \
    $EXTRA_ARGS
EOF
    
    echo "Queued: $JOB_NAME"
}

# 3. Loop through environments and submit the experiments
for env in "${TASKS[@]}"; do
    echo "------------------------------------------------"
    echo "🚀 Submitting jobs for $env..."
    echo "------------------------------------------------"

    # ==========================================
    # BLOCK 1: THE BASELINES
    # ==========================================
    submit_job "$env" "Baseline" "--contrastive_weight 0.0"
    submit_job "$env" "CCDT_Default_2B" "--contrastive_weight 0.1 --num_buckets 2 --temperature 0.1"

    # ==========================================
    # BLOCK 2: BUCKET SCALING (Granularity)
    # Testing if the model benefits from a finer safety spectrum
    # ==========================================
    submit_job "$env" "CCDT_3Buckets" "--contrastive_weight 0.1 --num_buckets 3 --temperature 0.1"
    submit_job "$env" "CCDT_5Buckets" "--contrastive_weight 0.1 --num_buckets 5 --temperature 0.1"
    submit_job "$env" "CCDT_10Buckets" "--contrastive_weight 0.1 --num_buckets 10 --temperature 0.1"

    # ==========================================
    # BLOCK 3: SYNERGY - BUCKETS x HIGH WEIGHT
    # If we have 10 buckets, the latent space is crowded. 
    # Does a higher weight help push them apart?
    # ==========================================
    submit_job "$env" "CCDT_HighWeight_2B" "--contrastive_weight 0.5 --num_buckets 2 --temperature 0.1"
    submit_job "$env" "CCDT_HighWeight_5B" "--contrastive_weight 0.5 --num_buckets 5 --temperature 0.1"
    submit_job "$env" "CCDT_HighWeight_10B" "--contrastive_weight 0.5 --num_buckets 10 --temperature 0.1"

    # ==========================================
    # BLOCK 4: PRE-TRAINING WARMUP
    # Giving the encoder time to build the safety map before the policy acts
    # ==========================================
    submit_job "$env" "CCDT_Pretrain_5k" "--contrastive_weight 0.1 --num_buckets 2 --pretrain_steps 5000"
    submit_job "$env" "CCDT_Pretrain_10k" "--contrastive_weight 0.1 --num_buckets 2 --pretrain_steps 10000"

    # ==========================================
    # BLOCK 5: TEMPERATURE TUNING
    # Stricter/sharper boundaries between safe and unsafe clusters
    # ==========================================
    submit_job "$env" "CCDT_LowTemp_2B" "--contrastive_weight 0.1 --num_buckets 2 --temperature 0.05"

done

echo ""
echo "✅ All 33 ablation jobs submitted successfully to tue.gpu2.q!"