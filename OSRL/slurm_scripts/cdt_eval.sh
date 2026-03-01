#!/bin/bash

#SBATCH --job-name=cdt_eval_suite
#SBATCH --output=logs/slurm_%j.out
#SBATCH --partition=tue.gpu2.q
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/20234949/thesis/OSRL

# Uncomment these if you need to initialize the shell environment
# module purge
# module load gcc/10.2
# source ~/.bashrc
# conda activate CDT_env

# ------------------------------------------------------------------
# LIST OF MODELS TO EVALUATE
# Add the path to the folder containing 'config.yaml' and 'checkpoint/'
# ------------------------------------------------------------------
MODELS=(
    "models/AntCircle/"
    "models/AntRun/"
    "models/CarCircle/"
    "models/CarRun/"
    "models/DroneCircle/"
    "models/DroneRun/"
    "models/PointGoal/"
)

# ------------------------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------------------------
for MODEL_PATH in "${MODELS[@]}"; do
    echo "========================================================"
    echo "Processing Model: $MODEL_PATH"
    echo "========================================================"

    # 1. Run Latent Space Visualization
    # Note: Using --output_dir instead of --save_plot because the script 
    # now generates the filename automatically based on the task name.
    echo "Running Latent Space Evaluation..."
    python examples/eval/eval_cdt_latent.py \
        --path "$MODEL_PATH" \
        --device "cuda:0" \
        --num_samples 2000 \
        --output_dir "figs"

    # 2. Run Tail Risk Analysis
    echo "Running Tail Risk Evaluation..."
    python examples/eval/eval_cdt_tail.py \
        --path "$MODEL_PATH" \
        --target_cost 0 \
        --eval_episodes 100 \
        --device "cuda:0" \
        --output_dir "figs"
        
    echo "Finished processing $MODEL_PATH"
    echo ""
done

echo "All evaluations complete."