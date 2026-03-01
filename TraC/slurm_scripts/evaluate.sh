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

python eval_script.py --exp_path exp_safetygym/OfflineDroneCircle-v0/SafetyDroneCircle-v0-100-1923/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_20_60/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_30_50/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_35_45/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_0_33/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_33_66/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_66_99/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_small_10/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_small_5/TraC/cost-40.0/seed-0-seg
# python eval_script.py --exp_path exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_small_25/TraC/cost-40.0/seed-0-seg