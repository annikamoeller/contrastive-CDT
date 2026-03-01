#!/bin/bash

DATASETS=(
    "datasets/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5"
    "datasets/SafetyDroneRun-v0-140-1990.hdf5"
    "datasets/SafetyDroneCircle-v0-100-1923.hdf5"
    "datasets/SafetyCarRun-v0-40-651.hdf5"
    "datasets/SafetyCarCircle1Gymnasium-v0-250-1271.hdf5"
    "datasets/SafetyAntRun-v0-150-1816.hdf5"
)

for i in ${!DATASETS[@]}; do
    sbatch --export=ALL,DATASET_PATH=${DATASETS[$i]} slurm_scripts/run_task.slurm
done
