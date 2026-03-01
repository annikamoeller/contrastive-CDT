set -x
env=$1
seg_ratio=$2
device_id=$3
config=./configs/trac_safetygym.yaml
path=./exp_safetygym
alpha=0.1
gamma=0.99
eta=0.5
kappa=0.7
safe_top_perc=0.5
safe_bottom_perc=0.5
eval_env=Offline${env}Gymnasium-v0
dataset_path="C:/Users/annik/.dsrl/datasets/Safety${env}Gymnasium-*"
seed=0
target_cost=20
python train.py --config=$config --path=$path --seed=$seed --eval_env=$eval_env --dataset_path=$dir --target_cost=$target_cost \
            --alpha=$alpha --gamma=$gamma --eta=$eta --seg_ratio=$seg_ratio --kappa=$kappa --safe_top_perc=$safe_top_perc --safe_bottom_perc=$safe_bottom_perc


python train.py --config=./configs/trac_safetygym.yaml --path=./exp_safetygym --seed=0 --eval_env=OfflinePointGoal1Gymnasium-v0 --dataset_path="datasets/SafetyPointGoal1Gymnasium-*" --target_cost=20 --alpha=0.1 --gamma=0.99 --eta=0.5 --seg_ratio=0.75 --kappa=0.7 --safe_top_perc=0.5 --safe_bottom_perc=0.5
python make_dsrl_variants.py --input datasets/SafetyPointGoal1Gymnasium-v0-100-2022.hdf5 --outdir datasets_variants/PointGoal/

# safetygym=(PointButton1 PointButton2 PointCircle1 PointCircle2
#             PointGoal1 PointGoal2 PointPush1 PointPush2
#             CarButton1 CarButton2 CarCircle1 CarCircle2
#             CarGoal1 CarGoal2 CarPush1 CarPush2
#             SwimmerVelocity HopperVelocity HalfCheetahVelocity Walker2dVelocity
#             AntVelocity)

# # for env in $bulletgym; do
# if [ $env = "SwimmerVelocity" ] || [ $env = "HopperVelocity" ] || [ $env = "HalfCheetahVelocity" ] || [ $env = "Walker2dVelocity" ] || [ $env = "AntVelocity" ]; then
#     eval_env=Offline${env}Gymnasium-v1
# else
#     eval_env=Offline${env}Gymnasium-v0
# fi
# dataset_path="C:/Users/annik/.dsrl/datasets/Safety${env}Gymnasium-*"
# echo $eval_env
# echo $dataset_path
# for dir in $dataset_path; do
#     for target_cost in 20 40 80; do
#         for seed in 0 10 20; do
#             echo 'alpha='$alpha 'seed='$seed
#             CUDA_VISIBLE_DEVICES=$device_id python train.py --config=$config --path=$path --seed=$seed --eval_env=$eval_env --dataset_path=$dir --target_cost=$target_cost \
#             --alpha=$alpha --gamma=$gamma --eta=$eta --seg_ratio=$seg_ratio --kappa=$kappa --safe_top_perc=$safe_top_perc --safe_bottom_perc=$safe_bottom_perc
#         done
#     done
# done
# # done