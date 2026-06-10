import os
import glob
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import traceback
import gymnasium as gym
import bullet_safety_gym # Essential for loading the Offline environments!
import dsrl  # <--- ADD THIS LINE!

# Ensure the project root is in the path
sys.path.insert(0, "/home/20234949/thesis")

from OSRL.osrl_contrastive.utils import load_model_and_config
from OSRL.osrl_contrastive.ccdt_trainer import ContrastiveCDTTrainer 

# --- CONFIGURATION ---
LOG_ROOT = "/home/20234949/thesis/logs"
TARGET_COST_SWEEP = [10, 20, 30, 40, 50, 60]
NUM_EPISODES = 20 
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def run_evaluation():
    results = []
    search_pattern = os.path.join(LOG_ROOT, "*-experiments", "*", "model.pt")
    checkpoint_files = glob.glob(search_pattern)
    
    for ckpt_path in checkpoint_files:
        folder_name = os.path.basename(os.path.dirname(ckpt_path))
        
        task_label = "Unknown"
        for t in ["AntRun", "CarCircle", "CarRun", "DroneCircle", "DroneRun"]:
            if t in folder_name:
                task_label = t
                break
        
        try:
            model, config = load_model_and_config(ckpt_path, device=DEVICE)
            
            # 1. Create the Physical Environment
            env_name = f"Offline{task_label}-v0"
            env = gym.make(env_name)
            
            # 2. Get boundaries safely
            boundaries = getattr(config, 'cost_boundaries', None)
            if boundaries is None:
                boundaries = [10.0, 20.0, 30.0, 40.0]
                
            # 3. Pass the Model AND the Env to the Trainer (just like train.py)
            trainer = ContrastiveCDTTrainer(
                model, 
                env, 
                cost_boundaries=boundaries, 
                device=DEVICE
            )
            
            seed = getattr(config, 'seed', 'N/A')
            reward_scale = getattr(config, 'reward_scale', 1.0)
            cost_scale = getattr(config, 'cost_scale', 1.0)
            
            # Get the base target return from config (usually the max possible)
            target_returns = getattr(config, 'target_returns', [(400, 10)])
            base_target_return = target_returns[0][0]
            
            for target in TARGET_COST_SWEEP:
                print(f"🚀 Eval: {task_label} | Seed {seed} | Target {target}")
                
                # 4. Use the exact signature from train.py
                ret, cost, length = trainer.evaluate(
                    num_rollouts=NUM_EPISODES, 
                    target_return=base_target_return * reward_scale,
                    target_cost=target * cost_scale
                )
                
                results.append({
                    "Task": task_label,
                    "Seed": seed,
                    "Target_Cost": target,
                    "Reward": ret,
                    "Actual_Cost": cost
                })
        except Exception as e:
            print(f"\n❌ CRITICAL CRASH on {folder_name}:")
            traceback.print_exc() 
            print("-" * 50)

    if not results:
        print("⚠️ No results gathered. Check if models loaded correctly.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.to_csv("eval_data.csv", index=False)
    return df

def generate_table1(df):
    table_data = df[df['Target_Cost'] == 10.0].copy()
    if table_data.empty: return
    
    table_data['Norm_Cost'] = table_data['Actual_Cost'] / 10.0
    summary = table_data.groupby('Task').agg({
        'Reward': ['mean', 'std'],
        'Norm_Cost': ['mean', 'std']
    })
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary.to_csv("table1_results.csv")
    print("\n--- Table 1 Data Saved ---")
    print(summary)

if __name__ == "__main__":
    data = run_evaluation()
    generate_table1(data)