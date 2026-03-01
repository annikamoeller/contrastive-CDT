import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pyrallis
from dataclasses import dataclass
from pyrallis import field

# Import your existing codebase classes
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms import CDT, CDTTrainer
from osrl.common.exp_util import seed_all

# --------------------------------------------------------
# Helper: Safe Config Loading
# --------------------------------------------------------
def custom_load_config_and_model(path: str, best: bool = False, device: str = "cpu"):
    if os.path.isfile(path):
        model_path = path
        config_path = os.path.join(os.path.dirname(os.path.dirname(path)), "config.yaml")
    else:
        config_path = os.path.join(path, "config.yaml")
        model_path = os.path.join(path, "checkpoint", "model_best.pt" if best else "model.pt")

    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    print(f"Loading model from {model_path}")
    # Load to CPU first to avoid GPU OOM, then move to device later
    model = torch.load(model_path, map_location=device)
    
    return cfg, model

# --------------------------------------------------------
# Experiment Config
# --------------------------------------------------------
@dataclass
class TailRiskConfig:
    path: str = "log/your_experiment_path/checkpoint/model.pt"
    
    # The command we want to stress-test (e.g., 0 cost)
    target_reward: float = 300.0
    target_cost: float = 0.0
    
    eval_episodes: int = 100
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "figs"

# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------
@pyrallis.wrap()
def main(args: TailRiskConfig):
    # 1. Load Config & Model State
    cfg, model_state = custom_load_config_and_model(args.path, device=args.device)
    seed_all(args.seed)

    # Detect Task Name
    task_name = cfg["task"]
    print(f"Task detected: {task_name}")

    # 2. Setup Environment
    if "Metadrive" in task_name:
        import gym
    else:
        import gymnasium as gym

    env = wrap_env(
        env=gym.make(task_name),
        reward_scale=cfg["reward_scale"],
    )
    env = OfflineEnvWrapper(env)
    
    # 3. Initialize CDT Model Structure
    target_entropy = -env.action_space.shape[0]
    cdt_model = CDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=cfg["embedding_dim"],
        seq_len=cfg["seq_len"],
        episode_len=cfg["episode_len"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        attention_dropout=cfg["attention_dropout"],
        residual_dropout=cfg["residual_dropout"],
        embedding_dropout=cfg["embedding_dropout"],
        time_emb=cfg["time_emb"],
        use_rew=cfg["use_rew"],
        use_cost=cfg["use_cost"],
        cost_transform=cfg["cost_transform"],
        add_cost_feat=cfg["add_cost_feat"],
        mul_cost_feat=cfg["mul_cost_feat"],
        cat_cost_feat=cfg["cat_cost_feat"],
        action_head_layers=cfg["action_head_layers"],
        cost_prefix=cfg["cost_prefix"],
        stochastic=cfg["stochastic"],
        init_temperature=cfg["init_temperature"],
        target_entropy=target_entropy,
    )
    
    # Load weights
    state_dict = model_state["model_state"] if "model_state" in model_state else model_state
    cdt_model.load_state_dict(state_dict)
    cdt_model.to(args.device)
    cdt_model.eval()

    # 4. Initialize Trainer (Used only for its rollout method)
    # We pass dummy values for optimizer args since we won't train
    trainer = CDTTrainer(
        model=cdt_model,
        env=env,
        reward_scale=cfg["reward_scale"],
        cost_scale=cfg["cost_scale"],
        cost_reverse=cfg.get("cost_reverse", False), # Handle older configs safely
        device=args.device
    )

    # 5. Run Rollout Loop
    print(f"\n--- Starting Cost Analysis for {task_name} ---")
    print(f"Target Reward: {args.target_reward}")
    print(f"Target Cost: {args.target_cost}")
    
    raw_costs = []
    
    # Scale targets once
    scaled_ret_target = args.target_reward * cfg["reward_scale"]
    scaled_cost_target = args.target_cost * cfg["cost_scale"]

    for i in range(args.eval_episodes):
        # We use the trainer's built-in rollout which handles the transformer context
        ret, length, cost = trainer.rollout(
            cdt_model, 
            env, 
            target_return=scaled_ret_target, 
            target_cost=scaled_cost_target
        )
        
        # Unscale the cost to get real world units
        real_cost = cost / cfg["cost_scale"]
        raw_costs.append(real_cost)
        
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}/{args.eval_episodes} | Cost: {real_cost:.2f}")

    # 6. Analysis
    costs = np.array(raw_costs)
    mean_cost = np.mean(costs)

    print("\n--- Results ---")
    print(f"Mean Cost: {mean_cost:.4f}")
    print(f"Max Cost: {np.max(costs):.4f}")
    print(f"Safe Episodes (<= {args.target_cost}): {np.sum(costs <= args.target_cost)}/{args.eval_episodes}")

    # 7. Determine Output Path based on Task
    save_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cdt_cost_histogram.png")

    # 8. Plotting
    plt.figure(figsize=(10, 6))
    # Use simple hist if kde fails due to low variance
    try:
        sns.histplot(costs, bins=20, kde=True, color='red', alpha=0.6)
    except:
        plt.hist(costs, bins=20, color='red', alpha=0.6)
        
    plt.axvline(mean_cost, color='blue', linestyle='--', label=f'Mean: {mean_cost:.2f}')
    plt.axvline(args.target_cost, color='green', linewidth=2, label=f'Target: {args.target_cost}')
    
    plt.title(f"CDT Cost Analysis: {task_name} (Target Cost={args.target_cost})")
    plt.xlabel("Actual Cost")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()