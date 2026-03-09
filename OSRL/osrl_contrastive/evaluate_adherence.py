import argparse
import os
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from osrl.algorithms import CDT
from osrl_contrastive.ccdt import ContrastiveCDT

@torch.no_grad()
def evaluate_model_adherence(model, env_id, target_costs, target_return, num_rollouts, device="cuda"):
    """
    Runs fast vectorized evaluations for a list of target costs and returns the actual costs incurred.
    """
    model.eval()
    actual_costs_incurred = []

    for t_cost in target_costs:
        print(f"Testing Target Cost: {t_cost}...")
        
        vec_env = gym.vector.make(env_id, num_envs=num_rollouts, asynchronous=True)
        obs, _ = vec_env.reset()
        
        seq_len = model.seq_len
        
        # Buffers
        states = torch.zeros((num_rollouts, seq_len, model.state_dim), dtype=torch.float32, device=device)
        actions = torch.zeros((num_rollouts, seq_len, model.action_dim), dtype=torch.float32, device=device)
        returns = torch.zeros((num_rollouts, seq_len, 1), dtype=torch.float32, device=device)
        costs = torch.zeros((num_rollouts, seq_len, 1), dtype=torch.float32, device=device)
        time_steps = torch.zeros((num_rollouts, seq_len), dtype=torch.long, device=device)
        
        episode_costs = np.zeros(num_rollouts)
        active_envs = np.ones(num_rollouts, dtype=bool)
        
        target_return_tensor = torch.tensor([target_return] * num_rollouts, device=device, dtype=torch.float32)
        target_cost_tensor = torch.tensor([t_cost] * num_rollouts, device=device, dtype=torch.float32)
        ep_costs_tensor = torch.zeros((num_rollouts, 1), device=device, dtype=torch.float32)
        
        t = 0
        while active_envs.any():
            states[:, t % seq_len] = torch.tensor(obs, dtype=torch.float32, device=device)
            time_steps[:, t % seq_len] = t
            returns[:, t % seq_len] = target_return_tensor.unsqueeze(-1)
            costs[:, t % seq_len] = target_cost_tensor.unsqueeze(-1)
            
            if t < seq_len:
                seq_s, seq_a = states[:, :t+1], actions[:, :t+1]
                seq_r, seq_c, seq_t = returns[:, :t+1], costs[:, :t+1], time_steps[:, :t+1]
            else:
                idx = torch.arange(t - seq_len + 1, t + 1) % seq_len
                seq_s, seq_a = states[:, idx], actions[:, idx]
                seq_r, seq_c, seq_t = returns[:, idx], costs[:, idx], time_steps[:, idx]
                
            # Forward pass (handles both Vanilla and Contrastive returns)
            preds = model(
                states=seq_s, actions=seq_a, returns_to_go=seq_r, costs_to_go=seq_c, 
                time_steps=seq_t, episode_cost=ep_costs_tensor
            )
            
            action_preds = preds[0] # First returned item is always the action prediction
            current_actions = action_preds[:, -1]
            
            # If stochastic, take the mean for deterministic evaluation
            if hasattr(current_actions, 'mean'):
                current_actions = current_actions.mean
                
            actions[:, t % seq_len] = current_actions
            cpu_actions = current_actions.cpu().numpy()
            
            obs, rews, terminateds, truncateds, infos = vec_env.step(cpu_actions)
            dones = terminateds | truncateds
            
            for i in range(num_rollouts):
                if active_envs[i]:
                    step_cost = infos["cost"][i] if "cost" in infos else 0.0
                    episode_costs[i] += step_cost
                    
                    target_return_tensor[i] -= rews[i]
                    target_cost_tensor[i] -= step_cost
                    ep_costs_tensor[i] += step_cost
                    if dones[i]: active_envs[i] = False
            t += 1
            
        vec_env.close()
        avg_cost = np.mean(episode_costs)
        print(f"  -> Actual Avg Cost Incurred: {avg_cost:.2f}")
        actual_costs_incurred.append(avg_cost)
        
    return actual_costs_incurred

def main():
    parser = argparse.ArgumentParser(description="Evaluate Constraint Adherence")
    parser.add_argument("--task", type=str, default="OfflineCarCircle-v0")
    parser.add_argument("--vanilla_weights", type=str, required=True, help="Path to Vanilla CDT .pt file")
    parser.add_argument("--ccdt_weights", type=str, required=True, help="Path to Contrastive CDT .pt file")
    parser.add_argument("--target_return", type=float, default=250.0)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    env = gym.make(args.task)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize Models
    print("Loading Vanilla CDT...")
    vanilla_model = CDT(state_dim=state_dim, action_dim=action_dim, max_action=max_action, seq_len=20, embedding_dim=128).to(args.device)
    vanilla_model.load_state_dict(torch.load(args.vanilla_weights, map_location=args.device))

    print("Loading Contrastive CDT...")
    ccdt_model = ContrastiveCDT(state_dim=state_dim, action_dim=action_dim, max_action=max_action, seq_len=20, embedding_dim=128, contrastive_dim=64).to(args.device)
    ccdt_model.load_state_dict(torch.load(args.ccdt_weights, map_location=args.device))

    # Define the sweep grid
    target_costs = [0, 5, 10, 20, 30, 40, 50]

    print("\n--- Evaluating Vanilla CDT ---")
    vanilla_actual_costs = evaluate_model_adherence(vanilla_model, args.task, target_costs, args.target_return, args.num_rollouts, args.device)

    print("\n--- Evaluating Contrastive CDT ---")
    ccdt_actual_costs = evaluate_model_adherence(ccdt_model, args.task, target_costs, args.target_return, args.num_rollouts, args.device)

    # --- Generate the Plot ---
    plt.figure(figsize=(8, 8))
    
    # The Ideal Line (y = x)
    plt.plot(target_costs, target_costs, 'k--', linewidth=2, label="Ideal Adherence (y=x)")
    
    # Vanilla Plot
    plt.plot(target_costs, vanilla_actual_costs, 'r-o', linewidth=2, markersize=8, label="Vanilla CDT")
    
    # CCDT Plot
    plt.plot(target_costs, ccdt_actual_costs, 'g-^', linewidth=2, markersize=8, label="Contrastive CDT")

    # Formatting
    plt.fill_between(target_costs, 0, target_costs, color='green', alpha=0.1, label="Safe Zone")
    plt.fill_between(target_costs, target_costs, max(max(vanilla_actual_costs), max(ccdt_actual_costs))+10, color='red', alpha=0.1, label="Constraint Violated")

    plt.xlabel("Target Cost Prompted", fontsize=14)
    plt.ylabel("Actual Cost Incurred", fontsize=14)
    plt.title(f"Constraint Adherence on {args.task}", fontsize=16, fontweight='bold')
    plt.xlim(-2, max(target_costs) + 5)
    plt.ylim(-2, max(max(vanilla_actual_costs), max(ccdt_actual_costs)) + 10)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Save
    save_path = f"adherence_plot_{args.task}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Evaluation complete! Plot saved to: {save_path}")

if __name__ == "__main__":
    main()