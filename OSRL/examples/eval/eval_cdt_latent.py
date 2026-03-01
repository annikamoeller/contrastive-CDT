import os
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pyrallis
from dataclasses import dataclass
from typing import Tuple

# Import your codebase
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from osrl.algorithms import CDT
from osrl.common.exp_util import seed_all

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------
def custom_load_config_and_model(path: str, device: str = "cpu"):
    """Loads config and model safely."""
    if os.path.isfile(path):
        model_path = path
        config_path = os.path.join(os.path.dirname(os.path.dirname(path)), "config.yaml")
    else:
        config_path = os.path.join(path, "config.yaml")
        model_path = os.path.join(path, "checkpoint", "model.pt")

    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    print(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location=device)
    return cfg, model

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
@dataclass
class LatentConfig:
    path: str = "log/your_experiment/checkpoint/model.pt"
    num_samples: int = 2000  # Total points to plot
    device: str = "cpu"
    seed: int = 42
    # Removed specific save_plot file name, we will generate it based on env
    output_dir: str = "figs" 

# --------------------------------------------------------
# Hook Function to Capture Latent Representations
# --------------------------------------------------------
def get_latent_embeddings(model, env, num_samples, device, cfg):
    """
    Runs episodes and extracts the internal transformer representation 
    for both SAFE and UNSAFE moments.
    """
    embeddings = []
    labels = []  # "Safe" or "Unsafe"
    costs = []   # Actual cost values for color gradient
    
    print(f"Collecting {num_samples} latent samples...")
    model.eval()
    
    # Define targets to provoke behavior
    mixed_targets = [10.0, 50.0] 
    
    collected_count = 0
    
    while collected_count < num_samples:
        for target_cost in mixed_targets:
            state, _ = env.reset()
            
            # State: [1, 1, state_dim]
            states = torch.from_numpy(state).reshape(1, 1, model.state_dim).to(device=device, dtype=torch.float32)
            
            # Action: [1, 1, action_dim] 
            actions = torch.zeros((1, 1, model.action_dim), device=device, dtype=torch.float32)
            
            # Targets: [1, 1] 
            target_return_val = torch.tensor(300.0 * cfg["reward_scale"], device=device, dtype=torch.float32).reshape(1, 1)
            target_cost_val = torch.tensor(target_cost * cfg["cost_scale"], device=device, dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor([0], device=device, dtype=torch.long).reshape(1, 1)

            done = False
            while not done and collected_count < num_samples:
                # --- PREPARE INPUTS ---
                K = model.seq_len
                
                s = states[:, -K:]
                a = actions[:, -K:]
                r = target_return_val[:, -K:]
                c = target_cost_val[:, -K:]
                t = timesteps[:, -K:]
                
                cur_len = s.shape[1]
                if cur_len < K:
                    pad_len = K - cur_len
                    s = torch.cat([torch.zeros((1, pad_len, model.state_dim), device=device), s], dim=1)
                    a = torch.cat([torch.zeros((1, pad_len, model.action_dim), device=device), a], dim=1)
                    r = torch.cat([torch.zeros((1, pad_len), device=device), r], dim=1)
                    c = torch.cat([torch.zeros((1, pad_len), device=device), c], dim=1)
                    t = torch.cat([torch.zeros((1, pad_len), device=device, dtype=torch.long), t], dim=1)
                
                # --- EXTRACT EMBEDDING ---
                with torch.no_grad():
                    if model.time_emb:
                        time_emb = model.timestep_emb(t)
                    else:
                        time_emb = 0.0
                        
                    state_emb = model.state_emb(s) + time_emb
                    act_emb = model.action_emb(a) + time_emb
                    returns_emb = model.return_emb(r.unsqueeze(-1)) + time_emb
                    costs_emb = model.cost_emb(c.unsqueeze(-1)) + time_emb
                    
                    seq_list = [costs_emb, returns_emb, state_emb, act_emb]
                    sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3).reshape(1, 4 * K, model.embedding_dim)
                    
                    out = model.emb_norm(sequence)
                    for block in model.blocks:
                        out = block(out, padding_mask=None)
                    out = model.out_norm(out)
                    out = out.reshape(1, K, 4, model.embedding_dim)
                    latent_vector = out[0, -1, 2, :].cpu().numpy()
                
                # --- EXECUTE ACTION ---
                action_preds = model.action_head(out[0, -1, 2, :].unsqueeze(0).unsqueeze(0))
                
                if model.stochastic:
                    action = action_preds.mean[0, 0]
                else:
                    action = action_preds[0, 0]
                
                action_np = action.detach().cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(action_np)
                
                # --- CHECK SAFETY ---
                is_unsafe = info.get("cost", 0.0) > 0
                
                embeddings.append(latent_vector)
                labels.append("Unsafe" if is_unsafe else "Safe")
                costs.append(info.get("cost", 0.0))
                collected_count += 1
                
                # --- UPDATE CONTEXT ---
                cur_state_t = torch.from_numpy(next_state).to(device=device, dtype=torch.float32).reshape(1, 1, -1)
                states = torch.cat([states, cur_state_t], dim=1)
                cur_action_t = action.reshape(1, 1, -1)
                actions = torch.cat([actions, cur_action_t], dim=1)
                
                pred_return = target_return_val[0, -1] - (reward * cfg["reward_scale"])
                pred_cost = target_cost_val[0, -1] - (info.get("cost", 0.0) * cfg["cost_scale"])
                
                target_return_val = torch.cat([target_return_val, pred_return.reshape(1, 1)], dim=1)
                target_cost_val = torch.cat([target_cost_val, pred_cost.reshape(1, 1)], dim=1)
                timesteps = torch.cat([timesteps, torch.tensor([[0]], device=device, dtype=torch.long)], dim=1)

                if terminated or truncated:
                    break
                    
    return np.array(embeddings), labels

# --------------------------------------------------------
# Main
# --------------------------------------------------------
@pyrallis.wrap()
def main(args: LatentConfig):
    # 1. Load Setup
    cfg, model_state = custom_load_config_and_model(args.path, device=args.device)
    seed_all(args.seed)
    
    task_name = cfg["task"]
    print(f"Task detected: {task_name}")

    if "Metadrive" in task_name:
        import gym
    else:
        import gymnasium as gym

    env = wrap_env(gym.make(task_name), reward_scale=cfg["reward_scale"])
    env = OfflineEnvWrapper(env)
    
    # 2. Init Model
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
    
    state_dict = model_state["model_state"] if "model_state" in model_state else model_state
    cdt_model.load_state_dict(state_dict)
    cdt_model.to(args.device)
    
    # 3. Collect Embeddings
    print("Extracting embeddings...")
    X, y = get_latent_embeddings(cdt_model, env, args.num_samples, args.device, cfg)
    
    # 4. Run t-SNE
    print("Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(X)
    
    # 5. Determine Output Path based on Task
    save_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cdt_latent_space.png")

    # 6. Plot
    print(f"Plotting to {save_path}...")
    df = pd.DataFrame({
        'x': X_embedded[:, 0], 
        'y': X_embedded[:, 1], 
        'Safety': y
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='x', y='y', hue='Safety', 
        palette={'Safe': 'dodgerblue', 'Unsafe': 'crimson'},
        alpha=0.6, s=50
    )
    
    plt.title(f"Latent Space Visualization: {task_name}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(title="Ground Truth State")
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Done! Saved at: {save_path}")

if __name__ == "__main__":
    main()