import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

@torch.no_grad()
def evaluate_representations(trainer, dataloader, device, step, num_buckets=2, save_dir=None):
    trainer.model.eval()
    all_latents = []
    all_ep_costs = []
    
    # Identify architecture dynamically for clean logging
    model_classname = type(trainer.model).__name__
    arch_label = "Front-Encoder" if "Front" in model_classname else "Back-Encoder"
    
    # Collect a large batch of latents
    for batch in dataloader:
        states, actions, returns, costs_return, time_steps, mask, ep_cost, costs = [b.to(device) for b in batch]
        
        padding_mask = ~mask.to(torch.bool)
        
        # Pass ALL arguments to the model (automatically returns the correct arch latents)
        _, _, _, latents = trainer.model(
            states=states, 
            actions=actions, 
            returns_to_go=returns, 
            costs_to_go=costs_return, 
            time_steps=time_steps, 
            padding_mask=padding_mask,  
            episode_cost=ep_cost,       
            return_latents=True
        )
        
        valid_latents = latents[mask > 0]
        expanded_ep_cost = ep_cost.unsqueeze(1).expand(-1, latents.shape[1])
        valid_ep_costs = expanded_ep_cost[mask > 0]
    
        all_latents.append(valid_latents.cpu().numpy())
        all_ep_costs.append(valid_ep_costs.cpu().numpy())
        
        if len(all_latents) > 10: 
            break
            
    X = np.concatenate(all_latents, axis=0)
    ep_costs = np.concatenate(all_ep_costs, axis=0)

    boundaries = trainer.cost_boundaries.cpu().numpy()
    y_labels = np.digitize(ep_costs, boundaries)
    
    # 1. Linear Probing 
    clf = LogisticRegression(max_iter=1000).fit(X, y_labels)
    probe_acc = clf.score(X, y_labels)
    
    # 2. Silhouette Score
    sil_score = silhouette_score(X, y_labels) if len(np.unique(y_labels)) > 1 else 0.0
    
    # 3. t-SNE Visualization with Safety Guard for Slicing
    slice_idx = min(2000, len(X)) # Prevents out-of-bounds crashes if tokens < 2000
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:slice_idx]) 
    
    plt.figure(figsize=(8, 6))
    cmap = 'coolwarm' if num_buckets == 2 else 'viridis'
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_labels[:slice_idx], cmap=cmap, alpha=0.6)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"Safety Severity (0 to {num_buckets - 1})" if num_buckets > 2 else "Safe (0) vs Unsafe (1)")
    plt.title(f"{arch_label} Latent Space | Step {step} ({num_buckets} Buckets)")
    
    # --- DUAL SAVE STRATEGY (W&B Cloud + Local Disk) ---
    if wandb.run is not None:
        wandb.log({
            "eval/linear_probe_acc": probe_acc,
            "eval/silhouette_score": sil_score,
            "eval/latent_space": wandb.Image(plt)
        }, step=step)
        
        # Save raw arrays to W&B cloud files
        wandb_save_path = os.path.join(wandb.run.dir, f"tsne_arrays_step_{step}.npz")
        np.savez(wandb_save_path, tsne_x=X_tsne[:, 0], tsne_y=X_tsne[:, 1], raw_costs=ep_costs[:slice_idx], labels=y_labels[:slice_idx])

    if save_dir is not None:
        # Save raw arrays locally to your output/ directory
        local_save_path = os.path.join(save_dir, f"tsne_arrays_step_{step}.npz")
        np.savez(local_save_path, tsne_x=X_tsne[:, 0], tsne_y=X_tsne[:, 1], raw_costs=ep_costs[:slice_idx], labels=y_labels[:slice_idx])
    
    plt.close()
    trainer.model.train()