import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

@torch.no_grad()
def evaluate_representations(trainer, dataloader, device, step, num_buckets=2):
    trainer.model.eval()
    all_latents = []
    all_ep_costs = []
    
    # Collect a large batch of latents
    for batch in dataloader:
        states, actions, returns, costs_return, time_steps, mask, ep_cost, costs = [b.to(device) for b in batch]
        _, _, _, latents = trainer.model(states, actions, returns, costs_return, time_steps, return_latents=True)
        
        # Mask out padding for latents
        valid_latents = latents[mask > 0]
        
        # Expand the [Batch, 1] episode cost to match the [Batch, Seq_Len] 
        # before masking, so every valid step inherits its trajectory's total cost.
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
    
    # 1. Linear Probing (Handles multi-class natively via OvR)
    clf = LogisticRegression(max_iter=1000).fit(X, y_labels)
    probe_acc = clf.score(X, y_labels)
    
    # 2. Silhouette Score
    sil_score = silhouette_score(X, y_labels) if len(np.unique(y_labels)) > 1 else 0.0
    
    # 3. t-SNE Visualization 
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:2000]) 
    
    plt.figure(figsize=(8, 6))
    cmap = 'coolwarm' if num_buckets == 2 else 'viridis'
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_labels[:2000], cmap=cmap, alpha=0.6)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"Safety Severity (0 to {num_buckets - 1})" if num_buckets > 2 else "Safe (0) vs Unsafe (1)")
    plt.title(f"t-SNE Latent Space at Step {step} ({num_buckets} Buckets)")
    
    wandb.log({
        "eval/linear_probe_acc": probe_acc,
        "eval/silhouette_score": sil_score,
        "eval/latent_space": wandb.Image(plt)
    }, step=step)
    
    plt.close()
    trainer.model.train()