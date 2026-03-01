import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

@torch.no_grad()
def evaluate_representations(trainer, dataloader, device, step):
    trainer.model.eval()
    all_latents = []
    all_costs = []
    
    # Collect a large batch of latents
    for batch in dataloader:
        states, actions, returns, costs_return, time_steps, mask, ep_cost, costs = [b.to(device) for b in batch]
        _, _, _, latents = trainer.model(states, actions, returns, costs_return, time_steps, return_latents=True)
        
        # Mask out padding
        valid_latents = latents[mask > 0]
        valid_costs = costs_return[mask > 0]
        
        all_latents.append(valid_latents.cpu().numpy())
        all_costs.append(valid_costs.cpu().numpy())
        
        if len(all_latents) > 10: # Just need a few thousand points
            break
            
    X = np.concatenate(all_latents, axis=0)
    costs = np.concatenate(all_costs, axis=0)
    y_binary = (costs > 0).astype(int) # 0 for Safe, 1 for Unsafe

    # 1. Linear Probing
    clf = LogisticRegression(max_iter=1000).fit(X, y_binary)
    probe_acc = clf.score(X, y_binary)
    
    # 2. Silhouette Score (Clustering tightness)
    sil_score = silhouette_score(X, y_binary) if len(np.unique(y_binary)) > 1 else 0.0
    
    # 3. t-SNE Visualization 
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X[:2000]) # Limit to 2000 points for speed
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_binary[:2000], cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label="Unsafe (1) vs Safe (0)")
    plt.title(f"t-SNE Latent Space at Step {step}")
    
    # Log to WandB
    wandb.log({
        "eval/linear_probe_acc": probe_acc,
        "eval/silhouette_score": sil_score,
        "eval/latent_space": wandb.Image(plt)
    }, step=step)
    
    plt.close()
    trainer.model.train()