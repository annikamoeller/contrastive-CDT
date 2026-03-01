import argparse
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from research.utils.config import Config

# Global storage
latent_storage = []

def get_activation_hook():
    def hook(model, input, output):
        if isinstance(output, tuple): output = output[0]
        latent_storage.clear()
        latent_storage.append(output.detach().cpu().numpy())
    return hook

def collect_embeddings(env, model, num_samples):
    embeddings, labels = [], []
    print(f"Collecting {num_samples} samples...")
    
    obs, _ = env.reset()
    collected = 0
    
    while collected < num_samples:
        with torch.no_grad():
            action = model.predict(dict(obs=obs))
        
        if latent_storage:
            embeddings.append(latent_storage[0].flatten())

        next_obs, _, terminal, timeout, info = env.step(action)
        
        labels.append("Unsafe" if info.get("cost", 40.0) > 0 else "Safe")
        collected += 1
        
        # --- FIX: Standard If/Else to handle unpacking correctly ---
        if terminal or timeout:
            obs, _ = env.reset()
        else:
            obs = next_obs # next_obs is already just the observation

    return np.array(embeddings), labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--save_plot", type=str, default="trac_latent_space.png")
    parser.add_argument("--num_samples", type=int, default=2000)
    args = parser.parse_args()

    # 1. Setup
    config = Config.load(os.path.join(args.exp_path, "config.yaml")).parse()
    trainer = config.get_trainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load Weights (Directly)
    model_path = os.path.join(args.exp_path, "best_model.pt")
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Load into trainer.model.network (TraC specific)
    trainer.model.network.load_state_dict(checkpoint['network'])
    trainer.model.network.to(device)
    
    print(f"✅ Loaded {model_path}")

    # 3. Register Hook
    target_layer = [m for m in trainer.model.network.modules() if isinstance(m, (torch.nn.Tanh, torch.nn.ReLU))][-1]
    target_layer.register_forward_hook(get_activation_hook())

    # 4. Run
    trainer.model.eval()
    env = trainer.eval_env 
    
    X, y = collect_embeddings(env, trainer.model, args.num_samples)
    
    # 5. Plot
    print("Plotting...")
    X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(X)
    df = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Safety': y})
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='Safety', palette={'Safe': 'dodgerblue', 'Unsafe': 'crimson'})
    plt.savefig(os.path.join(args.exp_path, args.save_plot))
    print("Done.")

if __name__ == "__main__":
    main()