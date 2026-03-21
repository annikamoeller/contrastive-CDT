import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
# import h5py # Uncomment if your datasets end in .hdf5

def analyze_dataset(file_path, task_name):
    print(f"Loading local dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find dataset at {file_path}. Check your path!")

    # 1. Direct File Loading
    if file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
    elif file_path.endswith('.hdf5') or file_path.endswith('.h5'):
        import h5py
        dataset = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                dataset[key] = f[key][()]
    else:
        raise ValueError("Unsupported file format. Need .pkl or .hdf5")

    # 2. Slice into trajectories
    # (Checking for both standard D4RL naming 'terminals' and custom 'dones')
    terminals_key = "terminals" if "terminals" in dataset else "dones"
    dones = np.logical_or(dataset[terminals_key], dataset["timeouts"])
    done_idx = np.where(dones)[0]
    
    trajectories = []
    start_idx = 0
    for end_idx in done_idx:
        end_idx += 1 
        traj = {
            "rewards": dataset["rewards"][start_idx:end_idx],
            "costs": dataset["costs"][start_idx:end_idx],
            "length": end_idx - start_idx
        }
        trajectories.append(traj)
        start_idx = end_idx

    num_trajs = len(trajectories)
    total_transitions = len(dataset["rewards"])
    
    print(f"--- DATASET STATS: {task_name} ---")
    print(f"Total Transitions: {total_transitions}")
    print(f"Total Trajectories: {num_trajs}")
    
    returns = np.array([np.sum(t["rewards"]) for t in trajectories])
    costs = np.array([np.sum(t["costs"]) for t in trajectories])
    lengths = np.array([t["length"] for t in trajectories])
    
    print(f"Returns -> Mean: {np.mean(returns):.1f}, Max: {np.max(returns):.1f}")
    print(f"Costs   -> Mean: {np.mean(costs):.1f}, Max: {np.max(costs):.1f}, Median: {np.median(costs):.1f}")
    print(f"Lengths -> Mean: {np.mean(lengths):.1f}, Max: {np.max(lengths)}")

    # 3. Setup Plotting
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Dataset Analysis: {task_name}", fontsize=18, weight='bold')

    # Plot A: Returns & Costs
    ax1 = plt.subplot(2, 2, 1)
    sns.histplot(returns, color="blue", alpha=0.5, label="Returns", kde=True, ax=ax1)
    sns.histplot(costs, color="red", alpha=0.5, label="Costs", kde=True, ax=ax1)
    ax1.set_title("Distribution of Episode Returns and Costs")
    ax1.legend()

    # Plot B: Pareto Scatter
    ax2 = plt.subplot(2, 2, 2)
    sns.scatterplot(x=costs, y=returns, alpha=0.6, color="purple", ax=ax2)
    ax2.set_title("Pareto Distribution (Return vs. Cost)")
    ax2.set_xlabel("Total Episode Cost")
    ax2.set_ylabel("Total Episode Return")

    # Plot C: Trajectory Lengths
    ax3 = plt.subplot(2, 2, 3)
    sns.histplot(lengths, bins=30, color="green", kde=False, ax=ax3)
    ax3.set_title("Trajectory Lengths")
    ax3.set_xlabel("Steps per Episode")

    # Plot D: Cumulative Accumulation
    ax4 = plt.subplot(2, 2, 4)
    max_len = np.max(lengths)
    
    cum_rewards = np.full((num_trajs, max_len), np.nan)
    cum_costs = np.full((num_trajs, max_len), np.nan)
    
    for i, t in enumerate(trajectories):
        cum_rewards[i, :t["length"]] = np.cumsum(t["rewards"])
        cum_costs[i, :t["length"]] = np.cumsum(t["costs"])
        
    mean_cum_rew = np.nanmean(cum_rewards, axis=0)
    mean_cum_cost = np.nanmean(cum_costs, axis=0)
    
    steps = np.arange(max_len)
    ax4.plot(steps, mean_cum_rew, label="Avg Cumulative Reward", color="blue", linewidth=2)
    ax4.plot(steps, mean_cum_cost, label="Avg Cumulative Cost", color="red", linewidth=2)
    ax4.set_title("Average Accumulation over Time")
    ax4.set_xlabel("Timestep")
    ax4.set_ylabel("Cumulative Value")
    ax4.legend()

    plt.tight_layout()
    save_name = f"dataset_analysis/dataset_analysis_{task_name}.png"
    plt.savefig(save_name, dpi=300)
    print(f"\n✅ Plot saved successfully as {save_name}!")

if __name__ == "__main__":    
    analyze_dataset("datasets/SafetyAntRun-v0-150-1816.hdf5", "AntRun")
    analyze_dataset("datasets/SafetyCarCircle-v0-100-1450.hdf5", "CarCircle")
    analyze_dataset("datasets/SafetyCarGoal1Gymnasium-v0-120-1671.hdf5", "CarGoal")