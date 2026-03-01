import json
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------
# 1. List of paths to your result.json files
# --------------------------------------------
exp_paths = [
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0-100-2022/TraC/cost-40.0/seed-0-seg/results.json",
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_20_60/TraC/cost-40.0/seed-0-seg/results.json",
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_30_50/TraC/cost-40.0/seed-0-seg/results.json",
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_cluster_35_45/TraC/cost-40.0/seed-0-seg/results.json",
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_small_10/TraC/cost-40.0/seed-0-seg/results.json",
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_small_5/TraC/cost-40.0/seed-0-seg/results.json",
    "exp_safetygym/OfflinePointGoal1Gymnasium-v0/SafetyPointGoal1Gymnasium-v0_small_25/TraC/cost-40.0/seed-0-seg/results.json",
]

# --------------------------------------------
# 2. Load all JSONs
# --------------------------------------------
labels = []
normalized_rewards = []
normalized_reward_stds = []
normalized_costs = []
normalized_cost_stds = []

for path in exp_paths:
    with open(path, "r") as f:
        data = json.load(f)

    # label = dataset name
    label = str(path.split("/")[-5])
    labels.append(label)

    # Force all values to plain Python floats
    normalized_rewards.append(float(data["normalized_reward"]))
    normalized_reward_stds.append(float(data["normalized_reward_std"]))
    normalized_costs.append(float(data["normalized_cost"]))
    normalized_cost_stds.append(float(data["normalized_cost_std"]))
print(normalized_rewards)
print(normalized_reward_stds)
print(normalized_costs)
print(normalized_cost_stds)
labels = np.array(labels)
x = np.arange(len(labels))

# --------------------------------------------
# 3. Plot grouped bar chart
# --------------------------------------------
width = 0.35
fig, ax = plt.subplots(figsize=(14, 6))

# Reward bars
ax.bar(
    x - width/2,
    normalized_rewards,
    width,
    yerr=normalized_reward_stds,
    label="Normalized Reward",
    color="tab:green",
    capsize=5,
)

# Cost bars
ax.bar(
    x + width/2,
    normalized_costs,
    width,
    yerr=normalized_cost_stds,
    label="Normalized Cost",
    color="tab:red",
    capsize=5,
)

# Styling
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Normalized Score")
ax.set_title("TraC Evaluation Across Dataset Variants")
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
print("fine")

print("---- DEBUG VALUES ----")
print("Labels:", labels)
print("Label types:", [type(l) for l in labels])

print("Rewards:", normalized_rewards)
print("Reward std:", normalized_reward_stds)
print("Costs:", normalized_costs)
print("Cost std:", normalized_cost_stds)

print("Nan in arrays:")
print(np.isnan(normalized_rewards).any(),
      np.isnan(normalized_reward_stds).any(),
      np.isnan(normalized_costs).any(),
      np.isnan(normalized_cost_stds).any())

print("Inf in arrays:")
print(np.isinf(normalized_rewards).any(),
      np.isinf(normalized_reward_stds).any(),
      np.isinf(normalized_costs).any(),
      np.isinf(normalized_cost_stds).any())

# --------------------------------------------
# SAVE TO FILE
# --------------------------------------------
output_file = "trac_eval_comparison.png"
plt.savefig(output_file)
print(f"Plot saved to {output_file}")
