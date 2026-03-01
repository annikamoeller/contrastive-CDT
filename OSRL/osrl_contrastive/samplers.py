import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm.auto import tqdm

class PairedContrastiveDataset(Dataset):
    """
    Wraps a standard SequenceDataset. For every item, it actively samples a 
    second 'positive' item from the same safety/cost bin.
    """
    def __init__(self, base_dataset, discretization="binary", cost_scale=1.0):
        self.base_dataset = base_dataset
        self.discretization = discretization
        self.cost_scale = cost_scale
        
        # Dictionary to hold indices belonging to each bin
        self.bins = defaultdict(list)
        self._precompute_bins()

    def _get_bin(self, idx):
        # In OSRL's SequenceDataset, __getitem__ returns:
        # states, actions, returns, costs_return, time_steps, mask, episode_cost, costs
        # Episode cost is index 6. We extract it to determine safety.
        item = self.base_dataset[idx]
        ep_cost = item[6][0].item() 
        
        if self.discretization == "binary":
            # 0 = strictly safe, 1 = unsafe
            return 0 if ep_cost == 0 else 1
        elif self.discretization == "granular":
            # Groups costs into chunks (e.g., 0-0.1, 0.1-0.2, etc.)
            return int(ep_cost // (0.1 * self.cost_scale))
        else:
            raise ValueError(f"Unknown discretization: {self.discretization}")

    def _precompute_bins(self):
        print(f"Precomputing {self.discretization} safety bins for explicit pairing...")
        for i in tqdm(range(len(self.base_dataset)), desc="Binning dataset"):
            bin_id = self._get_bin(i)
            self.bins[bin_id].append(i)
            
        # Ensure we only use indices where there is at least one other positive match
        self.valid_indices = [
            i for i in range(len(self.base_dataset)) 
            if len(self.bins[self._get_bin(i)]) > 1
        ]
        print(f"Dataset mapped. Found {len(self.bins)} unique cost bins.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. Get the Anchor sequence
        real_idx = self.valid_indices[idx]
        anchor = self.base_dataset[real_idx]
        
        # 2. Find a Positive sequence from the same bin
        bin_id = self._get_bin(real_idx)
        positive_idx = np.random.choice(self.bins[bin_id])
        
        # Ensure we don't pick the exact same sequence
        while positive_idx == real_idx:
            positive_idx = np.random.choice(self.bins[bin_id])
            
        positive = self.base_dataset[positive_idx]
        
        # 3. Stack them. If anchor has 8 tensors, this returns 8 tensors, 
        # where each tensor has shape [2, seq_len, ...]
        stacked_tensors = tuple(torch.stack([a, p], dim=0) for a, p in zip(anchor, positive))
        
        return stacked_tensors