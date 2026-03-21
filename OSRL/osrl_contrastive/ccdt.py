import torch
import torch.nn as nn
from osrl.common.net import DiagGaussianActor, TransformerBlock, mlp
from osrl.algorithms import CDT
import torch.nn.functional as F

class ContrastiveCDT(CDT): 
    def __init__(self, contrastive_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_dim = contrastive_dim
        
        # SimCLR-style deep projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, contrastive_dim)
        )

    def get_latents(self, state_emb, action_emb, cost_emb):
        # "Product" mechanism: condition the representation on the cost budget
        state_conditioned = state_emb * cost_emb
        action_conditioned = action_emb * cost_emb
        
        # Combine and project
        s_a_stack = state_conditioned + action_conditioned
        return self.contrastive_head(s_a_stack)

    def forward(self, states, actions, returns_to_go, costs_to_go, time_steps, padding_mask=None, episode_cost=None, return_latents=False):
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Helper to align targets (R, C) to the current sequence length
        def align(x):
            if x.ndim == 1: x = x.unsqueeze(-1) # [B] -> [B, 1]
            if x.ndim == 2: x = x.unsqueeze(-1) # [B, T] -> [B, T, 1]
            # If it's [B, 1, 1], broadcast the time dimension to match states [B, T, 1]
            if x.shape[1] < seq_len:
                x = x.expand(-1, seq_len, -1)
            return x

        # 1. Embeddings
        timestep_emb = self.timestep_emb(torch.clamp(time_steps, 0, self.timestep_emb.num_embeddings - 1)) if self.time_emb else 0.0        
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        # 2. Build the token list dynamically
        seq_list = []
        if self.use_rew:
            seq_list.append(self.return_emb(align(returns_to_go)) + timestep_emb)
        if self.use_cost:
            if self.cost_transform: costs_to_go = self.cost_transform(costs_to_go.detach())
            costs_emb = self.cost_emb(align(costs_to_go)) + timestep_emb
            seq_list.append(costs_emb)
        
        seq_list.extend([state_emb, act_emb])
        # --- DEBUG 1: Sequence Composition ---
        if return_latents and torch.rand(1) < 0.001:
            print(f"\n[MODEL DEBUG] Batch size: {batch_size}, Seq Len: {seq_len}")
            print(f"Tokens in seq_list: {len(seq_list)} (Expected: {self.seq_repeat})")
            print(f"Costs_to_go stats: Mean={costs_to_go.mean().item():.2f}, Max={costs_to_go.max().item():.2f}")
            
        # 3. Interleave tokens: (R, C, S, A)
        # Result: [batch, seq_len * num_tokens, embedding_dim]
        sequence = torch.stack(seq_list, dim=2).reshape(batch_size, -1, self.embedding_dim)

        # 4. Padding & Transformer Pass
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, self.seq_repeat).reshape(batch_size, -1)

        if self.cost_prefix:
            prefix = self.prefix_emb(episode_cost.to(states.dtype).view(batch_size, 1, 1))
            sequence = torch.cat([prefix, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        out = self.emb_drop(self.emb_norm(sequence))
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
        out = self.out_norm(out)

        # 5. Extract Features for Prediction
        if self.cost_prefix: out = out[:, 1:]
        if torch.isnan(out).any():
            print("❌ [MODEL DEBUG] CRITICAL: NaNs detected after Transformer blocks!")
            
        # Reshape back to [B, T, num_tokens, D]
        out = out.view(batch_size, seq_len, self.seq_repeat, self.embedding_dim)
        state_feat = out[:, :, -2] # State is always second to last
        action_feat = out[:, :, -1] # Action is always last

        # Prediction Heads
        action_preds = self.action_head(state_feat)
        cost_preds = F.log_softmax(self.cost_pred_head(action_feat), dim=-1)
        state_preds = self.state_pred_head(action_feat)
        
        if return_latents and self.use_cost:
            return action_preds, cost_preds, state_preds, self.get_latents(state_emb, act_emb, costs_emb)
        
        return action_preds, cost_preds, state_preds