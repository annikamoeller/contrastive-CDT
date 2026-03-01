import torch
import torch.nn as nn
from osrl.common.net import DiagGaussianActor, TransformerBlock, mlp
from osrl.algorithms import CDT
import torch.nn.functional as F

class ContrastiveCDT(CDT): # Inherit from your original CDT
    def __init__(self, contrastive_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.contrastive_dim = contrastive_dim
        
        # The projection head maps the transformer's output to the latent space
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, contrastive_dim)
        )

    def forward(self, states, actions, returns_to_go, costs_to_go, time_steps, padding_mask=None, episode_cost=None, return_latents=False):
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        timestep_emb = self.timestep_emb(torch.clamp(time_steps, 0, self.timestep_emb.num_embeddings - 1)) if self.time_emb else 0.0        
        state_emb = self.state_emb(states) + timestep_emb
        act_emb = self.action_emb(actions) + timestep_emb

        seq_list = [state_emb, act_emb]
        
        if self.cost_transform is not None:
            costs_to_go = self.cost_transform(costs_to_go.detach())

        if self.use_cost:
            costs_emb = self.cost_emb(costs_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, costs_emb)
        if self.use_rew:
            returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + timestep_emb
            seq_list.insert(0, returns_emb)

        sequence = torch.stack(seq_list, dim=1).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size, self.seq_repeat * seq_len, self.embedding_dim)

        if padding_mask is not None:
            padding_mask = torch.stack([padding_mask] * self.seq_repeat, dim=1).permute(0, 2, 1).reshape(batch_size, -1)

        if self.cost_prefix:
            episode_cost = episode_cost.to(states.dtype).unsqueeze(-1).unsqueeze(-1)
            episode_cost_emb = self.prefix_emb(episode_cost)
            sequence = torch.cat([episode_cost_emb, sequence], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat([padding_mask[:, :1], padding_mask], dim=1)

        out = self.emb_drop(self.emb_norm(sequence))

        # --- SINGLE PASS THROUGH TRANSFORMER ---
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        if self.cost_prefix:
            out = out[:, 1:]

        out = out.reshape(batch_size, seq_len, self.seq_repeat, self.embedding_dim).permute(0, 2, 1, 3)

        action_feature = out[:, self.seq_repeat - 1]
        state_feat = out[:, self.seq_repeat - 2]

        if self.add_cost_feat and self.use_cost:
            state_feat = state_feat + costs_emb.detach()
        if self.mul_cost_feat and self.use_cost:
            state_feat = state_feat * costs_emb.detach()
        if self.cat_cost_feat and self.use_cost:
            state_feat = torch.cat([state_feat, costs_emb.detach()], dim=2)

        action_preds = self.action_head(state_feat)
        cost_preds = F.log_softmax(self.cost_pred_head(action_feature), dim=-1)
        state_preds = self.state_pred_head(action_feature)
        
        # --- GET LATENTS ---
        latents = self.contrastive_head(action_feature)
        
        if return_latents:
            latents = self.contrastive_head(action_feature)
            return action_preds, cost_preds, state_preds, latents
        else:
            return action_preds, cost_preds, state_preds