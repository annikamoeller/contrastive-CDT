import torch
import torch.nn.functional as F
import numpy as np
from osrl.algorithms import CDTTrainer
import wandb

class ContrastiveCDTTrainer(CDTTrainer): # Inherit to keep evaluate() and rollout()
    def __init__(self, model, env, contrastive_weight=0.1, temperature=0.1, 
                 discretization="binary", **kwargs):
        super().__init__(model, env, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.discretization = discretization # "binary" or "granular"

    def compute_contrastive_loss(self, latents, costs, mask):
        flat_latents = F.normalize(latents[mask > 0], dim=1)
        flat_costs = costs[mask > 0]
        
        sim_matrix = torch.matmul(flat_latents, flat_latents.T) / self.temperature
        
        # --- THESIS AXIS: Discretization Strategy ---
        if self.discretization == "binary":
            # Strict Safe (0) vs Unsafe (>0)
            safety_labels = (flat_costs == 0).float()
            pos_mask = torch.eq(safety_labels.unsqueeze(0), safety_labels.unsqueeze(1)).float()
        elif self.discretization == "granular":
            # Binned by strict cost similarity (e.g. costs within 0.1 of each other)
            cost_diff = torch.abs(flat_costs.unsqueeze(0) - flat_costs.unsqueeze(1))
            pos_mask = (cost_diff < (0.1 * self.cost_scale)).float()
        else:
            raise ValueError("Unknown discretization strategy")

        # InfoNCE Math
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        mask_self = torch.eye(logits.shape[0], device=self.device)
        
        exp_logits = torch.exp(logits)
        denom = (exp_logits * (1 - mask_self)).sum(1, keepdim=True)
        log_prob = logits - torch.log(denom + 1e-6)
        
        pos_pairs_mask = pos_mask * (1 - mask_self)
        loss = - (pos_pairs_mask * log_prob).sum(1) / (pos_pairs_mask.sum(1) + 1e-6)
        
        # Track metrics for WandB
        with torch.no_grad():
            pos_dist = (sim_matrix * pos_pairs_mask).sum() / pos_pairs_mask.sum()
            neg_dist = (sim_matrix * (1 - pos_mask)).sum() / (1 - pos_mask).sum()

        return loss.mean(), pos_dist.item(), neg_dist.item()

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask, episode_cost, costs, is_pretraining=False):
        
        # Flatten explicit pairs if they exist
        if states.dim() == 4: # Shape: [Batch, 2, seq_len, dim]
            B, num_pairs, seq_len = states.shape[0], states.shape[1], states.shape[2]
            states = states.view(-1, seq_len, states.shape[-1])
            actions = actions.view(-1, seq_len, actions.shape[-1])
            returns = returns.view(-1, seq_len)
            costs_return = costs_return.view(-1, seq_len)
            time_steps = time_steps.view(-1, seq_len)
            mask = mask.view(-1, seq_len)
            episode_cost = episode_cost.view(-1)
            costs = costs.view(-1, seq_len)
            
        padding_mask = ~mask.to(torch.bool)
        
        action_preds, cost_preds, state_preds, latents = self.model(
            states, actions, returns, costs_return, time_steps, padding_mask, episode_cost, return_latents=True
        )

        # Standard losses
        if self.stochastic:
            log_likelihood = action_preds.log_prob(actions)[mask > 0].mean()
            entropy = action_preds.entropy()[mask > 0].mean()
            entropy_reg = self.model.temperature().detach()
            if self.no_entropy:
                entropy_reg = 0.0
            act_loss = -(log_likelihood + entropy_reg * entropy)
        else:
            act_loss = F.mse_loss(action_preds, actions.detach(), reduction="none")
            act_loss = (act_loss * mask.unsqueeze(-1)).mean()

        cost_preds_flat = cost_preds.reshape(-1, 2)
        costs_flat = costs.flatten().long().detach()
        cost_loss = F.nll_loss(cost_preds_flat, costs_flat, reduction="none")
        cost_loss = (cost_loss * mask.flatten()).mean()

        state_loss = F.mse_loss(state_preds[:, :-1], states[:, 1:].detach(), reduction="none")
        state_loss = (state_loss * mask[:, :-1].unsqueeze(-1)).mean()

        # Contrastive loss
        cont_loss, pos_dist, neg_dist = self.compute_contrastive_loss(latents, costs_return, mask)

        # --- THESIS AXIS: Optimization Strategy ---
        if is_pretraining:
            loss = cont_loss # Only train the representation
        else:
            # Joint training
            loss = act_loss + self.cost_weight * cost_loss + self.state_weight * state_loss + self.contrastive_weight * cont_loss

        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optim.step()
        self.scheduler.step()

        # Log to WandB
        self.logger.store(
            tab="train",
            total_loss=loss.item(),
            cont_loss=cont_loss.item(),
            pos_sim=pos_dist,  # Should go UP
            neg_sim=neg_dist,  # Should go DOWN
        )
        
    def evaluate(self, num_rollouts, target_return, target_cost):
        # 1. Official metric evaluation
        print(f"\n[Eval] Calculating metrics for target_cost: {target_cost}...")
        ret, cost, length = super().evaluate(num_rollouts, target_return, target_cost)
        print(f"[Eval] Results: Reward={ret:.2f}, Cost={cost:.2f}, Avg Len={length:.1f}")

        # 2. Recording the thesis video
        if wandb.run is not None:
            try:
                print(f"🎥 Recording video for target_cost {target_cost}...")
                frames = []
                original_step = self.env.step
                
                def step_with_video(action):
                    obs, reward, terminated, truncated, info = original_step(action)
                    frame = self.env.unwrapped.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(frame)
                    return obs, reward, terminated, truncated, info
                
                self.env.step = step_with_video
                
                # Setup environment
                self.env.reset()
                frame = self.env.unwrapped.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)
                
                # Execute rollout
                self.model.eval()
                # self.rollout returns the results of this specific video run
                _, vid_len, _ = self.rollout(self.model, self.env, target_return, target_cost)
                self.model.train()
                
                # Restore environment
                self.env.step = original_step
                
                # 3. Save and Upload
                if len(frames) > 0:
                    print(f"📦 Processing {len(frames)} frames into MP4 (Length: {vid_len})...")
                    video_array = np.array(frames)
                    video_array = np.transpose(video_array, (0, 3, 1, 2))
                    
                    wandb.log({
                        f"eval_video/c_{target_cost}_r_{target_return}": wandb.Video(video_array, fps=30, format="mp4")
                    }, commit=False)
                    print(f"✅ Video successfully queued for WandB.")
                else:
                    print(f"⚠️ Warning: No frames were captured during rollout.")
                    
            except Exception as e:
                print(f"\n❌ [Warning] Video logging failed: {e}")
                self.env.step = original_step 

        return ret, cost, length