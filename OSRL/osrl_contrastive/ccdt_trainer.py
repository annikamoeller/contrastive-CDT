import torch
import torch.nn.functional as F
import numpy as np
from osrl.algorithms import CDTTrainer
import wandb

class ContrastiveCDTTrainer(CDTTrainer): 
    def __init__(self, model, env, contrastive_weight=0.1, temperature=0.1, 
                 num_buckets=2, max_expected_cost=100.0, **kwargs):
        super().__init__(model, env, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.num_buckets = num_buckets
        self.max_expected_cost = max_expected_cost
        
        # The metric learning standard
        self.ntxent_loss = NTXentLoss(temperature=temperature)

    def train_one_step(self, states, actions, returns, costs_return, time_steps, mask, episode_cost, costs, is_pretraining=False):
        
        # (Removed the explicit pair flattening logic since we use standard batches now)
            
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

        # --- Contrastive loss & Dynamic Bucketing ---
        flat_latents = latents[mask > 0]
        flat_costs = costs[mask > 0]
        
        if self.num_buckets == 2:
            labels = (flat_costs > 0).long() # 0 for safe, 1 for unsafe
        else:
            bin_size = self.max_expected_cost / float(self.num_buckets)
            labels = torch.clamp((flat_costs / bin_size).long(), 0, self.num_buckets - 1)
            
        cont_loss = self.ntxent_loss(flat_latents, labels)

        # Optimization Strategy
        if is_pretraining:
            loss = cont_loss 
        else:
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
            act_loss=act_loss.item() if not is_pretraining else 0.0
        )
        
    @torch.no_grad()
    def evaluate(self, num_rollouts, target_return, target_cost):
        import gymnasium as gym
        import numpy as np
        import wandb
        
        print(f"\n[Eval] Running {num_rollouts} parallel rollouts for target_cost: {target_cost}...")
        self.model.eval()
        
        # ==========================================
        # 1. FAST VECTORIZED EVALUATION (For Metrics)
        # ==========================================
        env_id = self.env.unwrapped.spec.id
        vec_env = gym.vector.make(env_id, num_envs=num_rollouts, asynchronous=True)
        obs, _ = vec_env.reset()
        
        device = self.device
        seq_len = self.model.seq_len
        
        states = torch.zeros((num_rollouts, seq_len, self.model.state_dim), dtype=torch.float32, device=device)
        actions = torch.zeros((num_rollouts, seq_len, self.model.action_dim), dtype=torch.float32, device=device)
        returns = torch.zeros((num_rollouts, seq_len, 1), dtype=torch.float32, device=device)
        costs = torch.zeros((num_rollouts, seq_len, 1), dtype=torch.float32, device=device)
        time_steps = torch.zeros((num_rollouts, seq_len), dtype=torch.long, device=device)
        
        episode_returns = np.zeros(num_rollouts)
        episode_costs = np.zeros(num_rollouts)
        episode_lengths = np.zeros(num_rollouts)
        active_envs = np.ones(num_rollouts, dtype=bool)
        
        target_return_tensor = torch.tensor([target_return] * num_rollouts, device=device, dtype=torch.float32)
        target_cost_tensor = torch.tensor([target_cost] * num_rollouts, device=device, dtype=torch.float32)
        ep_costs_tensor = torch.zeros((num_rollouts, 1), device=device, dtype=torch.float32)
        
        t = 0
        while active_envs.any():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            states[:, t % seq_len] = obs_tensor
            time_steps[:, t % seq_len] = t
            returns[:, t % seq_len] = target_return_tensor.unsqueeze(-1)
            costs[:, t % seq_len] = target_cost_tensor.unsqueeze(-1)
            
            if t < seq_len:
                seq_s, seq_a = states[:, :t+1], actions[:, :t+1]
                seq_r, seq_c, seq_t = returns[:, :t+1], costs[:, :t+1], time_steps[:, :t+1]
            else:
                idx = torch.arange(t - seq_len + 1, t + 1) % seq_len
                seq_s, seq_a = states[:, idx], actions[:, idx]
                seq_r, seq_c, seq_t = returns[:, idx], costs[:, idx], time_steps[:, idx]
                
            action_preds, _, _ = self.model(
                states=seq_s, actions=seq_a, returns_to_go=seq_r, costs_to_go=seq_c, 
                time_steps=seq_t, episode_cost=ep_costs_tensor
            )
            
            current_actions = action_preds[:, -1]
            if self.stochastic:
                current_actions = current_actions.mean 
            
            actions[:, t % seq_len] = current_actions
            cpu_actions = current_actions.cpu().numpy()
            
            obs, rews, terminateds, truncateds, infos = vec_env.step(cpu_actions)
            dones = terminateds | truncateds
            
            for i in range(num_rollouts):
                if active_envs[i]:
                    episode_returns[i] += rews[i]
                    step_cost = infos["cost"][i] if "cost" in infos else 0.0
                    episode_costs[i] += step_cost
                    episode_lengths[i] += 1
                    
                    target_return_tensor[i] -= rews[i]
                    target_cost_tensor[i] -= step_cost
                    ep_costs_tensor[i] += step_cost
                    if dones[i]: active_envs[i] = False
            t += 1
            
        vec_env.close()
        
        avg_return = np.mean(episode_returns)
        avg_cost = np.mean(episode_costs)
        avg_length = np.mean(episode_lengths)
        print(f"[Eval] Results: Reward={avg_return:.2f}, Cost={avg_cost:.2f}, Avg Len={avg_length:.1f}")

        # ==========================================
        # 2. THESIS VIDEO RECORDING (Single Env)
        # ==========================================
        if wandb.run is not None:
            try:
                print(f"🎥 Recording video for target_cost {target_cost}...")
                frames = []
                original_step = self.env.step
                
                def step_with_video(action):
                    step_obs, step_rew, step_term, step_trunc, step_info = original_step(action)
                    frame = self.env.unwrapped.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(frame)
                    return step_obs, step_rew, step_term, step_trunc, step_info
                
                self.env.step = step_with_video
                
                self.env.reset()
                first_frame = self.env.unwrapped.render(mode="rgb_array")
                if first_frame is not None:
                    frames.append(first_frame)
                
                # Do 1 visual rollout using the parent class method
                _, vid_len, _ = self.rollout(self.model, self.env, target_return, target_cost)
                
                self.env.step = original_step
                
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

        self.model.train()
        return avg_return, avg_cost, avg_length