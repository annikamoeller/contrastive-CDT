import os
import types
from dataclasses import asdict, dataclass
from typing import Tuple

import bullet_safety_gym  # noqa
import dsrl
import gymnasium as gym  # noqa
import numpy as np
import pyrallis
import torch
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env  # noqa
from fsrl.utils import WandbLogger
from torch.utils.data import DataLoader
from tqdm.auto import trange  # noqa

from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG, CDTTrainConfig
from osrl.common import SequenceDataset
from osrl.common.exp_util import auto_name, seed_all

# --- IMPORT NEW CONTRASTIVE COMPONENTS ---
from osrl_contrastive.ccdt import ContrastiveCDT
from osrl_contrastive.ccdt_trainer import ContrastiveCDTTrainer
from osrl_contrastive.samplers import PairedContrastiveDataset
from osrl_contrastive.probe_and_vis import evaluate_representations

import gymnasium as gym

class RobustEnvWrapper(gym.Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 6:
            # Bullet Safety Gym: (obs, reward, cost, term, trunc, info)
            obs, rew, cost, term, trunc, info = result
            if isinstance(info, dict):
                info["cost"] = cost
            return obs, rew, term, trunc, info
        elif len(result) == 4:
            # Old Gym: (obs, reward, done, info)
            obs, rew, done, info = result
            return obs, rew, done, False, info
        return result # Assume it's the standard 5

@dataclass
class ContrastiveCDTTrainConfig(CDTTrainConfig):
    # New thesis arguments for ablation studies
    contrastive_dim: int = 64
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    discretization_type: str = "binary"  # Options: "binary" or "granular"
    use_explicit_pairs: bool = False     # Toggle between custom DataLoader and in-batch masking
    pretrain_steps: int = 0              # If > 0, pre-trains latents before joint training
    probe_every: int = 5000              # How often to run t-SNE and linear probing


@pyrallis.wrap()
def train(args: ContrastiveCDTTrainConfig):
    # update config
    cfg, old_cfg = asdict(args), asdict(ContrastiveCDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    # Inherit defaults from the base CDT task configs if available
    if args.task in CDT_DEFAULT_CONFIG:
        base_cfg = asdict(CDT_DEFAULT_CONFIG[args.task]())
    else:
        print(f"⚠️ Warning: '{args.task}' not found in CDT_DEFAULT_CONFIG.")
        print("Falling back to base CDTTrainConfig defaults.")
        base_cfg = asdict(CDTTrainConfig())
        
    cfg.update(base_cfg)
    cfg.update(differing_values)
    args = types.SimpleNamespace(**cfg)

    # setup logger
    if args.name is None:
        args.name = auto_name(base_cfg, cfg, args.prefix, args.suffix)
    if args.group is None:
        args.group = f"{args.task}-cost-{int(args.cost_limit)}-contrastive"
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
        
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    # set seed & device
    seed_all(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    # initialize environment
    env = gym.make(args.task, disable_env_checker=True)
    
    # 2. Wrap it in our universal adapter
    env = RobustEnvWrapper(env)
    
    # pre-process offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins, rbins = density_cfg["cbins"], density_cfg["rbins"]
        max_npb, min_npb = density_cfg["max_npb"], density_cfg["min_npb"]
        
    data = env.pre_process_data(data, args.outliers_percent, args.noise_scale,
                                args.inpaint_ranges, args.epsilon, args.density,
                                cbins=cbins, rbins=rbins, max_npb=max_npb, min_npb=min_npb)

    env = wrap_env(env=env, reward_scale=args.reward_scale)
    env = OfflineEnvWrapper(env)

    # --- MODEL SETUP: Using ContrastiveCDT ---
    model = ContrastiveCDT(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        max_action=env.action_space.high[0],
        embedding_dim=args.embedding_dim,
        contrastive_dim=args.contrastive_dim,
        seq_len=args.seq_len,
        episode_len=args.episode_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        embedding_dropout=args.embedding_dropout,
        time_emb=args.time_emb,
        use_rew=args.use_rew,
        use_cost=args.use_cost,
        cost_transform=args.cost_transform,
        add_cost_feat=args.add_cost_feat,
        mul_cost_feat=args.mul_cost_feat,
        cat_cost_feat=args.cat_cost_feat,
        action_head_layers=args.action_head_layers,
        cost_prefix=args.cost_prefix,
        stochastic=args.stochastic,
        init_temperature=args.init_temperature,
        target_entropy=-env.action_space.shape[0],
    ).to(args.device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.setup_checkpoint_fn(lambda: {"model_state": model.state_dict()})

    # --- TRAINER SETUP: Using ContrastiveCDTTrainer ---
    trainer = ContrastiveCDTTrainer(
        model, env, logger=logger,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        discretization=args.discretization_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=args.betas,
        clip_grad=args.clip_grad,
        lr_warmup_steps=args.lr_warmup_steps,
        reward_scale=args.reward_scale,
        cost_scale=args.cost_scale,
        loss_cost_weight=args.loss_cost_weight,
        loss_state_weight=args.loss_state_weight,
        cost_reverse=args.cost_reverse,
        no_entropy=args.no_entropy,
        device=args.device
    )

    ct = lambda x: 70 - x if args.linear else 1 / (x + 10)

    # --- DATASET SETUP ---
    dataset = SequenceDataset(
        data, seq_len=args.seq_len, reward_scale=args.reward_scale,
        cost_scale=args.cost_scale, deg=args.deg, pf_sample=args.pf_sample,
        max_rew_decrease=args.max_rew_decrease, beta=args.beta,
        augment_percent=args.augment_percent, cost_reverse=args.cost_reverse,
        max_reward=args.max_reward, min_reward=args.min_reward,
        pf_only=args.pf_only, rmin=args.rmin, cost_bins=args.cost_bins,
        npb=args.npb, cost_sample=args.cost_sample, cost_transform=ct,
        start_sampling=args.start_sampling, prob=args.prob,
        random_aug=args.random_aug, aug_rmin=args.aug_rmin,
        aug_rmax=args.aug_rmax, aug_cmin=args.aug_cmin, aug_cmax=args.aug_cmax,
        cgap=args.cgap, rstd=args.rstd, cstd=args.cstd,
    )
    
    # 'data' is the pre-processed list of trajectory dictionaries
    traj_returns = [np.sum(traj["rewards"]) for traj in dataset.dataset]
    traj_costs = [np.sum(traj["costs"]) for traj in dataset.dataset]

    print(f"\n" + "="*45)
    print(f"🌍 ENVIRONMENT: {args.task}")
    print(f"💰 RETURNS -> Max: {np.max(traj_returns):.1f} | Median: {np.median(traj_returns):.1f}")
    print(f"⚠️ COSTS   -> Max: {np.max(traj_costs):.1f} | Median: {np.median(traj_costs):.1f} | Min: {np.min(traj_costs):.1f}")
    print("="*45 + "\n")

    # --- THESIS DATA ENGINEERING AXIS ---
    if args.use_explicit_pairs:
        dataset = PairedContrastiveDataset(
            dataset, 
            discretization=args.discretization_type,
            cost_scale=args.cost_scale
        )

    trainloader = DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.num_workers, drop_last=True
    )
    trainloader_iter = iter(trainloader)

    # Helper function to get infinite batches
    def get_batch():
        nonlocal trainloader_iter
        try:
            return next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            return next(trainloader_iter)

    # --- THESIS OPTIMIZATION AXIS: PRE-TRAINING ---
    if args.pretrain_steps > 0:
        for step in trange(args.pretrain_steps, desc="Pre-training Latents"):
            batch = get_batch()
            states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
                b.to(args.device) for b in batch
            ]
            trainer.train_one_step(
                states, actions, returns, costs_return, time_steps, mask,
                episode_cost, costs, is_pretraining=True
            )

    # for saving the best
    best_reward = -np.inf
    best_cost = np.inf
    best_idx = 0

    # --- MAIN JOINT TRAINING LOOP ---
    for step in trange(args.update_steps, desc="Training"):
        batch = get_batch()
        states, actions, returns, costs_return, time_steps, mask, episode_cost, costs = [
            b.to(args.device) for b in batch
        ]
        
        trainer.train_one_step(
            states, actions, returns, costs_return, time_steps, mask,
            episode_cost, costs, is_pretraining=False
        )

        # --- REPRESENTATION EVALUATION PROBING ---
        if (step + 1) % args.probe_every == 0:
            evaluate_representations(trainer, trainloader, args.device, step)

        # --- STANDARD POLICY EVALUATION ---
        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost = [], []
            log_cost, log_reward, log_len = {}, {}, {}
            for target_return in args.target_returns:
                reward_return, cost_return = target_return
                if args.cost_reverse:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        (args.episode_len - cost_return) * args.cost_scale)
                else:
                    ret, cost, length = trainer.evaluate(
                        args.eval_episodes, reward_return * args.reward_scale,
                        cost_return * args.cost_scale)
                average_cost.append(cost)
                average_reward.append(ret)

                name = f"c_{int(cost_return)}_r_{int(reward_return)}"
                log_cost.update({name: cost})
                log_reward.update({name: ret})
                log_len.update({name: length})

            logger.store(tab="cost", **log_cost)
            logger.store(tab="ret", **log_reward)
            logger.store(tab="length", **log_len)

            logger.save_checkpoint()
            mean_ret, mean_cost = np.mean(average_reward), np.mean(average_cost)
            if mean_cost < best_cost or (mean_cost == best_cost and mean_ret > best_reward):
                best_cost, best_reward, best_idx = mean_cost, mean_ret, step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)
            logger.write(step, display=False)
        else:
            logger.write_without_reset(step)


if __name__ == "__main__":
    train()