import os
import sys
import types
from dataclasses import asdict, dataclass
import gymnasium as gym
import numpy as np
import pyrallis
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
import dsrl
import bullet_safety_gym
import datetime

from examples.configs.cdt_configs import CDT_DEFAULT_CONFIG, CDTTrainConfig
from osrl.common import SequenceDataset
from osrl.common.exp_util import seed_all
from fsrl.utils import WandbLogger

# --- THESIS COMPONENTS ---
from osrl_contrastive.ccdt import ContrastiveCDT
from osrl_contrastive.ccdt_trainer import ContrastiveCDTTrainer
from osrl_contrastive.probe_and_vis import evaluate_representations

def get_cost_boundaries(data: dict, num_buckets: int) -> list:
    """Calculates distribution-based boundaries for N equally-sized buckets."""
    if num_buckets <= 1:
        return []
        
    terminals = data['terminals']
    timeouts = data['timeouts']
    costs = data['costs']
    
    episode_ends = np.where(np.logical_or(terminals, timeouts))[0]
    
    episode_costs = []
    start_idx = 0
    for end_idx in episode_ends:
        ep_cost = np.sum(costs[start_idx:end_idx + 1])
        episode_costs.append(ep_cost)
        start_idx = end_idx + 1
        
    # Calculate the exact percentiles needed for N buckets
    quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1]
    boundaries = np.quantile(episode_costs, quantiles).tolist()
    
    print(f"\n🚀 [CCDT Setup] Global Cost Boundaries for {num_buckets} buckets: {boundaries}\n")
    return boundaries

@dataclass
class ContrastiveCDTTrainConfig(CDTTrainConfig):
    num_buckets: int = 2               
    pretrain_steps: int = 0            
    update_steps: int = 20_000          
    contrastive_dim: int = 64
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    probe_every: int = 5000            
    eval_episodes: int = 5      

@pyrallis.wrap()
def train(args: ContrastiveCDTTrainConfig):
    # 1. Identify which arguments were explicitly passed via command line
    cfg, old_cfg = asdict(args), asdict(ContrastiveCDTTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    
    # 2. Fetch the highly-tuned default parameters for this specific environment
    if args.task in CDT_DEFAULT_CONFIG:
        base_cfg = asdict(CDT_DEFAULT_CONFIG[args.task]())
    else:
        base_cfg = asdict(CDTTrainConfig())
        
    # 3. Apply the environment defaults, then apply any command line changes
    cfg.update(base_cfg)
    cfg.update(differing_values)
    
    # 4. THE FIX: Explicitly force your custom thesis parameters to survive
    cfg["update_steps"] = args.update_steps
    cfg["num_buckets"] = args.num_buckets
    cfg["contrastive_dim"] = args.contrastive_dim
    cfg["contrastive_weight"] = args.contrastive_weight
    cfg["temperature"] = args.temperature
    cfg["probe_every"] = args.probe_every
    cfg["eval_episodes"] = args.eval_episodes
    cfg["pretrain_steps"] = args.pretrain_steps

    # 5. Convert back to an argument namespace
    args = types.SimpleNamespace(**cfg)

    # --- NEW: Clean Naming for WandB & Local Logs ---
    env_short = args.task.split("-")[0].replace("Offline", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.name = f"{env_short}_{args.num_buckets}B_{args.pretrain_steps}Pre_{timestamp}"
    
    # DO NOT overwrite args.task here. Keep it as "OfflineCarCircle-v0"
    if args.group is None:
        args.group = f"{args.task}-contrastive-experiments"
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
        # THE SAFETY NET: Physically create the folder so torch.save never crashes
        os.makedirs(args.logdir, exist_ok=True)
        
    # Pass the custom name to the logger
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    seed_all(args.seed)

    # Initialize Environment
    all_envs = [env_id for env_id in gym.envs.registration.registry.keys() if "Offline" in env_id]
    print(f"DEBUG: Found {len(all_envs)} Offline environments in registry.")
    if args.task not in all_envs:
        print(f"⚠️ ERROR: {args.task} is NOT in the registry! Available: {all_envs[:5]}...")
        
    env = gym.make(args.task)
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)
    ct = lambda x: 70 - x if args.linear else 1 / (x + 10)
    try: 
        cost_boundaries = get_cost_boundaries(data, args.num_buckets)
    except:
        print("Could not create cost boundaries")
        sys.exit(1)
    
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

    # --- TRAINER SETUP ---
    trainer = ContrastiveCDTTrainer(
        model, env, logger=logger,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        num_buckets=args.num_buckets,         
        cost_boundaries=cost_boundaries,
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

    trainloader = DataLoader(
        dataset, batch_size=args.batch_size, pin_memory=True,
        num_workers=args.num_workers, drop_last=True
    )
    trainloader_iter = iter(trainloader)

    def get_batch():
        nonlocal trainloader_iter
        try:
            return next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            return next(trainloader_iter)

    # Phase 1: Pre-training
    if args.pretrain_steps > 0:
        for _ in trange(args.pretrain_steps, desc="Pre-training"):
            batch = [b.to(args.device) for b in get_batch()]
            trainer.train_one_step(*batch, is_pretraining=True)

    # Phase 2: Joint Training
    for step in trange(args.update_steps, desc="Training"):
        batch = [b.to(args.device) for b in get_batch()]
        trainer.train_one_step(*batch, is_pretraining=False)

        if (step + 1) % args.probe_every == 0:
            # Pass num_buckets to the probe so t-SNE colors match your labels
            evaluate_representations(trainer, trainloader, args.device, step, num_buckets=args.num_buckets)

        if (step + 1) % args.eval_every == 0 or step == args.update_steps - 1:
            average_reward, average_cost = [], []
            for target_return in args.target_returns:
                reward_to_go, cost_to_go = target_return

                # This calls your new vectorized evaluate method
                ret, cost, length = trainer.evaluate(
                    num_rollouts=args.eval_episodes, 
                    target_return=reward_to_go * args.reward_scale,
                    target_cost=cost_to_go * args.cost_scale
                )

                average_cost.append(cost)
                average_reward.append(ret)

                # Log to WandB
                name = f"c_{int(cost_to_go)}_r_{int(reward_to_go)}"
                logger.store(tab="cost", **{name: cost})
                logger.store(tab="ret", **{name: ret})
                logger.store(tab="length", **{name: length})

            logger.write(step, display=False)
            model_path = os.path.join(args.logdir, "model.pt")
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    train()