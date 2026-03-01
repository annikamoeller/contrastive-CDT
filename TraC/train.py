import argparse
import os


def try_wandb_setup(path, config):
    # === Load W&B API key from local file === #
    key_file = os.path.join(os.path.dirname(__file__), "wandb_key.txt")

    wandb_api_key = None
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            wandb_api_key = f.read().strip()

    # Optionally print for debugging
    print("Loaded WANDB key:", wandb_api_key)

    # If key exists → set environment AUTH BEFORE importing wandb
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
        os.environ["WANDB_LOGIN_MODE"] = "online"   # avoid interactive prompt

        try:
            import wandb
        except ImportError:
            return

        # Force non-interactive login (avoids browser prompt)
        wandb.login(key=wandb_api_key)

        project_dir = os.path.dirname(__file__)

        # === More descriptive W&B run naming === #
        dataset_name = config["dataset_kwargs"]["path"].split("/")[-1].replace(".hdf5", "")
        env_name = config["eval_env"]
        algo = config["alg"]
        cost = config["trainer_kwargs"]["target_cost"]
        seed = config["seed"]

        name = f"{dataset_name}_{env_name}_{algo}_cost{cost}_seed{seed}"

        # group can stay as before, or enhance it too
        group = env_name

        wandb.init(
            project=os.path.basename(project_dir),
            name=name,
            config=config.flatten(separator="-"),
            dir=os.path.join(project_dir, "exp"),
            group=group,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--eval_env", "-e", type=str, default=None)
    parser.add_argument("--dataset_path", "-dp", type=str, default=None)
    parser.add_argument("--target_cost", "-tc", type=float, default=None)
    parser.add_argument("--alpha", "-a", type=float, default=None)
    parser.add_argument("--gamma", "-g", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--seg_ratio", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--safe_top_perc", type=float, default=None)
    parser.add_argument("--safe_bottom_perc", type=float, default=None)
    args = parser.parse_args()

    import configs.metadrive_env as metaenv
    metaenv.METADRIVE_ENV = True if "Metadrive" in args.eval_env else False

    from research.utils.config import Config
    config = Config.load(args.config)
    config["seed"] = args.seed
    config["eval_env"] = args.eval_env
    config["dataset_kwargs"]["path"] = args.dataset_path
    config["trainer_kwargs"]["target_cost"] = args.target_cost  # BulletGym: [10, 20, 40]; SafetyGym: [20, 40, 80]
    config["alg_kwargs"]["alpha"] = args.alpha
    config["alg_kwargs"]["gamma"] = args.gamma
    config["alg_kwargs"]["eta"] = args.eta
    config["dataset_kwargs"]["seg_ratio"] = args.seg_ratio
    config["dataset_kwargs"]["kappa"] = args.kappa
    config["dataset_kwargs"]["safe_top_perc"] = args.safe_top_perc
    config["dataset_kwargs"]["safe_bottom_perc"] = args.safe_bottom_perc

    log_path = os.path.join(args.path, config["eval_env"], config["dataset_kwargs"]["path"].split("/")[-1][:-5], config["alg"], 
                            f"cost-{str(config['trainer_kwargs']['target_cost'])}", f"seed-{str(config['seed'])}-seg")
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)  # Change this to false temporarily so we don't recreate experiments
    try_wandb_setup(log_path, config)
    config.save(log_path)  # Save the config
    
    # Parse the config file to resolve names.
    config = config.parse()
    trainer = config.get_trainer(device=args.device)

    # Train the model
    trainer.train(log_path)