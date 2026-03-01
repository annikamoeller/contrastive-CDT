import argparse
import os
import torch
import glob
from research.utils.config import Config
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True,
        help="Path to experiment folder (seed-x-seg)")
    args = parser.parse_args()

    # -------------------------------
    # 1. Load config
    # -------------------------------
    config_path = os.path.join(args.exp_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Did not find config at {config_path}")

    config = Config.load(config_path)
    config = config.parse()
    trainer = config.get_trainer()

    # -------------------------------
    # 2. Robust Model Loading
    # -------------------------------
    # Priority list of models to try
    priority_models = ["best_model.pt", "final_model.pt", "model.pt"]
    
    # Also look for any numbered checkpoints (e.g. checkpoint_10000.pt) and sort them
    checkpoints = glob.glob(os.path.join(args.exp_path, "checkpoint_*.pt"))
    checkpoints.sort(key=os.path.getmtime, reverse=True) # Newest first
    priority_models.extend([os.path.basename(c) for c in checkpoints])

    loaded = False
    
    for model_name in priority_models:
        model_path = os.path.join(args.exp_path, model_name)
        
        if not os.path.exists(model_path):
            continue
            
        # Check file size (skip empty files)
        if os.path.getsize(model_path) < 1024: 
            print(f"[Warning] Skipping {model_name} (file too small/corrupt)")
            continue

        print(f"Attempting to load: {model_path}")
        try:
            # FIX: Load to CPU first to bypass Stream/Zip errors
            state_dict = torch.load(model_path, map_location="cpu")
            
            # Handle standard vs nested dicts
            if "model" in state_dict:
                trainer.model.load_state_dict(state_dict["model"])
            else:
                trainer.model.load_state_dict(state_dict)
                
            trainer.model.to(trainer.device) # Move to GPU only after successful load
            print(f"Successfully loaded {model_name}!")
            loaded = True
            break
        except Exception as e:
            print(f"[Error] Failed to load {model_name}: {e}")

    if not loaded:
        raise RuntimeError("Could not load ANY valid model checkpoint.")

    # -------------------------------
    # 3. Evaluate
    # -------------------------------
    metrics = trainer.evaluate(args.exp_path, 10)

    with open(f"{args.exp_path}/results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Results saved.")

if __name__ == "__main__":
    main()