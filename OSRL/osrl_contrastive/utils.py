import torch
import os
import yaml
import re
from types import SimpleNamespace
from OSRL.osrl_contrastive.ccdt import ContrastiveCDTFront, ContrastiveCDTBack

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d

def load_model_and_config(ckpt_path, device="cpu"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # 1. Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 2. Find and load config.yaml
    model_dir = os.path.dirname(ckpt_path)
    folder_name = os.path.basename(model_dir)
    possible_paths = [
        os.path.join(model_dir, "config.yaml"),
        os.path.join(model_dir, folder_name, "config.yaml")
    ]
    
    config_dict = None
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                print(f"📖 Loaded config from: {path}")
                break

    if config_dict is None:
        raise ValueError(f"❌ No config.yaml found in {model_dir}")

    # 3. Setup arguments for the Model
    # THE FIX: Strip out all `None` values so we don't overwrite defaults with NoneType
    model_kwargs = {k: v for k, v in config_dict.items() if v is not None}

    # --- INFER GEOMETRY FROM SAVED WEIGHTS ---
    if 'state_emb.weight' in checkpoint:
        model_kwargs['state_dim'] = checkpoint['state_emb.weight'].shape[1]
    if 'action_emb.weight' in checkpoint:
        model_kwargs['action_dim'] = checkpoint['action_emb.weight'].shape[1]
    
    # Securely set max_action
    if model_kwargs.get('max_action') is None:
        model_kwargs['max_action'] = 1.0

    # --- THE SMART INITIALIZATION LOOP ---
    while True:
        try:
            model = ContrastiveCDTBack(**model_kwargs)
            break 
        except TypeError as e:
            error_msg = str(e)
            if "unexpected keyword argument" in error_msg:
                match = re.search(r"unexpected keyword argument '(.*?)'", error_msg)
                if match:
                    bad_key = match.group(1)
                    model_kwargs.pop(bad_key, None) 
                else:
                    raise e 
            else:
                raise e 

    # 4. Load weights into the successfully built model
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 5. Create config object for Trainer
    config_dict['state_dim'] = model_kwargs['state_dim']
    config_dict['action_dim'] = model_kwargs['action_dim']
    config_dict['max_action'] = model_kwargs['max_action']
    
    config = dict_to_namespace(config_dict)

    return model, config