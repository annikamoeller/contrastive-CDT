import torch
import os

try:
    # Load to CPU to avoid GPU errors
    model = torch.load('exp_safetygym/OfflineDroneCircle-v0/SafetyDroneCircle-v0-100-1923/TraC/cost-40.0/seed-0-seg/best_model.pt', map_location='cpu')
    print("✅ Load Successful!")
    print(f"Type: {type(model)}")
    if isinstance(model, dict): print(f"Keys: {list(model.keys())[:3]}...") 
except Exception as e:
    print(f"❌ Error: {e}")