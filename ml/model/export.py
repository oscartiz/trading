import os
import json
import torch
from safetensors.torch import save_file
from .deeplob import DeepLOB

def export_model(model: DeepLOB, export_path: str = "deeplob.safetensors", config_path: str = "model_config.json"):
    """
    Export PyTorch model weights to .safetensors format for Rust Candle-Core inference.
    Also exports the configuration to JSON.
    """
    print(f"Exporting model to {export_path}...")
    
    # 1. Export weights
    # Note: safetensors standard doesn't save the architecture, only state dict.
    # The state dict keys will match `candle_nn::VarBuilder` expectations exactly.
    state_dict = model.state_dict()
    
    # Ensure all tensors are contiguous before saving
    for k, v in state_dict.items():
        state_dict[k] = v.contiguous()
        
    save_file(state_dict, export_path)
    print("Safetensors export complete.")
    
    # 2. Export configuration
    config = {
        "num_levels": 10,
        "features_per_level": 4,
        "lookback": 100,
        "num_classes": 3,
        "conv_channels": [1, 32, 32, 32],
        "inception_channels": 64,
        "lstm_hidden": 64,
        "lstm_layers": 1
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Exported model config to {config_path}")
    
if __name__ == "__main__":
    # Test export
    dummy_model = DeepLOB()
    export_model(dummy_model, "test_deeplob.safetensors", "test_config.json")
