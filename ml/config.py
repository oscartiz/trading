import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # LOB characteristics
    num_levels: int = 10
    features_per_level: int = 4  # ask_price, ask_vol, bid_price, bid_vol
    
    # Temporal horizon
    lookback: int = 100
    
    # Model architecture
    num_classes: int = 3
    conv_channels: tuple = (1, 32, 32, 32)
    inception_channels: int = 64
    lstm_hidden: int = 64
    lstm_layers: int = 1

@dataclass
class TrainConfig:
    # Labeling parameters
    horizon: int = 50
    alpha: float = 0.0005
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 7
    
    # Hardware
    device: str = "cpu" # Default; will be updated in train.py dynamically
    
@dataclass
class PathsConfig:
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    data_dir: str = os.path.join(base_dir, "data_store")
    models_dir: str = os.path.join(base_dir, "checkpoints")
    
    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

MODEL_CONFIG = ModelConfig()
TRAIN_CONFIG = TrainConfig()
PATHS = PathsConfig()
