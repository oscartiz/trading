import numpy as np
import torch

class ZScorePreprocessor:
    """
    Handles robust Z-Score normalization by fitting parameters strictly on the training set
    and applying them to validation/test sets to prevent forward-looking data leakage.
    """
    def __init__(self, num_levels: int = 10, lookback: int = 100):
        self.num_levels = num_levels
        self.lookback = lookback
        self.mean = None
        self.std = None
        
    def fit(self, snapshots: np.ndarray):
        """Compute mean and std on the training split."""
        self.mean = np.mean(snapshots, axis=0)
        self.std = np.std(snapshots, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1e-8
        
    def save(self, filepath: str):
        if self.mean is None or self.std is None:
            raise ValueError("Cannot save unfitted preprocessor.")
        np.savez(filepath, mean=self.mean, std=self.std)
        
    def load(self, filepath: str):
        data = np.load(filepath)
        self.mean = data['mean']
        self.std = data['std']
        
    def transform(self, snapshots: np.ndarray) -> torch.Tensor:
        """Apply Z-score normalization and convert to windowed Time-Series Tensors."""
        if self.mean is None or self.std is None:
            raise ValueError("Preprocessor has not been fitted.")
            
        N = len(snapshots)
        if N < self.lookback:
            raise ValueError(f"Not enough snapshots ({N}) for lookback window ({self.lookback}).")
            
        # Z-score normalization
        processed = (snapshots - self.mean) / self.std
        
        # Construct Time-Series Tensors using rolling windows
        # Shape we want: (N - lookback + 1, 1, lookback, 40)
        num_samples = N - self.lookback + 1
        
        # Using numpy stride tricks for memory efficiency before converting to torch
        stride = processed.strides[0]
        windowed = np.lib.stride_tricks.as_strided(
            processed, 
            shape=(num_samples, self.lookback, self.num_levels * 4), 
            strides=(stride, stride, processed.strides[1])
        )
        
        # Add channel dimension
        windowed = np.expand_dims(windowed, axis=1)
        
        return torch.from_numpy(windowed.copy()).float()
        
    def fit_transform(self, snapshots: np.ndarray) -> torch.Tensor:
        self.fit(snapshots)
        return self.transform(snapshots)
