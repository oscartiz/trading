import numpy as np
import torch

def process_snapshots(snapshots: np.ndarray, num_levels: int = 10, lookback: int = 100):
    """
    Construct input tensors from raw snapshot data using decimal-scale and level-relative normalization.
    
    Args:
        snapshots: np.ndarray of shape (N, 40) where 40 = 10 levels * 2 sides * 2 (price, vol)
                   Format: [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...]
        num_levels: number of levels in the order book (default 10)
        lookback: number of historical steps for the time series (default 100)
        
    Returns:
        torch.Tensor of shape (N - lookback + 1, 1, lookback, 40)
    """
    N = len(snapshots)
    if N < lookback:
        raise ValueError(f"Not enough snapshots ({N}) for lookback window ({lookback}).")
        
    # The layout is: for level i in 0..num_levels-1:
    # 4*i + 0: ask_p
    # 4*i + 1: ask_v
    # 4*i + 2: bid_p
    # 4*i + 3: bid_v
    
    processed = np.zeros_like(snapshots, dtype=np.float32)
    
    # 1. Price Normalization (Level-Relative to Mid-Price)
    ask_p1 = snapshots[:, 0]
    bid_p1 = snapshots[:, 2]
    mid_prices = (ask_p1 + bid_p1) / 2.0
    
    for i in range(num_levels):
        ask_idx = 4 * i
        bid_idx = 4 * i + 2
        
        # Relative deviation from mid_price: (p - mid) / mid
        # Ensures spread is preserved as a ratio
        processed[:, ask_idx] = (snapshots[:, ask_idx] - mid_prices) / mid_prices
        processed[:, bid_idx] = (snapshots[:, bid_idx] - mid_prices) / mid_prices
        
    # 2. Volume Normalization (Decimal-Scale / Fraction of total visible)
    vol_indices = [4 * i + 1 for i in range(num_levels)] + [4 * i + 3 for i in range(num_levels)]
    total_visible_volume = np.sum(snapshots[:, vol_indices], axis=1)
    # Avoid division by zero
    total_visible_volume[total_visible_volume == 0] = 1.0 
    
    for idx in vol_indices:
        processed[:, idx] = snapshots[:, idx] / total_visible_volume
        
    # 3. Construct Time-Series Tensors using rolling windows
    # Shape we want: (N - lookback + 1, 1, lookback, 40)
    num_samples = N - lookback + 1
    
    # Using numpy stride tricks for memory efficiency before converting to torch
    stride = processed.strides[0]
    windowed = np.lib.stride_tricks.as_strided(
        processed, 
        shape=(num_samples, lookback, 40), 
        strides=(stride, stride, processed.strides[1])
    )
    
    # Add channel dimension
    windowed = np.expand_dims(windowed, axis=1)
    
    return torch.from_numpy(windowed.copy())
