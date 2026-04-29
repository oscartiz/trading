import numpy as np

def compute_labels(snapshots: np.ndarray, horizon: int = 50, alpha: float = 0.002) -> np.ndarray:
    """
    Compute labels using a smoothed forward-rolling window.
    
    Args:
        snapshots: np.ndarray of shape (N, 40) where 40 = 10 levels * 2 sides * 2 (price, vol)
                   Format: [ask_p1, ask_v1, bid_p1, bid_v1, ...]
        horizon: The future window to average over
        alpha: The threshold for defining the stationary band
        
    Returns:
        np.ndarray of shape (N,) containing labels: 0 (Down), 1 (Stationary), 2 (Up)
        The last `horizon` elements will be filled with 1 (Stationary) or discarded during training.
    """
    N = len(snapshots)
    labels = np.ones(N, dtype=np.int64) # Default to 1 (Stationary)
    
    if N <= horizon:
         return labels
         
    ask_p1 = snapshots[:, 0]
    bid_p1 = snapshots[:, 2]
    mid_prices = (ask_p1 + bid_p1) / 2.0
    
    # Smoothed forward rolling window
    # future_avg[t] = mean(mid_prices[t+1 : t+horizon+1])
    # Efficient computation using convolution
    kernel = np.ones(horizon) / horizon
    
    # We use 'valid' mode and then pad, or we can just iterate.
    # For large arrays, a rolling sum or convolution is faster.
    # np.convolve(mid_prices, kernel, mode='valid') gives the rolling means starting at index horizon-1.
    rolling_means = np.convolve(mid_prices, kernel, mode='valid')
    
    # For time t, the rolling_means index corresponding to [t+1 : t+horizon+1] is t+1 (since valid convolution starts exactly there)
    # rolling_means length is N - horizon + 1. 
    # rolling_means[0] = mean(mid_prices[0:horizon]), which is for t=-1 (invalid)
    # Actually rolling_means[1] = mean(mid_prices[1:horizon+1]), which is future for t=0.
    
    future_avgs = rolling_means[1:] # shape is (N - horizon,)
    
    # Compute percentage change
    current_mids = mid_prices[:N - horizon]
    pct_changes = (future_avgs - current_mids) / current_mids
    
    # Assign labels
    # 0 = Down (pct_change < -alpha)
    # 1 = Stationary (-alpha <= pct_change <= alpha)
    # 2 = Up (pct_change > alpha)
    
    labels[:N - horizon][pct_changes > alpha] = 2
    labels[:N - horizon][pct_changes < -alpha] = 0
    
    return labels

def align_data_and_labels(tensors: np.ndarray, labels: np.ndarray, lookback: int):
    """
    Aligns the tensors from preprocessor with labels from compute_labels.
    
    tensors: shape (N - lookback + 1, 1, lookback, 40)
    labels: shape (N,)
    
    The tensor at index `i` ends exactly at original snapshot `i + lookback - 1`.
    The corresponding label should be predicting the future from time `i + lookback - 1`.
    """
    aligned_labels = labels[lookback - 1:]
    return tensors, aligned_labels
