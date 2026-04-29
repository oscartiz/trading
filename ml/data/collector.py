import requests
import numpy as np
import time
import os
import json

def fetch_snapshot(symbol="BTCUSDT", limit=10):
    """
    Fetch a single depth snapshot from Binance REST API.
    """
    url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    # Process into our 40-feature format
    # [ask_p1, ask_v1, bid_p1, bid_v1, ...]
    row = []
    asks = data.get("asks", [])
    bids = data.get("bids", [])
    
    for i in range(limit):
        ask_p = float(asks[i][0]) if i < len(asks) else 0.0
        ask_v = float(asks[i][1]) if i < len(asks) else 0.0
        bid_p = float(bids[i][0]) if i < len(bids) else 0.0
        bid_v = float(bids[i][1]) if i < len(bids) else 0.0
        row.extend([ask_p, ask_v, bid_p, bid_v])
        
    return row

def collect_data(num_samples: int = 1000, interval_ms: int = 100, output_file: str = "snapshots.npy"):
    """
    Collects a specified number of snapshots and saves to disk.
    For production, consider using websockets.
    """
    print(f"Collecting {num_samples} snapshots...")
    snapshots = []
    
    try:
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Collected {i}/{num_samples}")
            row = fetch_snapshot()
            snapshots.append(row)
            time.sleep(interval_ms / 1000.0)
    except KeyboardInterrupt:
        print("Data collection interrupted.")
    except Exception as e:
        print(f"Error during collection: {e}")
        
    snapshots_arr = np.array(snapshots, dtype=np.float32)
    np.save(output_file, snapshots_arr)
    print(f"Saved {len(snapshots_arr)} snapshots to {output_file}")
    return snapshots_arr

def generate_synthetic_data(num_samples: int = 5000, num_levels: int = 10) -> np.ndarray:
    """
    Generates synthetic order book data for testing the pipeline when Binance is unreachable.
    """
    print(f"Generating {num_samples} synthetic snapshots...")
    snapshots = np.zeros((num_samples, num_levels * 4), dtype=np.float32)
    
    # Base mid price follows a random walk
    mids = 50000.0 + np.cumsum(np.random.randn(num_samples) * 5.0)
    
    for i in range(num_samples):
        mid = mids[i]
        for l in range(num_levels):
            spread = 0.5 + l * 0.5
            ask_p = mid + spread
            bid_p = mid - spread
            ask_v = np.random.uniform(0.1, 5.0)
            bid_v = np.random.uniform(0.1, 5.0)
            
            idx = 4 * l
            snapshots[i, idx] = ask_p
            snapshots[i, idx+1] = ask_v
            snapshots[i, idx+2] = bid_p
            snapshots[i, idx+3] = bid_v
            
    return snapshots

if __name__ == "__main__":
    # Example usage
    # data = collect_data(num_samples=2000, interval_ms=100)
    pass
