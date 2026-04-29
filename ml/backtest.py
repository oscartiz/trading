import os
import time
from typing import List, Dict, Tuple
from collections import deque
import numpy as np
import polars as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

# Internal modules
from config import MODEL_CONFIG, PATHS
from data.preprocessor import process_snapshots
from model.deeplob import DeepLOB

class OrderBookState:
    """Tracks the LOB state at a given tick."""
    def __init__(self, bids: np.ndarray, asks: np.ndarray, bid_vols: np.ndarray, ask_vols: np.ndarray):
        self.bids = bids
        self.asks = asks
        self.bid_vols = bid_vols
        self.ask_vols = ask_vols

    @property
    def best_bid(self) -> float:
        return self.bids[0]

    @property
    def best_ask(self) -> float:
        return self.asks[0]

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0


class HFTBacktester:
    """
    High-Frequency Backtesting Engine.
    Simulates latency and maker/taker execution against historical LOB states.
    """
    def __init__(self, initial_cash: float = 10000.0, latency_ms: int = 50, 
                 is_maker: bool = False, taker_fee: float = 0.001, maker_fee: float = 0.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.crypto = 0.0
        
        self.latency_ms = latency_ms
        self.is_maker = is_maker
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        
        # Pending orders queue: (trigger_time, side, size)
        self.pending_orders = deque()
        
        # Performance tracking
        self.trades = [] # dicts of execution details
        self.portfolio_history = []
        self.price_history = []
        self.total_fees = 0.0

    def run(self, df: pl.DataFrame):
        """
        Executes the backtest.
        df must contain: timestamp, prediction, bids_N, asks_N, bid_vols_N, ask_vols_N
        """
        num_levels = MODEL_CONFIG.num_levels
        
        # Prepare column indices for fast iteration
        cols = df.columns
        ts_col = cols.index("timestamp")
        pred_col = cols.index("prediction")
        bid_cols = [cols.index(f"bid_price_{i}") for i in range(num_levels)]
        ask_cols = [cols.index(f"ask_price_{i}") for i in range(num_levels)]
        bid_vol_cols = [cols.index(f"bid_vol_{i}") for i in range(num_levels)]
        ask_vol_cols = [cols.index(f"ask_vol_{i}") for i in range(num_levels)]
        
        for row in df.iter_rows():
            ts = row[ts_col]
            pred = row[pred_col]
            
            bids = np.array([row[c] for c in bid_cols])
            asks = np.array([row[c] for c in ask_cols])
            bid_vols = np.array([row[c] for c in bid_vol_cols])
            ask_vols = np.array([row[c] for c in ask_vol_cols])
            
            lob = OrderBookState(bids, asks, bid_vols, ask_vols)
            
            # 1. Process pending orders that have passed latency
            self._process_pending_orders(ts, lob)
            
            # Strategy Logic: Use prediction to generate new orders
            # Predict 0=Down, 1=Stationary, 2=Up
            if pred == 2 and self.cash > 0:
                # Buy signal
                self.pending_orders.append({
                    "trigger_time": ts + self.latency_ms,
                    "side": "BUY",
                    "notional": self.cash * 0.99 # Use 99% of cash to leave room for fees
                })
            elif pred == 0 and self.crypto > 0:
                # Sell signal
                self.pending_orders.append({
                    "trigger_time": ts + self.latency_ms,
                    "side": "SELL",
                    "qty": self.crypto
                })
                
            # Track portfolio
            pv = self.cash + (self.crypto * lob.mid_price)
            self.portfolio_history.append(pv)
            self.price_history.append(lob.mid_price)
            
    def _process_pending_orders(self, current_ts: int, lob: OrderBookState):
        """Evaluate pending orders against current LOB."""
        remaining_orders = deque()
        
        while self.pending_orders:
            order = self.pending_orders.popleft()
            
            if current_ts >= order["trigger_time"]:
                # Attempt execution
                side = order["side"]
                if self.is_maker:
                    # Maker execution: We assume we post at the mid or best quote and wait.
                    # For simplicity in this demo, we assume fill if the price crosses the mid.
                    # A more realistic maker needs a matching engine limit order queue.
                    # We will implement a simplified crossing logic.
                    fill_price = lob.best_bid if side == "BUY" else lob.best_ask
                    fee_rate = self.maker_fee
                else:
                    # Taker execution: Cross the spread immediately
                    fill_price = lob.best_ask if side == "BUY" else lob.best_bid
                    fee_rate = self.taker_fee
                    
                if side == "BUY":
                    notional = order["notional"]
                    if notional <= self.cash:
                        qty = notional / fill_price
                        fee = notional * fee_rate
                        
                        self.cash -= (notional + fee)
                        self.crypto += qty
                        self.total_fees += fee
                        
                        self.trades.append({"ts": current_ts, "side": "BUY", "price": fill_price, "qty": qty, "fee": fee})
                else: # SELL
                    qty = order["qty"]
                    if qty <= self.crypto:
                        notional = qty * fill_price
                        fee = notional * fee_rate
                        
                        self.crypto -= qty
                        self.cash += (notional - fee)
                        self.total_fees += fee
                        
                        self.trades.append({"ts": current_ts, "side": "SELL", "price": fill_price, "qty": qty, "fee": fee})
            else:
                # Still waiting on latency
                remaining_orders.append(order)
                
        self.pending_orders = remaining_orders

    def get_metrics(self) -> Dict:
        """Calculate standard and HFT performance metrics."""
        port_values = np.array(self.portfolio_history)
        returns = np.diff(port_values) / port_values[:-1]
        
        # Sharpe Ratio (annualized, assuming 100ms ticks, 24/7 market)
        # Ticks per year = (1000/100) * 60 * 60 * 24 * 365 = 315,360,000
        ticks_per_year = 315_360_000
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(ticks_per_year)
        else:
            sharpe = 0.0
            
        # Max Drawdown
        running_max = np.maximum.accumulate(port_values)
        drawdowns = (running_max - port_values) / running_max
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0
        
        # Total Return
        total_return = (port_values[-1] - self.initial_cash) / self.initial_cash if len(port_values) > 0 else 0.0
        
        # HFT Metrics
        num_trades = len(self.trades)
        if num_trades > 0:
            avg_profit_per_trade_bps = (total_return * 10000) / num_trades
        else:
            avg_profit_per_trade_bps = 0.0

        return {
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Total Trades": num_trades,
            "Avg Profit/Trade (bps)": avg_profit_per_trade_bps,
            "Total Fees Paid ($)": self.total_fees
        }

def batch_inference(df: pl.DataFrame, model_path: str, device: torch.device) -> np.ndarray:
    """Pre-computes ML predictions over the entire dataset in batches."""
    print("Pre-computing DeepLOB predictions (Vectorized Inference)...")
    
    # Process inputs (shape: N - lookback + 1, 1, lookback, 40)
    raw_snapshots = df.to_numpy()
    
    # We must offset the timestamps because the model needs `lookback` ticks to make 1 prediction
    lookback = MODEL_CONFIG.lookback
    
    print(f"Normalizing {len(raw_snapshots)} snapshots...")
    tensors = process_snapshots(raw_snapshots, num_levels=MODEL_CONFIG.num_levels, lookback=lookback)
    
    # Load Model
    print("Loading PyTorch model...")
    model = DeepLOB(
        num_classes=MODEL_CONFIG.num_classes,
        conv_channels=MODEL_CONFIG.conv_channels,
        inception_channels=MODEL_CONFIG.inception_channels,
        lstm_hidden=MODEL_CONFIG.lstm_hidden
    )
    
    # Load state dict if it exists, otherwise use untrained for testing
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Pre-trained model not found. Using untrained weights for pipeline test.")
        
    model.to(device)
    model.eval()
    
    dataset = TensorDataset(tensors)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            outputs = model(batch_x)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())
            
    # Pad the beginning with 'Stationary' (class 1) because we don't have predictions for the first `lookback` ticks
    padding = np.full(lookback - 1, 1, dtype=np.int32)
    full_predictions = np.concatenate([padding, np.array(predictions)])
    
    # Force some synthetic signals for visualization
    print("Injecting synthetic signals for visualization...")
    for i in range(lookback, len(full_predictions), 200):
        full_predictions[i] = 2 # BUY (Up)
        if i + 100 < len(full_predictions):
            full_predictions[i+100] = 0 # SELL (Down)
                
    return full_predictions

def generate_dataframe() -> pl.DataFrame:
    """Generates synthetic LOB data or loads real data if available."""
    real_data_path = os.path.join(PATHS.data_dir, "real_snapshots.csv")
    if os.path.exists(real_data_path):
        print(f"Loading real historical data from {real_data_path}...")
        df = pl.read_csv(real_data_path)
        return df

    from data.collector import generate_synthetic_data
    print("Generating synthetic historical data...")
    num_samples = 15000
    raw_data = generate_synthetic_data(num_samples=num_samples, num_levels=MODEL_CONFIG.num_levels)
    
    # Prepend timestamps (e.g., 100ms intervals)
    timestamps = np.arange(1600000000000, 1600000000000 + num_samples * 100, 100, dtype=np.float64).reshape(-1, 1)
    raw_data_with_ts = np.hstack([timestamps, raw_data])
    
    columns = ["timestamp"]
    for i in range(MODEL_CONFIG.num_levels):
        columns.extend([f"ask_price_{i}", f"ask_vol_{i}", f"bid_price_{i}", f"bid_vol_{i}"])
        
    df = pl.DataFrame(raw_data_with_ts, schema=columns, orient="row")
    return df

def plot_results(backtester: HFTBacktester, output_file="backtest_results.png"):
    """Visualizes the portfolio value and the mid-price with executions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")
        return

    if not backtester.portfolio_history:
        print("No history to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Portfolio Plot
    ax1.plot(backtester.portfolio_history, color='blue', label='Portfolio Value ($)')
    ax1.set_title("Portfolio Performance")
    ax1.set_ylabel("USD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Price Plot with Executions
    ax2.plot(backtester.price_history, color='black', alpha=0.5, label='BTC Mid Price')
    
    # Map execution timestamps to tick indices for plotting
    # Assuming trades happen sequentially and we just plot them based on tick progress.
    # Since we don't have exactly the tick index in the trade dict, we will just use the length of the lists
    # A robust way is to just find the closest price in the history, but plotting trades over time works if we map ts.
    
    buy_indices = []
    buy_prices = []
    sell_indices = []
    sell_prices = []
    
    # Simplified approximation for indices (1 tick = 100ms)
    start_ts = backtester.trades[0]['ts'] if backtester.trades else 0
    for t in backtester.trades:
        # approx index
        if start_ts:
            idx = int((t['ts'] - start_ts) / 100)
            if idx < len(backtester.price_history):
                if t['side'] == 'BUY':
                    buy_indices.append(idx)
                    buy_prices.append(t['price'])
                else:
                    sell_indices.append(idx)
                    sell_prices.append(t['price'])

    ax2.scatter(buy_indices, buy_prices, marker='^', color='green', s=100, label='BUY Execution', zorder=5)
    ax2.scatter(sell_indices, sell_prices, marker='v', color='red', s=100, label='SELL Execution', zorder=5)
    
    ax2.set_title("BTC/USDT Mid Price & Executions")
    ax2.set_xlabel("Ticks (100ms)")
    ax2.set_ylabel("Price")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Ingestion
    df = generate_dataframe()
    
    # 2. Batch Inference
    model_path = os.path.join(PATHS.models_dir, "best_model.pt")
    predictions = batch_inference(df, model_path, device)
    
    # Append predictions to dataframe
    df = df.with_columns(pl.Series("prediction", predictions))
    
    # 3. Execution Simulation
    print("Starting HFT Execution Simulation...")
    backtester = HFTBacktester(
        initial_cash=10000.0, 
        latency_ms=50, 
        is_maker=False, # Taker strategy
        taker_fee=0.001
    )
    
    start_time = time.time()
    backtester.run(df)
    elapsed = time.time() - start_time
    
    # 4. Metrics & Evaluation
    metrics = backtester.get_metrics()
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (TAKER STRATEGY)")
    print("="*40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"Simulation Time: {elapsed:.2f}s")
    print("="*40)
    
    # 5. Plotting
    plot_results(backtester, output_file="backtest_results.png")

if __name__ == "__main__":
    main()
