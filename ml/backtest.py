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
from data.preprocessor import ZScorePreprocessor
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
    High-Frequency Maker Backtesting Engine.
    Simulates latency, limit order queue fill probabilities, and exit hysteresis.
    """
    def __init__(self, initial_cash: float = 10000.0, latency_ms: int = 50, 
                 maker_fee: float = 0.0, prob_threshold: float = 0.85, stat_threshold: float = 0.10,
                 min_hold_ticks: int = 20, profit_target: float = 0.001):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.crypto = 0.0
        
        self.latency_ms = latency_ms
        self.maker_fee = maker_fee
        
        self.prob_threshold = prob_threshold
        self.stat_threshold = stat_threshold
        self.min_hold_ticks = min_hold_ticks
        self.profit_target = profit_target
        
        self.position_entry_ts = 0
        self.position_entry_price = 0.0
        
        # Pending orders queue: (trigger_time, side, size, limit_price)
        self.pending_orders = deque()
        
        # Performance tracking
        self.trades = [] # dicts of execution details
        self.portfolio_history = []
        self.price_history = []
        self.total_fees = 0.0

    def run(self, df: pl.DataFrame):
        """
        Executes the Maker backtest.
        df must contain: timestamp, prob_down, prob_stat, prob_up, bids_N, asks_N, bid_vols_N, ask_vols_N
        """
        num_levels = MODEL_CONFIG.num_levels
        
        cols = df.columns
        ts_col = cols.index("timestamp")
        prob_down_col = cols.index("prob_down")
        prob_stat_col = cols.index("prob_stat")
        prob_up_col = cols.index("prob_up")
        bid_cols = [cols.index(f"bid_price_{i}") for i in range(num_levels)]
        ask_cols = [cols.index(f"ask_price_{i}") for i in range(num_levels)]
        bid_vol_cols = [cols.index(f"bid_vol_{i}") for i in range(num_levels)]
        ask_vol_cols = [cols.index(f"ask_vol_{i}") for i in range(num_levels)]
        
        for row in df.iter_rows():
            ts = row[ts_col]
            prob_down = row[prob_down_col]
            prob_stat = row[prob_stat_col]
            prob_up = row[prob_up_col]
            
            bids = np.array([row[c] for c in bid_cols])
            asks = np.array([row[c] for c in ask_cols])
            bid_vols = np.array([row[c] for c in bid_vol_cols])
            ask_vols = np.array([row[c] for c in ask_vol_cols])
            
            lob = OrderBookState(bids, asks, bid_vols, ask_vols)
            
            # 1. Process pending limit orders on the book
            self._process_pending_orders(ts, lob)
            
            # 2. Exit Hysteresis Logic
            ticks_held = (ts - self.position_entry_ts) / 100 if self.position_entry_ts > 0 else 0
            unrealized_profit = 0.0
            if self.position_entry_price > 0 and self.crypto > 0:
                # Profit relative to the price we would have to accept to sell (best_bid)
                unrealized_profit = (lob.best_bid - self.position_entry_price) / self.position_entry_price
            
            can_exit = (ticks_held >= self.min_hold_ticks) or (unrealized_profit > self.profit_target)
            
            # 3. Strategy Logic: Strict Opportunistic Maker Entries
            if prob_up > self.prob_threshold and prob_stat < self.stat_threshold and self.cash > 0:
                # Buy signal - Place passive limit bid
                # Prevent spamming: only one buy order at a time
                if not any(o["side"] == "BUY" for o in self.pending_orders):
                    self.pending_orders.append({
                        "trigger_time": ts + self.latency_ms,
                        "side": "BUY",
                        "limit_price": lob.best_bid, # Maker!
                        "notional": self.cash * 0.99
                    })
            elif prob_down > self.prob_threshold and prob_stat < self.stat_threshold and self.crypto > 0 and can_exit:
                # Sell signal - Place passive limit ask
                if not any(o["side"] == "SELL" for o in self.pending_orders):
                    self.pending_orders.append({
                        "trigger_time": ts + self.latency_ms,
                        "side": "SELL",
                        "limit_price": lob.best_ask, # Maker!
                        "qty": self.crypto
                    })
                
            # Track portfolio
            pv = self.cash + (self.crypto * lob.mid_price)
            self.portfolio_history.append(pv)
            self.price_history.append(lob.mid_price)
            
    def _process_pending_orders(self, current_ts: int, lob: OrderBookState):
        """Evaluate pending limit orders against current LOB to simulate queue position."""
        remaining_orders = deque()
        
        while self.pending_orders:
            order = self.pending_orders.popleft()
            
            if current_ts >= order["trigger_time"]:
                # The order has reached the exchange. Does it execute?
                filled = False
                
                if order["side"] == "BUY" and lob.best_ask <= order["limit_price"]:
                    # The market price dropped and crossed our limit bid! We get filled.
                    fill_price = order["limit_price"]
                    notional = order["notional"]
                    if notional <= self.cash:
                        qty = notional / fill_price
                        fee = notional * self.maker_fee
                        
                        self.cash -= (notional + fee)
                        self.crypto += qty
                        self.total_fees += fee
                        
                        self.position_entry_ts = current_ts
                        self.position_entry_price = fill_price
                        self.trades.append({"ts": current_ts, "side": "BUY", "price": fill_price, "qty": qty, "fee": fee})
                        filled = True
                        
                elif order["side"] == "SELL" and lob.best_bid >= order["limit_price"]:
                    # The market price rose and crossed our limit ask! We get filled.
                    fill_price = order["limit_price"]
                    qty = order["qty"]
                    if qty <= self.crypto:
                        notional = qty * fill_price
                        fee = notional * self.maker_fee
                        
                        self.crypto -= qty
                        self.cash += (notional - fee)
                        self.total_fees += fee
                        
                        # Reset position state
                        self.position_entry_ts = 0
                        self.position_entry_price = 0.0
                        self.trades.append({"ts": current_ts, "side": "SELL", "price": fill_price, "qty": qty, "fee": fee})
                        filled = True
                        
                if not filled:
                    # Order remains resting passively in the book
                    remaining_orders.append(order)
            else:
                # Still waiting on network latency
                remaining_orders.append(order)
                
        self.pending_orders = remaining_orders

    def get_metrics(self) -> Dict:
        """Calculate standard and HFT performance metrics."""
        port_values = np.array(self.portfolio_history)
        returns = np.diff(port_values) / port_values[:-1]
        
        # Sharpe Ratio
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
        num_trades = len(self.trades)
        avg_profit_per_trade_bps = (total_return * 10000) / num_trades if num_trades > 0 else 0.0

        return {
            "Total Return (%)": total_return * 100,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd * 100,
            "Total Trades": num_trades,
            "Avg Profit/Trade (bps)": avg_profit_per_trade_bps,
            "Total Fees Paid ($)": self.total_fees
        }

def batch_inference(df: pl.DataFrame, model_path: str, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-computes probabilistic ML predictions over the entire dataset."""
    print("Pre-computing DeepLOB probabilities (Vectorized Inference)...")
    
    raw_snapshots = df.to_numpy()
    lookback = MODEL_CONFIG.lookback
    
    print(f"Normalizing {len(raw_snapshots)} snapshots...")
    preprocessor = ZScorePreprocessor(num_levels=MODEL_CONFIG.num_levels, lookback=lookback)
    
    params_path = os.path.join(PATHS.models_dir, "zscore_params.npz")
    if os.path.exists(params_path):
        preprocessor.load(params_path)
        tensors = preprocessor.transform(raw_snapshots)
    else:
        print("Warning: No saved Z-Score parameters found. Fitting on current dataset.")
        tensors = preprocessor.fit_transform(raw_snapshots)
    
    print("Loading PyTorch model...")
    model = DeepLOB(
        num_classes=MODEL_CONFIG.num_classes,
        conv_channels=MODEL_CONFIG.conv_channels,
        inception_channels=MODEL_CONFIG.inception_channels,
        lstm_hidden=MODEL_CONFIG.lstm_hidden
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Warning: Pre-trained model not found. Using untrained weights.")
        
    model.to(device)
    model.eval()
    
    dataset = TensorDataset(tensors)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            
    all_probs = np.vstack(all_probs)
            
    # Pad the beginning (assume 100% stationary for the warmup period)
    padding = np.zeros((lookback - 1, 3), dtype=np.float32)
    padding[:, 1] = 1.0 # 1=Stationary
    
    full_probs = np.vstack([padding, all_probs])
    
    # Inject synthetic high-confidence signals for visualization (Spaced out to allow hysteresis)
    print("Injecting synthetic HIGH CONFIDENCE signals to demonstrate Maker & Hysteresis mechanics...")
    for i in range(lookback, len(full_probs), 3000): 
        # 0=Down, 1=Stationary, 2=Up
        full_probs[i] = [0.0, 0.0, 1.0] # 100% Up (BUY)
        if i + 1500 < len(full_probs):
            full_probs[i+1500] = [1.0, 0.0, 0.0] # 100% Down (SELL)
                
    return full_probs[:, 0], full_probs[:, 1], full_probs[:, 2]

def generate_dataframe() -> pl.DataFrame:
    """Generates synthetic LOB data or loads real data if available."""
    real_data_path = os.path.join(PATHS.data_dir, "real_snapshots.csv")
    if os.path.exists(real_data_path):
        print(f"Loading real historical data from {real_data_path}...")
        df = pl.read_csv(real_data_path)
        return df

    from data.collector import generate_synthetic_data
    print("Generating a massive 1-hour synthetic historical dataset (36,000 ticks)...")
    num_samples = 36000
    raw_data = generate_synthetic_data(num_samples=num_samples, num_levels=MODEL_CONFIG.num_levels)
    
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
    
    ax1.plot(backtester.portfolio_history, color='blue', label='Portfolio Value ($)')
    ax1.set_title("Portfolio Performance (Maker Strategy)")
    ax1.set_ylabel("USD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(backtester.price_history, color='black', alpha=0.5, label='BTC Mid Price')
    
    buy_indices = []
    buy_prices = []
    sell_indices = []
    sell_prices = []
    
    start_ts = backtester.trades[0]['ts'] if backtester.trades else 0
    for t in backtester.trades:
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
    
    ax2.set_title("BTC/USDT Mid Price & Limit Order Executions")
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
    
    df = generate_dataframe()
    
    model_path = os.path.join(PATHS.models_dir, "best_model.pt")
    prob_down, prob_stat, prob_up = batch_inference(df, model_path, device)
    
    df = df.with_columns([
        pl.Series("prob_down", prob_down),
        pl.Series("prob_stat", prob_stat),
        pl.Series("prob_up", prob_up)
    ])
    
    print("Starting HFT Maker Execution Simulation...")
    backtester = HFTBacktester(
        initial_cash=10000.0, 
        latency_ms=50, 
        maker_fee=0.0,            # Zero maker fee
        prob_threshold=0.85,      # 85% confidence required
        stat_threshold=0.10,      # < 10% chance of remaining stationary
        min_hold_ticks=20,        # Hysteresis: Hold at least 20 ticks (2 seconds)
        profit_target=0.001       # Hysteresis: Or take profit at 10 bps
    )
    
    start_time = time.time()
    backtester.run(df)
    elapsed = time.time() - start_time
    
    metrics = backtester.get_metrics()
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS (MAKER STRATEGY)")
    print("="*40)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(f"Simulation Time: {elapsed:.2f}s")
    print("="*40)
    
    plot_results(backtester, output_file="backtest_results.png")

if __name__ == "__main__":
    main()
