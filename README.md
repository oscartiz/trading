# DeepLOB HFT Quantitative Pipeline

A high-frequency quantitative trading framework for the Binance BTC-USDT spot market. This repository implements a complete pipeline from Level-2 Limit Order Book (LOB) data ingestion and feature engineering to deep neural network training, vectorized backtesting, and a low-latency Rust execution engine.

> **Note**: This environment is configured for simulated execution against live market feeds. No exchange credentials or API keys are required for local paper trading.

---

## System Architecture

The pipeline is partitioned into a Research & Machine Learning Environment (Python/PyTorch) and an Autonomous Execution Engine (Rust/Candle).

```text
┌────────────────────────────────┐         ┌───────────────────────────────┐
│     Research & ML (Python)     │         │    Execution Engine (Rust)    │
│                                │         │                               │
│  1. LOB Snapshot Collection    │         │  1. Async WebSocket Feed      │
│  2. DeepLOB Model (PyTorch)    │  weights│  2. Candle-Core Inference     │
│  3. Vectorized Backtesting     │────────▶│  3. Deterministic Matching    │
│  4. Feature Normalization      │ .safetensors 4. Live Portfolio Monitor  │
└────────────────────────────────┘         └───────────────────────────────┘
```

---

## 1. Research and Machine Learning Environment

The machine learning pipeline is designed for high-frequency price prediction using order book microstructure.

### Feature Extraction
- **Input Tensors**: The model ingest a sliding window of 100 LOB snapshots.
- **Dimensionality**: Each snapshot consists of 40 features (10 price/volume levels for both bid and ask sides).
- **Normalization**: Real-time Z-score scaling is applied using parameters derived strictly from the chronological training split to avoid look-ahead bias.

### Model Architecture (DeepLOB)
- **Spatial Filters**: 2D Convolutional layers extract spatial relationships between price and volume levels.
- **Inception Module**: Multi-scale convolutional kernels capture varied LOB dynamics.
- **Temporal Modeling**: Long Short-Term Memory (LSTM) layers process the temporal dependencies of the extracted features.
- **Classification**: A softmax output layer provides probabilities for three classes: Mid-price increase, Mid-price decrease, and Stationary.

### Vectorized Backtesting
- **Latency Emulation**: The matching engine simulates 50ms of network and exchange queuing latency.
- **Post-Trade Analysis**: Vectorized PnL calculation accounts for taker fees and slippage on every tick.
- **Precision vs Recall**: The system prioritizes high-confidence thresholds (85%+) to mitigate transaction cost erosion.

---

## 2. Autonomous Execution Engine

The Rust engine provides a zero-overhead environment for live inference and strategy execution.

### Low-Latency Inference (`candle-core`)
- Utilizes the `candle` framework for hardware-accelerated tensor operations on the Apple Silicon M4 CPU.
- Implements custom asymmetric padding and stride emulation to ensure 100% parity with the PyTorch-trained weights.
- Inference latency is optimized to sub-millisecond durations per snapshot.

### Strategy Implementation: DeepLOB Maker
- **Strict Maker Posture**: The strategy attempts to minimize fee bleed by placing resting orders at the best bid/ask.
- **Confidence Thresholding**: Execution is gated by an 85% probability threshold.
- **Hysteresis & Exit Logic**: Implements a minimum holding time of 20 ticks or a fixed profit target to prevent high-frequency churning.

### Concurrent Pipeline
- **Tokio Tasks**: The feed handler, strategy evaluator, and matching engine run as independent tasks communicating via MPSC channels.
- **Non-Blocking I/O**: The system maintains 100% throughput even during high-volatility bursts.

---

## Operational Guide

### Training and Research
To collect data and train the model weights:

```bash
cd ml
# 1. Collect live LOB snapshots
python data/ws_collector.py

# 2. Execute training loop and export .safetensors
python train.py

# 3. Validate strategy via vectorized backtest
python backtest.py
```

### Production Monitoring
To launch the autonomous execution engine:

```bash
# Compile and run the release binary
cargo run --release
```

The live monitor is accessible at [http://localhost:3030](http://localhost:3030). This dashboard provides real-time visualization of portfolio metrics, allocation dynamics, and the raw probability outputs from the DeepLOB model.

---

## Technical Specifications
- **Inference Backend**: Candle Core (Rust)
- **Training Backend**: PyTorch (Python)
- **Communication Protocol**: Binance Multiplexed WebSocket (aggTrade, depth20@100ms)
- **Hardware Optimization**: Apple Silicon M4 (AArch64)

## License
MIT
