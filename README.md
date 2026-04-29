# Crypto Paper Trading Bot

A low-latency, async paper trading bot targeting Binance, built in Rust. Designed for long-term accumulation with zero leverage and multiple layers of risk protection.

> **Paper trading only** — all trades are simulated locally. No Binance account or API keys required.

---

## Architecture

The system is built on three decoupled pillars communicating exclusively through async MPSC channels. The WebSocket feed never blocks the strategy thread, and the strategy never touches account state directly.

```
┌─────────────────────┐         ┌──────────────────────┐
│   Binance WebSocket  │  ticks  │   DCA Accumulator     │
│   @aggTrade stream   │────────▶│   Strategy            │
│                      │──┐      │                       │
└─────────────────────┘  │      └───────────┬───────────┘
                          │                  │ orders
                          │ ticks            ▼
                   ┌──────▼──────────────────────────────┐
                   │         Paper Engine                  │
                   │   (virtual balances, slippage, fees)  │
                   └──────────────────┬───────────────────┘
                                      │ execution reports
                                      ▼
                               back to Strategy
```

| Pillar | Responsibility |
|--------|----------------|
| **Market Data Feed** | Maintains a resilient WebSocket connection to Binance with automatic reconnection and exponential backoff. Deserializes the raw `@aggTrade` stream into normalized ticks. |
| **Paper Engine** | Acts as a local matching engine. Simulates order fills with configurable slippage and taker fees. Manages virtual USDT and crypto balances. All math uses `rust_decimal` for deterministic base-10 calculations. |
| **Strategy** | Consumes ticks, computes indicators, and emits orders through an `ExecutionClient` trait. The strategy is entirely unaware of whether it's talking to the paper engine or a live exchange. |

---

## Quick Start

To run the bot and monitor the live dashboard simultaneously:

1. **Start the Bot**: Run the following command in your terminal:
   ```bash
   cargo run --release
   ```
2. **Open the Dashboard**: Once the bot is running, open your browser and go to:
   [http://localhost:3030](http://localhost:3030)

> [!TIP]
> Keep the terminal window and browser side-by-side. The terminal will show the raw execution logs and strategy decisions, while the dashboard provides a high-level visual overview of your portfolio and RSI trends.

---

## Strategy: DCA Accumulator

The bot implements a conservative **Dollar-Cost Averaging** strategy designed for long-term holding with minimal risk. The core philosophy is simple: *time in the market beats timing the market.*

### How It Works

The strategy buys small, fixed amounts of BTC at regular intervals — but only when market conditions are favorable. Every potential buy passes through a cascade of filters:

1. **SMA-200 Trend Filter** — Only buy when the current price is above the 200-period simple moving average. This confirms we're in a long-term uptrend and avoids catching falling knives.

2. **RSI Filter** — Skip buys entirely when the 14-period RSI exceeds 70 (overbought). When RSI drops below 30 (oversold), the buy size is increased by 1.5× to take advantage of dips.

3. **Risk Manager Approval** — Every order must pass through six independent guardrails before it can be submitted (see below).

### Sell Behavior

The strategy has exactly two sell triggers — one offensive, one defensive:

| Trigger | Condition | What It Sells |
|---------|-----------|---------------|
| **RSI Profit Exit** | RSI > 80 | Only the *profit portion* — keeps cost basis invested |
| **Trailing Stop-Loss** | Price drops 12% from peak | Entire position — full exit to protect capital |

#### RSI Profit-Taking — How It Works

The bot tracks your **average entry price** (cost basis) across all DCA buys. When RSI exceeds 80 (exceedingly overbought), it calculates how much of your position represents unrealized profit and sells only that:

**Example:** You've DCA'd $500 total into 0.005 BTC (avg entry: $100k). Price runs to $120k and RSI hits 82:

```
Position value:     0.005 × $120k = $600
Cost basis at $120k: $500 / $120k  = 0.00417 BTC  ← kept
Profit portion:     0.005 - 0.00417 = 0.00083 BTC  ← sold
```

You bank ~$100 in profit and keep 0.00417 BTC — your original $500 of value stays invested. The cost basis accounting scales proportionally on partial sells, so repeated RSI exits remain accurate.

### What It Will NOT Do

- **No short selling** — sells are only defensive (stop-loss) or profit-taking (RSI exit)
- **No leverage** — order size is always capped to available cash
- **No chasing** — SMA and RSI filters prevent buying at peaks
- **No over-concentration** — hard 50% portfolio allocation ceiling

---

## Risk Management

Every trade must pass through the risk manager, which enforces six independent guardrails. Each one can independently block a trade:

| Guard | Default | What It Does |
|-------|---------|--------------|
| **Max Allocation** | 50% | Never hold more than 50% of total portfolio value in crypto |
| **Per-Trade Limit** | $200 | Caps any single DCA buy regardless of other factors |
| **Drawdown Circuit Breaker** | −15% | Halts ALL buying if portfolio drops 15% from its high-water mark. Auto-resets on new all-time highs. |
| **Cooldown** | 5 min | Minimum interval between consecutive trades to prevent overtrading |
| **Volatility Scaling** | 2.5% / 5% | Halves position size when volatility exceeds 2.5%. Suspends buying entirely above 5%. |
| **Trailing Stop** | −12% | Sells the entire position if price drops 12% from its peak |

---

## Technical Indicators

All indicators are computed using `rust_decimal` for deterministic, floating-point-free calculations:

- **RSI (Relative Strength Index)** — 14-period, Wilder smoothing method. Used to filter overbought/oversold conditions.
- **SMA (Simple Moving Average)** — 200-period. Used as a long-term trend confirmation filter.
- **Volatility (MAD)** — Mean Absolute Deviation of returns over a 20-period window. Used for position sizing adjustments.

---

## Project Structure

```
src/
├── main.rs                  # Runtime init, channel plumbing, task spawning
├── config.rs                # Runtime configuration (symbol, fees, slippage)
├── types.rs                 # Shared channel message types
├── error.rs                 # Unified error types
├── feed/
│   ├── mod.rs
│   └── binance.rs           # WebSocket feed with reconnection
├── engine/
│   ├── mod.rs
│   ├── account.rs           # Virtual balance tracking
│   └── matching.rs          # Paper matching engine
└── strategy/
    ├── mod.rs
    ├── traits.rs            # ExecutionClient trait + channel impl
    ├── indicators.rs        # RSI, SMA, volatility (pure math)
    ├── risk.rs              # Risk manager (6 guardrails)
    └── dca_accumulator.rs   # DCA strategy implementation
```

---

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (1.75+)
- Internet connection (for the Binance WebSocket stream)
- No Binance account needed

### Run

```bash
# Default (info-level logging)
cargo run --release

# Verbose (see every indicator computation)
RUST_LOG=debug cargo run --release
```

The bot will:
1. Connect to Binance's public WebSocket stream
2. Warm up the 200-tick indicator buffer (~5–15 minutes depending on volume)
3. Begin evaluating DCA entry signals
4. Log every trade decision, fill, and risk gate activation

Stop with `Ctrl+C`.

### Configuration

Edit `src/config.rs` for runtime settings:

```rust
initial_balance: dec!(10_000),   // Starting virtual USDT
symbol: "btcusdt".into(),        // Trading pair
fee_rate: dec!(0.001),           // Simulated taker fee (10 bps)
slippage_bps: dec!(0.0005),      // Simulated slippage (5 bps)
```

Edit `src/strategy/risk.rs` → `RiskParams::default()` for risk parameters.

### US Users

If `stream.binance.com` is geo-blocked, change the WebSocket URL in `config.rs`:

```rust
ws_url: "wss://stream.binance.us:9443/ws".into(),
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `tokio` | Async runtime with full features |
| `tokio-tungstenite` | WebSocket client for Binance stream |
| `serde` / `serde_json` | Stream deserialization |
| `rust_decimal` | Deterministic base-10 financial math |
| `tracing` | Structured, leveled logging |
| `async-trait` | Async trait support for `ExecutionClient` |
| `thiserror` / `anyhow` | Error handling |

---

---

## Machine Learning Pipeline (DeepLOB)

The project includes a state-of-the-art Python machine learning pipeline (`ml/`) designed to train a Deep Limit Order Book (DeepLOB) model. This model predicts high-frequency mid-price movements (Up, Down, Stationary) directly from raw level-2 market data.

### Architecture Features
- **CNN + Inception + LSTM**: Uses 1D/2D convolutions to extract spatial patterns from the order book, an Inception module for multi-scale feature extraction, and an LSTM to model temporal sequence dependencies.
- **Smoothed Forward-Rolling Window Labeling**: Avoids micro-structure noise by comparing current prices to a rolling average of future horizons, classifying moves based on a dynamically adaptable volatility threshold.
- **Relative Spread Normalization**: Normalizes prices relative to the mid-price to preserve spread dynamics, and scales volumes as a fraction of total visible depth.
- **Rust-Ready Export**: The pipeline automatically exports trained weights to a `deeplob.safetensors` file along with a `model_config.json`, making it trivial to load into the Rust `candle-core` inference engine.

To run the training pipeline (using synthetic data for demonstration):
```bash
cd ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

---

## Future Roadmap

- **ML Inference** — The `ml/` directory contains a PyTorch implementation of the DeepLOB (Deep Convolutional Neural Networks for Limit Order Books) architecture. This pipeline trains on high-frequency snapshot data and exports its weights to `.safetensors`. The upcoming goal is to replace the RSI/SMA filters with `candle-core` forward passes on this model.
- **Live Trading** — Implement `ExecutionClient` against the Binance REST API. The strategy code changes zero lines.
- **Multi-Symbol** — DCA into a basket (BTC + ETH) with per-asset allocation limits.
- **Persistence** — Log fills to SQLite for post-session P&L analysis and backtesting.

---

## License

MIT
