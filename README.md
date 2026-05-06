# trading

Algorithmic trading bot for [Hyperliquid](https://hyperliquid.xyz) perpetual futures, written in Python.

## Strategies

### Funding Rate
Collects funding payments by sitting on the receiving side when rates are extreme.

- **Positive funding** → opens short (longs pay us hourly)
- **Negative funding** → opens long (shorts pay us hourly)

Exits when funding normalises, flips, a 2% stop loss fires, or the 48h time cap is hit. No price prediction — purely a carry trade on crowded positioning.

Default parameters (conservative, small account):

| Parameter | Value | Notes |
|---|---|---|
| Position size | $50 notional | |
| Leverage | 1x cross | |
| Entry threshold | 0.02%/hr | ≈175%/yr annualised |
| Exit threshold | 0.005%/hr | rate no longer worth holding |
| Stop loss | 2% | hard directional cut |
| Max hold | 48h | time cap regardless of funding |
| Poll interval | 10 min | REST API, not WebSocket |

### Regime Switching
Fits a 3-state Gaussian Hidden Markov Model (HMM) to hourly log returns and labels every bar as **bear**, **chop**, or **bull** by sorting the latent states on emission mean. Trades the smoothed posterior:

- **P(bull) ≥ 0.65** and chop not dominant → opens long
- **P(bear) ≥ 0.65** and chop not dominant → opens short
- Exits when the held regime's posterior drops below 0.45, the opposite regime takes over, a 3% stop fires, a 6% take-profit fills, or the 240-bar (10-day) time cap is hit

The HMM is refit on a rolling 3000-bar window every `refit_every_bars` (weekly by default). Implementation is pure NumPy (`strategies/hmm.py` for forward-backward / Viterbi / Baum-Welch, `strategies/regime.py` for the 3-state wrapper). See `strategies/configs.py:RegimeSwitchingConfig` for the full parameter set.

## Backtest results

### Funding rate — BTC 2024 | 12 trades | Net P&L: -$6.63

![Funding BTC 2024](data/charts/backtest_BTC_20240101_20250101.png)

The three panels show: BTC price with labelled long/short entries and profitable/loss exits; the hourly funding rate against entry (0.02%/hr) and exit (0.005%/hr) thresholds; and cumulative USD P&L over the year. The strategy captured several high-funding episodes but gave back gains during the flat mid-year period, finishing slightly negative at $50 notional / 1× leverage — consistent with a conservative parameter set on a year where funding was often below threshold.

### Regime switching — BTC 2024 | 315 trades | Net P&L: -$2.48

![Regime BTC 2024](data/charts/regime_BTC_20240101_20250101.png)

Run with `python regime_backtest.py --coin BTC --start 2024-01-01 --end 2025-01-01 --refit-every 336 --chart`. The three panels show: BTC price with regime-shaded background (red=bear, grey=chop, green=bull) and labelled trade entries/exits; the rolling smoothed posteriors P(bear), P(chop), P(bull) with the entry/exit thresholds; and cumulative USD P&L. With the default thresholds (0.65/0.45) the strategy overtrades — 298 of 315 exits fire on `regime_weakened`, indicating the band is being whipsawed by smoothing noise. Win rate 44.8%, Sharpe -0.116, max drawdown 25% of position size. A wider entry/exit band, a minimum-hold-bars floor, or switching from smoothed to filtered posteriors at the live edge are the natural next levers to pull.

## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/<you>/trading
cd trading

python -m venv .venv && source .venv/bin/activate

pip install \
  "hyperliquid-python-sdk>=0.9.0" \
  "pandas>=2.2.0" "numpy>=1.26.0" \
  "python-dotenv>=1.0.0" "loguru>=0.7.0" \
  "websockets>=12.0" "aiohttp>=3.9.0" \
  "eth-account>=0.10.0"
```

## Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
HL_PRIVATE_KEY=0x...          # wallet private key
HL_ACCOUNT_ADDRESS=0x...      # wallet address
HL_TESTNET=true               # set false for mainnet
LOG_LEVEL=INFO
```

> **Never commit `.env`.** It is in `.gitignore`.

## Running

```bash
source .venv/bin/activate
PYTHONPATH=. python main.py --strategy funding --coin BTC
PYTHONPATH=. python main.py --strategy regime  --coin BTC
```

Logs go to stderr and `logs/trading.log`. Example output:

```
INFO | Hyperliquid clients ready | testnet=True
INFO | funding_rate | BTC started | entry≥0.0200%/hr exit≤0.0050%/hr stop=2% size=$50
INFO | BTC | funding=+0.02341%/hr (+205.1%/yr) mid=65432.00 in_position=False
INFO | BTC | entering Sell | funding=+0.02341%/hr size=0.000764 @ 65432.00
INFO | BTC | funding=+0.01100%/hr (+96.4%/yr) mid=65210.00 in_position=True
INFO | BTC | exiting | reasons: funding normalised (+0.00412%/hr)
```

## Project structure

```
trading/
├── config/           # env-based settings
├── data/             # async WebSocket feed (trades + L2 book)
├── execution/        # Hyperliquid client, order manager
├── risk/             # position limits, drawdown guard
├── strategies/
│   ├── base.py               # abstract Strategy base class
│   ├── funding_rate.py       # funding rate carry strategy
│   ├── regime_switching.py   # HMM-based 3-state regime strategy
│   ├── regime.py             # 3-state classifier wrapping the HMM
│   ├── hmm.py                # numpy-only Gaussian HMM
│   ├── configs.py            # strategy parameter dataclasses
│   └── example_momentum.py   # stub example
├── backtesting/
│   ├── engine.py             # funding-rate backtest engine
│   ├── regime_engine.py      # regime-switching backtest engine
│   ├── charts.py             # funding-rate chart
│   ├── regime_charts.py      # regime-switching chart
│   ├── data.py               # cached price + funding history
│   └── metrics.py            # backtest metrics
├── backtest.py       # CLI: funding-rate backtest
├── regime_backtest.py # CLI: regime-switching backtest
├── research/         # Jupyter notebooks
├── tests/
└── main.py           # entry point (--strategy funding|regime)
```

## Writing a new strategy

Subclass `Strategy` and override `run()` for polling strategies or `on_trade`/`on_book` for event-driven ones:

```python
from strategies.base import Strategy

class MyStrategy(Strategy):
    def name(self) -> str:
        return "my_strategy"

    async def run(self) -> None:
        while True:
            # your logic here
            await asyncio.sleep(60)
```

Wire it up in `main.py` alongside the existing strategies.

## Risk management

All orders pass through `RiskManager` before execution:

- Per-order USD cap
- Max total position USD
- Drawdown halt (shuts off trading if equity drops past threshold)

Defaults in `main.py` — adjust to match your account size.

## Disclaimer

This is experimental software. Crypto perpetuals carry liquidation risk. Use testnet first, keep sizes small, and never risk more than you can afford to lose.
