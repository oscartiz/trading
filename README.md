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
Fits a 3-state Gaussian Hidden Markov Model (HMM) to hourly log returns and labels every bar as **bear**, **chop**, or **bull** by sorting the latent states on emission mean. Designed to **ride a regime to its end**: enter only when the model is highly confident, then hold until the regime itself changes — the regime exit, not a fixed take-profit, is what closes winning trades.

- **P(bull) ≥ 0.85** and chop not dominant → opens long
- **P(bear) ≥ 0.85** and chop not dominant → opens short
- Holds for at least `min_hold_bars` (3 days) so the smoothed posterior can't whipsaw a fresh entry
- Exits when the held regime's posterior drops below 0.45, the opposite regime takes over, the wide 12% stop fires, or the 30-day time cap is hit — there is **no fixed take-profit**
- After exit, refuses to re-enter the *same* regime for `same_regime_cooldown_bars` (7 days) so we don't immediately re-open the trade we just closed

The HMM is refit on a rolling 3000-bar window every `refit_every_bars` (weekly by default). Implementation is pure NumPy (`strategies/hmm.py` for forward-backward / Viterbi / Baum-Welch, `strategies/regime.py` for the 3-state wrapper). See `strategies/configs.py:RegimeSwitchingConfig` for the full parameter set, and `tools/regime_sweep.py` for the parameter-sweep harness used to pick the defaults.

## Backtest results

### Funding rate — BTC 2024 | 12 trades | Net P&L: -$6.63

![Funding BTC 2024](data/charts/backtest_BTC_20240101_20250101.png)

The three panels show: BTC price with labelled long/short entries and profitable/loss exits; the hourly funding rate against entry (0.02%/hr) and exit (0.005%/hr) thresholds; and cumulative USD P&L over the year. The strategy captured several high-funding episodes but gave back gains during the flat mid-year period, finishing slightly negative at $50 notional / 1× leverage — consistent with a conservative parameter set on a year where funding was often below threshold.

### Regime switching — BTC 2024 | 28 trades | Net P&L: +$30.16 | Sharpe 1.49

![Regime BTC 2024](data/charts/regime_BTC_20240101_20250101.png)

Run with `python regime_backtest.py --coin BTC --start 2024-01-01 --end 2025-01-01 --refit-every 336 --chart`. The three panels show: BTC price with regime-shaded background (red=bear, grey=chop, green=bull) and labelled trade entries/exits; the rolling smoothed posteriors P(bear), P(chop), P(bull) with the entry/exit thresholds; and cumulative USD P&L.

The first iteration of this strategy used `entry ≥ 0.65 / exit < 0.45 / 3% stop / 6% TP / 240-bar max hold` and lost money on 315 trades (-$2.48, Sharpe -0.12). 298 of those 315 exits fired on `regime_weakened`, meaning the smoothed posterior was wobbling across the 0.45 threshold and stopping us out before any regime had time to play out. Three changes fixed it:

1. **High-confidence entry** — `P ≥ 0.85` instead of `0.65`. The market spends a small fraction of time in a high-conviction regime, and entries during that fraction have real edge; entries below that threshold are noise.
2. **Wide stop, no take-profit** — 12% stop and `take_profit_pct=None`. The intent is to ride the regime to its actual end, so the regime change (not a price cap) does the exiting. With the 0.85 entry gate, the wide stop is rarely tested — zero stop-outs across all 28 trades on this period.
3. **Persistence guards** — `min_hold_bars=72` (3 days) before any regime-based exit fires, plus `same_regime_cooldown_bars=168` (7 days) preventing re-entry into the same regime we just left. Together these turn a single regime episode into a single trade.

Result on BTC 2024: **28 trades (14 long / 14 short), 57.1% win rate, 74.4h avg hold, +$30.16 net at $100 notional / 1×, Sharpe 1.49, max drawdown -17% of position size, 27 of 28 exits via `regime_weakened`** (the remaining one was the open trade closed at the end of the backtest). The defaults in `RegimeSwitchingConfig` were picked by `tools/regime_sweep.py`, which compares 9 configurations across the same period; see that script for the alternatives considered. **Caveats:** this is a single-period in-sample fit on a single coin — running the same sweep on ETH and on 2022/2023 BTC is the obvious next step before trusting these defaults out-of-sample.

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
├── tools/
│   └── regime_sweep.py # parameter-sweep harness for the regime strategy
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
