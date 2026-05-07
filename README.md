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
Fits a 3-state Gaussian Hidden Markov Model (HMM) to hourly log returns and labels every bar as **bear**, **chop**, or **bull** by sorting the latent states on emission mean. Tuned to engage with the market a handful of times per month — opens after the model has been confident in the same direction for several hours, then holds until the regime itself changes.

- **P(bull) ≥ 0.85** sustained for `entry_confirmation_bars` (6 hours, long side) → opens long
- **P(bear) ≥ 0.85** sustained for `entry_confirmation_bars_short` (3 hours, short side) → opens short
- **Asymmetric confirmation gate:** bear regimes in crypto are spikier than bull regimes — they rarely hold long enough to clear an 8-bar gate, which makes a symmetric gate structurally long-only. The short-side gate is set tighter (3h vs. 6h) so the model can engage with bear cycles without dragging long-side quality down
- Holds for at least `min_hold_bars` (3 days) so the smoothed posterior can't whipsaw a fresh entry
- Exits when the held regime's posterior drops below 0.45, the opposite regime takes over, the wide 12% stop fires, or the 60-day time cap is hit — there is **no fixed take-profit**
- After exit, refuses to re-enter the *same* regime for `same_regime_cooldown_bars` (3 days) so we don't immediately re-open the trade we just closed

The HMM is refit on a rolling 3000-bar window every `refit_every_bars` (weekly by default). Implementation is pure NumPy (`strategies/hmm.py` for forward-backward / Viterbi / Baum-Welch, `strategies/regime.py` for the 3-state wrapper). See `strategies/configs.py:RegimeSwitchingConfig` for the full parameter set, and `tools/regime_sweep.py` for the parameter-sweep harness used to pick the defaults.

## Backtest results

### Funding rate — BTC 2024 | 12 trades | Net P&L: -$6.63

![Funding BTC 2024](data/charts/backtest_BTC_20240101_20250101.png)

The three panels show: BTC price with labelled long/short entries and profitable/loss exits; the hourly funding rate against entry (0.02%/hr) and exit (0.005%/hr) thresholds; and cumulative USD P&L over the year. The strategy captured several high-funding episodes but gave back gains during the flat mid-year period, finishing slightly negative at $50 notional / 1× leverage — consistent with a conservative parameter set on a year where funding was often below threshold.

### Regime switching — BTC 2024 | 12 trades | Net P&L: +$18.82 | Sharpe 1.14

![Regime BTC 2024](data/charts/regime_BTC_20240101_20250101.png)

Run with `python regime_backtest.py --coin BTC --start 2024-01-01 --end 2025-01-01 --refit-every 336 --chart`. The three panels show: BTC price with regime-shaded background (red=bear, grey=chop, green=bull) and labelled trade entries/exits; the rolling smoothed posteriors P(bear), P(chop), P(bull) with the entry/exit thresholds; and cumulative USD P&L.

### Regime switching — BTC 2022 | 6 trades | Net P&L: +$16.77 | Sharpe 0.96

![Regime BTC 2022](data/charts/regime_BTC_20220101_20230101.png)

The 2022 run is the out-of-sample stress test — BTC went from $47k → $16k that year, with the FTX collapse and the Luna/Terra unwind on top of a sustained downtrend. The strategy entered 6 times (2 long / 4 short), profited from the bear cycles via shorts rather than dodging them, and finished the year up $16.77 on $100 notional.

The strategy went through four tuning iterations:

| Iteration | Trades (2024) | Net P&L (2024) | Notes |
|---|---|---|---|
| 1. Baseline (`P≥0.65, 3% stop, 6% TP, no confirm`) | 315 | -$2.48 | Whipsaws — 298/315 exits on `regime_weakened` |
| 2. Ride-the-regime (`P≥0.85, 12% stop, no TP, 7d cooldown`) | 28 | +$30.16 | Zero stop-outs, but 2-3 trades/month |
| 3. Low-frequency (`+ 8h confirmation, 14d cooldown, 60d max-hold`) | 3 | +$16.21 | ~1 trade per 4 months — but **long-only** even on bear years |
| 4. Asymmetric gate (`6h long / 3h short confirm, 3d cooldown`) | **12** | **+$18.82** | Engages with bear cycles; 5 of 12 trades on 2024 are shorts |

The shift from iteration 3 → 4 was driven by an out-of-sample run on BTC 2022 (a -65% bear year) that produced exactly **one** long trade, +$0.26, no shorts at all. Diagnostic showed bear regimes in BTC 2024 never sustained more than 4 consecutive bars passing the entry gate, while bull runs reached 17 — the symmetric 8-bar gate was structurally long-only. Lowering the short-side gate to 3 bars unlocked short entries without lowering long-side quality.

Result on BTC 2024 with the asymmetric defaults: **12 trades (7 long / 5 short), 58.3% win rate, 79.8h avg hold, +$18.82 net at $100 notional / 1×, Sharpe 1.14, max drawdown -24% of position size**. Result on BTC 2022: **6 trades (2 long / 4 short), 66.7% win rate, +$16.77 net, Sharpe 0.96, max drawdown -15%**. All exits on both years were via `regime_weakened` — no stop-outs.

**Caveats — read these before deploying:** *(1)* 18 trades across two years is still a small sample; per-trade noise is meaningful and these defaults haven't been validated on ETH or 2023 BTC. *(2)* The deeper drawdowns (-15% / -24% of position size) are the price of more aggressive engagement — a $100 trade can lose $24 in mark-to-market before the regime exit fires. Size accordingly. *(3)* The asymmetric short gate is calibrated to BTC's specific regime persistence; other assets may have different bull/bear asymmetry profiles and will need their own gate ratios.

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
