"""Pure data config classes for strategies — no execution dependencies."""
from dataclasses import dataclass


@dataclass
class FundingConfig:
    # Entry: only trade when annualised rate exceeds this (0.02%/hr = ~175%/yr)
    entry_threshold: float = 0.0002
    # Exit: close when rate drops below this (0.005%/hr — no longer worth holding)
    exit_threshold: float = 0.00005
    # Hard stop: close if position moves this much against us
    stop_loss_pct: float = 0.02
    # Hard time cap — never hold longer than this regardless of funding
    max_hold_hours: int = 48
    # Notional size in USD — kept small and conservative
    position_size_usd: float = 50.0
    # How often to poll the REST API (live only)
    poll_interval_seconds: int = 600


@dataclass
class RegimeSwitchingConfig:
    """Settings for the 3-state regime-switching strategy.

    Returns flow as: fetch hourly closes → log returns → fit 3-state Gaussian
    HMM → use the latest smoothed posterior to drive long/short/flat decisions.
    """
    # ---- model fit ----
    candle_interval: str = "1h"
    # Rolling window of bars used to fit the HMM (3000 hours ≈ 125 days)
    train_window_bars: int = 3000
    # Refit the model every N bars; in between we just run inference.
    refit_every_bars: int = 168           # weekly when interval=1h
    hmm_max_iter: int = 100
    hmm_tol: float = 1e-4
    hmm_random_state: int = 0

    # ---- entry ----
    # Take a long when P(bull) ≥ this; short when P(bear) ≥ this.
    entry_proba: float = 0.65
    # Require regime expected return to clear this absolute log-return threshold
    # so we don't trade marginally-positive bull regimes. Per-bar units.
    min_expected_return_per_bar: float = 1e-4
    # Don't enter if the chop probability dominates by this margin.
    max_chop_proba: float = 0.50

    # ---- exit ----
    # Exit if posterior of the held regime drops below this.
    exit_proba: float = 0.45
    # Hard stop loss on adverse price move (fraction).
    stop_loss_pct: float = 0.03
    # Optional take profit (set to None to disable).
    take_profit_pct: float | None = 0.06
    # Max hold time regardless of regime (bars).
    max_hold_bars: int = 240              # 10 days at 1h

    # ---- sizing ----
    # Notional USD when fully sized.
    position_size_usd: float = 100.0
    # Scale notional by P(target regime); below this floor we skip.
    min_size_scale: float = 0.5
    leverage: int = 1
    is_cross: bool = True

    # ---- live loop ----
    # How often (seconds) to poll for a new bar in live trading.
    poll_interval_seconds: int = 60 * 5
