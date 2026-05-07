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
    # High threshold: only enter when the model is confident about the regime,
    # which keeps trade count low and stop-loss hits near zero.
    entry_proba: float = 0.85
    # Require regime expected return to clear this absolute log-return threshold
    # so we don't trade marginally-positive bull regimes. Per-bar units.
    min_expected_return_per_bar: float = 1e-4
    # Don't enter if the chop probability dominates by this margin.
    max_chop_proba: float = 0.50

    # ---- exit ----
    # Exit if posterior of the held regime drops below this.
    exit_proba: float = 0.45
    # Hard stop loss on adverse price move (fraction). Wide on purpose —
    # the strategy is designed to ride a regime to its end, so we want
    # the regime change (not a price stop) to do the exiting.
    stop_loss_pct: float = 0.12
    # Take profit disabled by default — let the regime change be the exit
    # signal so we don't cap winning trends.
    take_profit_pct: float | None = None
    # Max hold time regardless of regime (bars). 60 days at 1h candles —
    # generous because the new defaults rarely trade and we want to ride
    # multi-month regime episodes to completion.
    max_hold_bars: int = 1440
    # Floor before regime-based exits can fire — stops still fire below this.
    # Suppresses whipsaw when the smoothed posterior wobbles right after entry.
    min_hold_bars: int = 72               # 3 days at 1h

    # ---- regime persistence ----
    # After exiting a regime, refuse to re-enter that same regime for N bars.
    # Forces a cool-off before opening another trade in the same direction.
    same_regime_cooldown_bars: int = 336   # 14 days at 1h
    # Require the candidate target regime to have been the (smoothed-posterior or
    # viterbi) entry signal for this many *consecutive* bars before opening.
    # Filters out posterior spikes that revert. 8h sustained signal at 1h candles.
    entry_confirmation_bars: int = 8
    # "smoothed" — drive decisions from the posterior P(state | x).
    # "viterbi" — drive decisions from the MAP state at the latest bar; this
    # yields one trade per Viterbi run of bull/bear bars but is slower to
    # react to early regime shifts. Smoothed lets us gate entries on a
    # confidence threshold (entry_proba), which is the lever we tune.
    signal_mode: str = "smoothed"

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
