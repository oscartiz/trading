"""Pure data config classes for strategies — no execution dependencies."""
from dataclasses import dataclass

from execution.fees import TAKER_FEE_RATE


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
    # EM is local-optimum sensitive — refit with N seeded restarts and keep
    # the one with the highest log-likelihood. Cheap insurance against
    # landing on a bad mode for a week. n_seeds=1 reproduces the old behaviour.
    hmm_n_seeds: int = 5
    # Minimum acceptable bull-µ minus bear-µ. Below this the regimes have
    # collapsed onto each other and gates downstream will pass junk; we warn
    # via the notifier so the operator can investigate.
    min_regime_separation: float = 1e-4
    # If a refit lands while a position is open and the held-regime mean
    # has drifted by more than this fraction of the original mean, alert.
    # Catches the silent-failure mode where a refit lands on a different
    # mode and the strategy is now holding into a different distribution
    # than it entered on.
    regime_drift_alert_pct: float = 0.5

    # ---- entry ----
    # Take a long when P(bull) ≥ this; short when P(bear) ≥ this.
    # High threshold: only enter when the model is confident about the regime,
    # which keeps trade count low and stop-loss hits near zero.
    entry_proba: float = 0.85
    # Require regime expected return to clear an absolute log-return threshold
    # so we don't trade marginally-positive bull regimes. When None, the
    # threshold is derived from frictions: 2 * TAKER_FEE_RATE + slippage
    # (so the strategy only trades when the model expects to clear costs).
    # Set explicitly to a float to bypass the friction-aware derivation.
    min_expected_return_per_bar: float | None = None
    # Expected one-way slippage when sizing the friction-aware gate. 1 bp by
    # default — matches PaperOrderManager's slippage_bps default.
    expected_slippage_bps: float = 1.0
    # Don't enter if the chop probability dominates by this margin.
    max_chop_proba: float = 0.50

    # ---- exit ----
    # Exit if posterior of the held regime drops below this.
    exit_proba: float = 0.45
    # Soft-exit: reduce the position by ``staged_exit_fraction`` when the
    # held-regime posterior first drops below this (between exit_proba and
    # entry_proba). When None, staged exits are disabled and the strategy
    # exits in one shot at ``exit_proba``.
    soft_exit_proba: float | None = 0.60
    # Fraction of the original position to close on a soft-exit trigger.
    # 0.5 keeps half the exposure for the regime-recovery case while
    # banking gains / cutting risk on the rest.
    staged_exit_fraction: float = 0.5
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
    same_regime_cooldown_bars: int = 72   # 3 days at 1h
    # Require the candidate target regime to have been the (smoothed-posterior or
    # viterbi) entry signal for this many *consecutive* bars before opening.
    # Filters out posterior spikes that revert. Acts as the long-side gate.
    entry_confirmation_bars: int = 6
    # Optional override for the short side. If None, shorts inherit
    # `entry_confirmation_bars`. Bear regimes are spikier than bull regimes in
    # crypto — they rarely sustain long enough to clear the long-side gate, so
    # a smaller value here unlocks short entries without lowering long quality.
    entry_confirmation_bars_short: int | None = 3
    # "smoothed" — drive decisions from the posterior P(state | x).
    # "viterbi" — drive decisions from the MAP state at the latest bar; this
    # yields one trade per Viterbi run of bull/bear bars but is slower to
    # react to early regime shifts. Smoothed lets us gate entries on a
    # confidence threshold (entry_proba), which is the lever we tune.
    signal_mode: str = "smoothed"

    # ---- sizing ----
    # Notional USD when fully sized. Used as a fallback when position_size_pct
    # is None (e.g. in backtests, which don't have a live equity series).
    position_size_usd: float = 100.0
    # Optional: size as a fraction of live equity. When set, the strategy reads
    # equity via its injected equity_source and uses notional = equity * pct.
    # Falls back to position_size_usd when the source is unavailable (paper or
    # backtest contexts without an equity feed).
    position_size_pct: float | None = None
    # Scale notional by P(target regime); below this floor we skip.
    min_size_scale: float = 0.5
    # Volatility-targeting: when set, the base notional is scaled by
    # vol_target / expected_vol so per-trade risk normalises across regimes.
    # Capped between min_vol_scale and max_vol_scale so a very quiet regime
    # can't ask for absurd notional. None disables vol scaling.
    vol_target_per_bar: float | None = None
    min_vol_scale: float = 0.5
    max_vol_scale: float = 2.0
    leverage: int = 1
    is_cross: bool = True

    # ---- ops / live loop ----
    # How often (seconds) to poll for a new bar in live trading.
    poll_interval_seconds: int = 60 * 5
    # Escalate to ERROR (and so to the notifier) when this many consecutive
    # candle fetches raise. At 5-min polling this is ~30 min of API outage.
    candle_failure_alert_threshold: int = 6
    # Periodic ERROR when we haven't received a closed bar in this many
    # seconds — catches the "API is responding but returning empty" mode.
    # 3h covers a missed bar plus reasonable polling jitter at 1h candles.
    stale_bar_alert_seconds: int = 3 * 3600

    # ---- friction helper ----
    def derive_min_expected_return(self) -> float:
        """Friction-aware floor: covers two-leg fees + expected slippage."""
        if self.min_expected_return_per_bar is not None:
            return float(self.min_expected_return_per_bar)
        slippage = self.expected_slippage_bps / 10_000.0
        return 2.0 * TAKER_FEE_RATE + slippage
