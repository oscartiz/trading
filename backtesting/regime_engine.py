"""Backtest engine for the 3-state regime-switching strategy.

Walk-forward design:
    - At t=0 we require `train_window_bars` of history to fit the initial HMM.
    - On each subsequent bar we compute the smoothed posterior of the latest
      bar against the rolling window of returns.
    - The HMM is refit every `refit_every_bars` bars to keep parameters fresh
      while avoiding the cost of refitting every bar.
    - Trade decisions follow the same logic as the live strategy.

Inputs:
    prices_df : DataFrame[timestamp, open, high, low, close, volume]
    config    : RegimeSwitchingConfig
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from execution import TAKER_FEE_RATE
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime, RegimeClassifier, log_returns_from_close

Side = Literal["long", "short"]


@dataclass
class RegimeTrade:
    entry_time: datetime
    exit_time: datetime
    side: Side
    entry_price: float
    exit_price: float
    position_size_usd: float
    entry_regime: str
    entry_proba: float
    exit_reason: str
    fees: float

    @property
    def price_pnl(self) -> float:
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (self.exit_price - self.entry_price) / self.entry_price * self.position_size_usd

    @property
    def total_pnl(self) -> float:
        return self.price_pnl - self.fees

    @property
    def hold_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class RegimeBacktestResult:
    trades: list[RegimeTrade]
    equity_curve: pd.Series
    regimes: pd.DataFrame                 # timestamp-indexed: regime, p_bear, p_chop, p_bull
    config: RegimeSwitchingConfig
    coin: str
    start: datetime
    end: datetime
    n_refits: int = 0
    debug: dict = field(default_factory=dict)

    @property
    def total_fees(self) -> float:
        return sum(t.fees for t in self.trades)

    @property
    def gross_pnl(self) -> float:
        return sum(t.price_pnl for t in self.trades)

    @property
    def net_pnl(self) -> float:
        return sum(t.total_pnl for t in self.trades)


def run_regime_backtest(
    prices_df: pd.DataFrame,
    coin: str,
    config: RegimeSwitchingConfig | None = None,
    fee_rate: float = TAKER_FEE_RATE,
) -> RegimeBacktestResult:
    cfg = config or RegimeSwitchingConfig()
    df = prices_df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if "close" not in df.columns:
        raise ValueError("prices_df must include a 'close' column")

    closes = df["close"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64) if "high" in df.columns else closes
    lows = df["low"].to_numpy(dtype=np.float64) if "low" in df.columns else closes
    timestamps = df["timestamp"].tolist()
    n = len(df)

    classifier = RegimeClassifier(
        n_iter=cfg.hmm_max_iter,
        tol=cfg.hmm_tol,
        random_state=cfg.hmm_random_state,
    )

    if n <= cfg.train_window_bars + 1:
        raise ValueError(
            f"Not enough bars to backtest: have {n}, need > {cfg.train_window_bars + 1}"
        )

    trades: list[RegimeTrade] = []
    equity_pts: list[tuple[datetime, float]] = []
    regime_rows: list[dict] = []

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_ts: datetime = timestamps[0]
    side: Side = "long"
    target_regime: Regime = Regime.CHOP
    bars_since_fit = 0
    n_refits = 0
    last_exit_regime: Regime | None = None
    last_exit_idx: int = -10**9
    # Tracks how many consecutive bars the same regime has been the entry
    # candidate. Resets when the candidate flips or disappears.
    streak_target: Regime | None = None
    streak_bars: int = 0

    # Initial fit on the first window
    init_returns = log_returns_from_close(closes[: cfg.train_window_bars + 1])
    classifier.fit(init_returns)
    n_refits += 1

    realised_pnl = 0.0

    for t in range(cfg.train_window_bars, n):
        bars_since_fit += 1
        if bars_since_fit >= cfg.refit_every_bars:
            window_start = max(0, t - cfg.train_window_bars)
            returns = log_returns_from_close(closes[window_start : t + 1])
            classifier.fit(returns)
            bars_since_fit = 0
            n_refits += 1

        window_start = max(0, t - cfg.train_window_bars)
        returns = log_returns_from_close(closes[window_start : t + 1])
        if returns.size < 5:
            continue
        snap = classifier.snapshot(returns)
        viterbi_label: Regime | None = None
        if cfg.signal_mode == "viterbi":
            viterbi_label = Regime(int(classifier.predict(returns)[-1]))
        ts = timestamps[t]
        close = closes[t]

        regime_rows.append({
            "timestamp": ts,
            "regime": (viterbi_label.label if viterbi_label is not None else snap.label),
            "p_bear": float(snap.proba[0]),
            "p_chop": float(snap.proba[1]),
            "p_bull": float(snap.proba[2]),
            "expected_return": snap.expected_return,
        })

        # ---- exit handling ----
        if in_position:
            held_bars = t - entry_idx
            direction = 1.0 if side == "long" else -1.0
            adverse = lows[t] if side == "long" else highs[t]
            adverse_pnl_pct = direction * (adverse - entry_price) / entry_price
            favourable = highs[t] if side == "long" else lows[t]
            favourable_pnl_pct = direction * (favourable - entry_price) / entry_price

            reason = ""
            exit_price = close
            # Hard risk exits fire regardless of min_hold_bars.
            if adverse_pnl_pct < -cfg.stop_loss_pct:
                reason = "stop_loss"
                exit_price = entry_price * (1 - cfg.stop_loss_pct) if side == "long" \
                    else entry_price * (1 + cfg.stop_loss_pct)
            elif cfg.take_profit_pct is not None and favourable_pnl_pct > cfg.take_profit_pct:
                reason = "take_profit"
                exit_price = entry_price * (1 + cfg.take_profit_pct) if side == "long" \
                    else entry_price * (1 - cfg.take_profit_pct)
            elif held_bars >= cfg.min_hold_bars:
                if cfg.signal_mode == "viterbi":
                    if viterbi_label is not None and viterbi_label != target_regime:
                        reason = "regime_flipped"
                else:
                    held_proba = float(snap.proba[int(target_regime)])
                    opposite = Regime.BEAR if target_regime == Regime.BULL else Regime.BULL
                    opposite_proba = float(snap.proba[int(opposite)])
                    if held_proba < cfg.exit_proba:
                        reason = "regime_weakened"
                    elif opposite_proba >= cfg.entry_proba:
                        reason = "regime_flipped"
                if not reason and held_bars >= cfg.max_hold_bars:
                    reason = "max_hold"

            if reason:
                fees = 2 * fee_rate * cfg.position_size_usd
                trade = RegimeTrade(
                    entry_time=entry_ts,
                    exit_time=ts,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size_usd=cfg.position_size_usd,
                    entry_regime=target_regime.label,
                    entry_proba=float(snap.proba[int(target_regime)]),  # current proba at exit time
                    exit_reason=reason,
                    fees=fees,
                )
                trades.append(trade)
                realised_pnl += trade.total_pnl
                in_position = False
                last_exit_regime = target_regime
                last_exit_idx = t

        # ---- entry handling (new entries are gated on the same bar after exits) ----
        # First compute the candidate target ignoring streak/cooldown so we can
        # update the per-bar streak counter even when we're already in position.
        candidate: Regime | None = None
        if cfg.signal_mode == "viterbi":
            if viterbi_label == Regime.BULL and snap.expected_return >= cfg.min_expected_return_per_bar:
                candidate = Regime.BULL
            elif viterbi_label == Regime.BEAR and snap.expected_return <= -cfg.min_expected_return_per_bar:
                candidate = Regime.BEAR
        elif snap.proba[1] <= cfg.max_chop_proba:
            if (snap.proba[2] >= cfg.entry_proba
                    and snap.expected_return >= cfg.min_expected_return_per_bar):
                candidate = Regime.BULL
            elif (snap.proba[0] >= cfg.entry_proba
                    and snap.expected_return <= -cfg.min_expected_return_per_bar):
                candidate = Regime.BEAR

        if candidate is not None and candidate == streak_target:
            streak_bars += 1
        elif candidate is not None:
            streak_target = candidate
            streak_bars = 1
        else:
            streak_target = None
            streak_bars = 0

        if not in_position:
            target: Regime | None = candidate

            # Confirmation: target must have been the candidate for ≥ N consecutive bars.
            # Short side may use a smaller gate than long.
            if target is not None:
                required = cfg.entry_confirmation_bars
                if target == Regime.BEAR and cfg.entry_confirmation_bars_short is not None:
                    required = cfg.entry_confirmation_bars_short
                if streak_bars < required:
                    target = None

            # Same-regime cooldown: don't re-enter the regime we just exited.
            if (target is not None
                    and last_exit_regime == target
                    and (t - last_exit_idx) < cfg.same_regime_cooldown_bars):
                target = None

            if target is not None:
                in_position = True
                side = "long" if target == Regime.BULL else "short"
                target_regime = target
                entry_price = close
                entry_idx = t
                entry_ts = ts

        # ---- mark-to-market equity ----
        unrealised = 0.0
        if in_position:
            direction = 1.0 if side == "long" else -1.0
            unrealised = direction * (close - entry_price) / entry_price * cfg.position_size_usd
        equity_pts.append((ts, realised_pnl + unrealised))

    # Close any open position at the end
    if in_position:
        final_close = float(closes[-1])
        fees = 2 * fee_rate * cfg.position_size_usd
        trades.append(RegimeTrade(
            entry_time=entry_ts,
            exit_time=timestamps[-1],
            side=side,
            entry_price=entry_price,
            exit_price=final_close,
            position_size_usd=cfg.position_size_usd,
            entry_regime=target_regime.label,
            entry_proba=0.0,
            exit_reason="backtest_end",
            fees=fees,
        ))

    equity_curve = pd.Series(
        [p for _, p in equity_pts],
        index=pd.DatetimeIndex([t for t, _ in equity_pts]),
        name="pnl",
    )
    regimes_df = pd.DataFrame(regime_rows).set_index("timestamp") if regime_rows else pd.DataFrame()

    start_dt = timestamps[0] if isinstance(timestamps[0], datetime) else timestamps[0].to_pydatetime()
    end_dt = timestamps[-1] if isinstance(timestamps[-1], datetime) else timestamps[-1].to_pydatetime()

    return RegimeBacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        regimes=regimes_df,
        config=cfg,
        coin=coin,
        start=start_dt,
        end=end_dt,
        n_refits=n_refits,
    )


def regime_metrics(result: RegimeBacktestResult) -> dict:
    """Mirror of backtesting.metrics.compute_metrics, adapted for regime trades."""
    import math

    trades = result.trades
    equity = result.equity_curve
    if not trades:
        return {"error": "no trades executed"}

    pnls = [t.total_pnl for t in trades]
    n = len(trades)
    win_rate = sum(1 for p in pnls if p > 0) / n
    cummax = equity.cummax()
    drawdown = equity - cummax
    max_dd_usd = float(drawdown.min())
    # Express drawdown relative to the capital deployed per trade so the figure
    # stays interpretable when realised P&L is small (peak P&L can be near zero).
    max_dd_pct = (max_dd_usd / result.config.position_size_usd * 100)
    delta = equity.diff().dropna()
    std = float(delta.std())
    sharpe = (float(delta.mean()) / std * math.sqrt(8760)) if std > 0 else 0.0
    reasons: dict[str, int] = {}
    sides_count = {"long": 0, "short": 0}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
        sides_count[t.side] += 1
    duration_days = (result.end - result.start).total_seconds() / 86400
    return {
        "coin": result.coin,
        "period_days": round(duration_days, 1),
        "n_refits": result.n_refits,
        "n_trades": n,
        "longs": sides_count["long"],
        "shorts": sides_count["short"],
        "total_pnl_usd": round(sum(pnls), 4),
        "gross_pnl_usd": round(result.gross_pnl, 4),
        "total_fees_usd": round(result.total_fees, 4),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_pnl_per_trade_usd": round(sum(pnls) / n, 4),
        "best_trade_usd": round(max(pnls), 4),
        "worst_trade_usd": round(min(pnls), 4),
        "avg_hold_hours": round(sum(t.hold_hours for t in trades) / n, 1),
        "max_drawdown_usd": round(max_dd_usd, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 3),
        "exit_reasons": reasons,
    }


def print_regime_metrics(result: RegimeBacktestResult) -> None:
    m = regime_metrics(result)
    cfg = result.config
    if "error" in m:
        print(f"\nRegime backtest complete — {m['error']}\n")
        return

    print("\n" + "=" * 60)
    print(f"  Regime-Switching Backtest — {m['coin']}")
    print("=" * 60)
    print(f"  Period       : {result.start:%Y-%m-%d} → {result.end:%Y-%m-%d}  ({m['period_days']} days)")
    print(f"  HMM window   : {cfg.train_window_bars} bars   refits: {m['n_refits']}")
    print(f"  Signal mode  : {cfg.signal_mode}")
    print(f"  Entry P      : ≥{cfg.entry_proba:.2f}    Exit P: <{cfg.exit_proba:.2f}")
    print(f"  Stop / TP    : {cfg.stop_loss_pct:.0%} / "
          f"{(f'{cfg.take_profit_pct:.0%}' if cfg.take_profit_pct else 'off')}")
    short_conf = (cfg.entry_confirmation_bars_short
                  if cfg.entry_confirmation_bars_short is not None
                  else cfg.entry_confirmation_bars)
    print(f"  Hold bounds  : min={cfg.min_hold_bars}  max={cfg.max_hold_bars}  cooldown={cfg.same_regime_cooldown_bars}  confirm={cfg.entry_confirmation_bars}/{short_conf} (long/short)")
    print(f"  Position     : ${cfg.position_size_usd}")
    print()
    print(f"  Trades       : {m['n_trades']}   (long={m['longs']} short={m['shorts']})")
    print(f"  Win rate     : {m['win_rate_pct']}%")
    print(f"  Avg hold     : {m['avg_hold_hours']}h")
    print()
    print(f"  Gross P&L    : ${m['gross_pnl_usd']:+.4f}")
    print(f"  Fees paid    : ${m['total_fees_usd']:.4f}")
    print(f"  Net P&L      : ${m['total_pnl_usd']:+.4f}")
    print(f"  Per trade    : ${m['avg_pnl_per_trade_usd']:+.4f}")
    print(f"  Best / Worst : ${m['best_trade_usd']:+.4f}  /  ${m['worst_trade_usd']:+.4f}")
    print()
    print(f"  Max drawdown : ${m['max_drawdown_usd']:.4f}  ({m['max_drawdown_pct']:.2f}% of position size)")
    print(f"  Sharpe ratio : {m['sharpe_ratio']:.3f}")
    print()
    reasons_str = "  ".join(f"{k}={v}" for k, v in m["exit_reasons"].items())
    print(f"  Exit reasons : {reasons_str}")
    print("=" * 60 + "\n")
