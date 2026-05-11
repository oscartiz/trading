"""Per-coin gate calibration: pick entry_confirmation_bars / _short from history.

The asymmetric long/short confirmation gate in
:class:`strategies.configs.RegimeSwitchingConfig` is currently tuned by hand on
BTC. Other coins almost certainly have different bull/bear regime-persistence
characteristics, and walking the knobs manually per coin doesn't scale.

This tool does the legwork: fit the HMM on a coin's price history, compute the
smoothed posterior for every bar, and measure how long the smoothed posterior
sustains above ``entry_proba`` for each direction. We then suggest a
confirmation-bars value per side based on that distribution.

Usage::

    python tools/calibrate_gates.py --coin BTC
    python tools/calibrate_gates.py --coin ETH --start 2022-01-01 --end 2024-12-31
    python tools/calibrate_gates.py --coin SOL --entry-proba 0.80

The recommended values come from the streak-length distribution — by default we
pick the ``p25`` (so roughly the top 75% of streaks clear the gate). Override
with ``--quantile`` to be more or less selective.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running directly: `python tools/calibrate_gates.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np    # noqa: E402

from backtesting.data import fetch_price_history    # noqa: E402
from strategies.configs import RegimeSwitchingConfig    # noqa: E402
from strategies.regime import Regime, RegimeClassifier, log_returns_from_close    # noqa: E402


@dataclass
class StreakStats:
    """Summary of how often, and for how long, a regime stays above the gate."""
    regime: str                    # "bull" or "bear"
    n_streaks: int
    total_bars: int                # sum of streak lengths
    mean: float
    p25: int
    p50: int
    p75: int
    p90: int
    max: int
    streaks: list[int]             # raw streak lengths (kept for further analysis)


@dataclass
class CalibrationReport:
    coin: str
    period_start: datetime
    period_end: datetime
    entry_proba: float
    max_chop_proba: float
    bull: StreakStats
    bear: StreakStats
    recommended_long_bars: int
    recommended_short_bars: int
    state_means: tuple[float, float, float]    # bear, chop, bull


# --------------------------------------------------------------------------- #
#  Streak analysis                                                             #
# --------------------------------------------------------------------------- #


def find_streaks(mask: np.ndarray) -> list[int]:
    """Return the lengths of every run of True in ``mask``.

    Empty or all-False input returns ``[]``. A single True bar is a streak of 1.
    """
    if mask.size == 0:
        return []
    bool_mask = mask.astype(bool)
    # Pad with False on both ends so np.diff catches edges.
    padded = np.concatenate(([False], bool_mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [int(e - s) for s, e in zip(starts, ends)]


def summarise_streaks(regime: str, streaks: list[int]) -> StreakStats:
    """Quantiles of a streak distribution. Robust to empty / tiny inputs."""
    if not streaks:
        return StreakStats(
            regime=regime, n_streaks=0, total_bars=0, mean=0.0,
            p25=0, p50=0, p75=0, p90=0, max=0, streaks=[],
        )
    arr = np.asarray(streaks, dtype=np.int64)
    return StreakStats(
        regime=regime,
        n_streaks=int(arr.size),
        total_bars=int(arr.sum()),
        mean=float(arr.mean()),
        p25=int(np.quantile(arr, 0.25)),
        p50=int(np.quantile(arr, 0.50)),
        p75=int(np.quantile(arr, 0.75)),
        p90=int(np.quantile(arr, 0.90)),
        max=int(arr.max()),
        streaks=streaks,
    )


def candidate_mask(
    proba: np.ndarray,
    target: Regime,
    entry_proba: float,
    max_chop_proba: float,
) -> np.ndarray:
    """Bars where ``target`` would be the smoothed-mode entry candidate.

    Mirrors the long/short selection in :class:`RegimeSwitchingStrategy`
    (and ``backtesting.regime_engine``): the target's posterior clears
    ``entry_proba`` and chop is not dominant.
    """
    p_chop = proba[:, int(Regime.CHOP)]
    p_target = proba[:, int(target)]
    return (p_target >= entry_proba) & (p_chop <= max_chop_proba)


# --------------------------------------------------------------------------- #
#  End-to-end calibration                                                      #
# --------------------------------------------------------------------------- #


def calibrate(
    closes: np.ndarray,
    coin: str,
    period_start: datetime,
    period_end: datetime,
    cfg: RegimeSwitchingConfig,
    quantile: float = 0.25,
) -> CalibrationReport:
    """Fit the HMM on the supplied close series and analyse regime streaks.

    The ``quantile`` argument picks the recommended confirmation-bars value.
    The default 0.25 means we admit roughly the top 75% of streaks: tighter
    than the median (which would admit half) but not so tight that real
    regimes get filtered out.
    """
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if closes.size < cfg.train_window_bars + 2:
        raise ValueError(
            f"need > {cfg.train_window_bars + 1} closes to calibrate, got {closes.size}"
        )

    returns = log_returns_from_close(closes)
    classifier = RegimeClassifier(
        n_iter=cfg.hmm_max_iter,
        tol=cfg.hmm_tol,
        random_state=cfg.hmm_random_state,
    )
    classifier.fit(returns)
    proba = classifier.predict_proba(returns)        # shape (T, 3)
    means = classifier.state_means

    bull_mask = candidate_mask(proba, Regime.BULL, cfg.entry_proba, cfg.max_chop_proba)
    bear_mask = candidate_mask(proba, Regime.BEAR, cfg.entry_proba, cfg.max_chop_proba)

    bull_stats = summarise_streaks("bull", find_streaks(bull_mask))
    bear_stats = summarise_streaks("bear", find_streaks(bear_mask))

    rec_long = _recommend_bars(bull_stats, quantile)
    rec_short = _recommend_bars(bear_stats, quantile)

    return CalibrationReport(
        coin=coin,
        period_start=period_start,
        period_end=period_end,
        entry_proba=cfg.entry_proba,
        max_chop_proba=cfg.max_chop_proba,
        bull=bull_stats,
        bear=bear_stats,
        recommended_long_bars=rec_long,
        recommended_short_bars=rec_short,
        state_means=(float(means[0]), float(means[1]), float(means[2])),
    )


def _recommend_bars(stats: StreakStats, quantile: float) -> int:
    """Pick a confirmation-bars value from the streak distribution.

    A streak shorter than the recommendation will be filtered as a posterior
    spike; a streak longer will be admitted. Floor at 1 because zero would
    disable the gate entirely (any single qualifying bar enters).
    """
    if stats.n_streaks == 0:
        return 1
    arr = np.asarray(stats.streaks, dtype=np.int64)
    value = int(np.quantile(arr, quantile))
    return max(value, 1)


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #


def print_report(report: CalibrationReport) -> None:
    print("\n" + "=" * 72)
    print(f"  Gate calibration — {report.coin}")
    print("=" * 72)
    print(
        f"  Period   : {report.period_start:%Y-%m-%d} → {report.period_end:%Y-%m-%d}"
    )
    print(
        f"  Thresholds: entry_proba={report.entry_proba:.2f}  "
        f"max_chop_proba={report.max_chop_proba:.2f}"
    )
    print(
        f"  Regime µ  : bear={report.state_means[0]:+.5f}  "
        f"chop={report.state_means[1]:+.5f}  bull={report.state_means[2]:+.5f}"
    )
    print()
    for stats in (report.bull, report.bear):
        print(f"  {stats.regime.upper()} streaks (P ≥ {report.entry_proba:.2f}):")
        if stats.n_streaks == 0:
            print("    (no qualifying bars in this window)")
            continue
        print(
            f"    count={stats.n_streaks:>4d}  total_bars={stats.total_bars:>5d}  "
            f"mean={stats.mean:>5.1f}"
        )
        print(
            f"    p25={stats.p25:>3d}  p50={stats.p50:>3d}  "
            f"p75={stats.p75:>3d}  p90={stats.p90:>3d}  max={stats.max:>3d}"
        )
        print()
    print(
        f"  RECOMMENDED: entry_confirmation_bars        = {report.recommended_long_bars}"
    )
    print(
        f"               entry_confirmation_bars_short  = {report.recommended_short_bars}"
    )
    print("=" * 72 + "\n")


# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "Gate calibration",
    )
    parser.add_argument("--coin", default="BTC")
    parser.add_argument("--start", type=parse_date,
                        default=datetime.now(timezone.utc) - timedelta(days=730),
                        help="ISO date (UTC). Default: 2 years ago.")
    parser.add_argument("--end", type=parse_date,
                        default=datetime.now(timezone.utc),
                        help="ISO date (UTC). Default: today.")
    parser.add_argument("--entry-proba", type=float, default=None,
                        help="Override config entry_proba.")
    parser.add_argument("--max-chop-proba", type=float, default=None,
                        help="Override config max_chop_proba.")
    parser.add_argument("--quantile", type=float, default=0.25,
                        help="Streak-distribution quantile to use as the "
                             "recommendation (default 0.25 = admit ~top 75%%).")
    args = parser.parse_args()

    cfg = RegimeSwitchingConfig()
    if args.entry_proba is not None:
        cfg.entry_proba = args.entry_proba
    if args.max_chop_proba is not None:
        cfg.max_chop_proba = args.max_chop_proba

    prices_df = fetch_price_history(args.coin, args.start, args.end, interval=cfg.candle_interval)
    closes = prices_df["close"].to_numpy(dtype=np.float64)

    report = calibrate(
        closes=closes,
        coin=args.coin,
        period_start=args.start,
        period_end=args.end,
        cfg=cfg,
        quantile=args.quantile,
    )
    print_report(report)


if __name__ == "__main__":
    main()
