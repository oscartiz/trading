"""Tests for tools/calibrate_gates.py — per-coin gate calibration."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

# tools/ isn't a package, so import it directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import calibrate_gates as cg    # noqa: E402

from strategies.configs import RegimeSwitchingConfig    # noqa: E402
from strategies.regime import Regime    # noqa: E402
from tests.conftest import regime_prices    # noqa: E402


# --------------------------------------------------------------------------- #
#  find_streaks                                                                #
# --------------------------------------------------------------------------- #


def test_find_streaks_empty():
    assert cg.find_streaks(np.array([], dtype=bool)) == []


def test_find_streaks_all_false():
    assert cg.find_streaks(np.zeros(10, dtype=bool)) == []


def test_find_streaks_all_true():
    assert cg.find_streaks(np.ones(7, dtype=bool)) == [7]


def test_find_streaks_mixed():
    # F T T F T F T T T F
    mask = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert cg.find_streaks(mask) == [2, 1, 3]


def test_find_streaks_edges():
    # Streaks that start at the first bar and end at the last bar must be caught.
    mask = np.array([1, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert cg.find_streaks(mask) == [2, 3]


# --------------------------------------------------------------------------- #
#  summarise_streaks                                                           #
# --------------------------------------------------------------------------- #


def test_summarise_streaks_empty():
    s = cg.summarise_streaks("bull", [])
    assert s.regime == "bull"
    assert s.n_streaks == 0
    assert s.total_bars == 0
    assert s.mean == 0.0
    assert s.p25 == s.p50 == s.p75 == s.p90 == s.max == 0


def test_summarise_streaks_quantiles():
    streaks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    s = cg.summarise_streaks("bull", streaks)
    assert s.n_streaks == 10
    assert s.total_bars == 55
    assert s.mean == 5.5
    assert s.p50 == int(np.quantile(streaks, 0.50))
    assert s.p25 == int(np.quantile(streaks, 0.25))
    assert s.max == 10


# --------------------------------------------------------------------------- #
#  candidate_mask                                                              #
# --------------------------------------------------------------------------- #


def test_candidate_mask_requires_both_target_high_and_chop_low():
    # Construct a (T, 3) proba matrix with rows: [p_bear, p_chop, p_bull]
    proba = np.array([
        [0.05, 0.05, 0.90],     # bull qualifies (P_bull>=0.85, P_chop<=0.50)
        [0.05, 0.60, 0.35],     # bull does NOT qualify (chop dominates)
        [0.10, 0.05, 0.85],     # bull qualifies (boundary)
        [0.40, 0.20, 0.40],     # bull does NOT qualify (P_bull < 0.85)
    ])
    mask = cg.candidate_mask(proba, Regime.BULL, entry_proba=0.85, max_chop_proba=0.50)
    assert mask.tolist() == [True, False, True, False]

    bear_mask = cg.candidate_mask(proba, Regime.BEAR, entry_proba=0.85, max_chop_proba=0.50)
    # No bear bars meet the 0.85 threshold here.
    assert bear_mask.tolist() == [False, False, False, False]


# --------------------------------------------------------------------------- #
#  _recommend_bars                                                             #
# --------------------------------------------------------------------------- #


def test_recommend_bars_returns_quantile_value():
    streaks = [1, 2, 3, 4, 5, 10, 20]
    stats = cg.summarise_streaks("bull", streaks)
    rec = cg._recommend_bars(stats, quantile=0.50)
    assert rec == int(np.quantile(streaks, 0.50))


def test_recommend_bars_floors_at_one():
    """A quantile of 0 (or all-1 streaks) should not return zero."""
    stats = cg.summarise_streaks("bull", [1, 1, 1])
    assert cg._recommend_bars(stats, quantile=0.0) == 1


def test_recommend_bars_zero_streaks_returns_one():
    stats = cg.summarise_streaks("bear", [])
    assert cg._recommend_bars(stats, quantile=0.25) == 1


# --------------------------------------------------------------------------- #
#  calibrate — end-to-end on synthetic regime data                            #
# --------------------------------------------------------------------------- #


def test_calibrate_rejects_short_history():
    cfg = RegimeSwitchingConfig(train_window_bars=1000)
    closes = np.linspace(100.0, 110.0, 50)
    with pytest.raises(ValueError, match="need >"):
        cg.calibrate(
            closes=closes,
            coin="BTC",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            cfg=cfg,
        )


def test_calibrate_rejects_invalid_quantile():
    cfg = RegimeSwitchingConfig(train_window_bars=50)
    closes = np.linspace(100.0, 110.0, 200)
    with pytest.raises(ValueError, match="quantile"):
        cg.calibrate(
            closes=closes, coin="BTC",
            period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            period_end=datetime(2024, 6, 1, tzinfo=timezone.utc),
            cfg=cfg, quantile=1.5,
        )


def test_calibrate_produces_recommendations_on_real_regimes():
    """Synthetic series with strong bull and bear sequences must produce a non-empty distribution."""
    df = regime_prices(
        n_warmup=80, bull_bars=300, bear_bars=300, chop_bars=80, seed=42,
    )
    closes = df["close"].to_numpy(dtype=np.float64)
    cfg = RegimeSwitchingConfig(train_window_bars=80, entry_proba=0.85)

    report = cg.calibrate(
        closes=closes, coin="BTC",
        period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        cfg=cfg, quantile=0.25,
    )

    # Bull and bear should both fire at least one qualifying streak.
    assert report.bull.n_streaks >= 1
    assert report.bear.n_streaks >= 1
    # Recommendations are bounded by the longest streak in their direction.
    assert 1 <= report.recommended_long_bars <= report.bull.max
    assert 1 <= report.recommended_short_bars <= report.bear.max
    # Sanity: state means are ordered bear < chop < bull (sort-by-mean is the
    # whole point of the wrapper).
    bear_mu, chop_mu, bull_mu = report.state_means
    assert bear_mu < chop_mu < bull_mu


def test_calibrate_recommendation_is_quantile_of_streaks():
    """Pin: recommended_long_bars must come from the bull-streak distribution at the requested quantile."""
    df = regime_prices(
        n_warmup=80, bull_bars=400, bear_bars=200, chop_bars=60, seed=7,
    )
    closes = df["close"].to_numpy(dtype=np.float64)
    cfg = RegimeSwitchingConfig(train_window_bars=80, entry_proba=0.85)
    report = cg.calibrate(
        closes=closes, coin="BTC",
        period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        cfg=cfg, quantile=0.50,
    )
    expected = max(int(np.quantile(report.bull.streaks, 0.50)), 1)
    assert report.recommended_long_bars == expected


# --------------------------------------------------------------------------- #
#  print_report — smoke test that all sections render                         #
# --------------------------------------------------------------------------- #


def test_print_report_renders_recommendations(capsys):
    df = regime_prices(
        n_warmup=80, bull_bars=300, bear_bars=300, chop_bars=80, seed=2,
    )
    closes = df["close"].to_numpy(dtype=np.float64)
    cfg = RegimeSwitchingConfig(train_window_bars=80, entry_proba=0.85)
    report = cg.calibrate(
        closes=closes, coin="ETH",
        period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 12, 31, tzinfo=timezone.utc),
        cfg=cfg, quantile=0.25,
    )
    cg.print_report(report)
    out = capsys.readouterr().out
    assert "ETH" in out
    assert "BULL" in out and "BEAR" in out
    assert "entry_confirmation_bars" in out
    assert "RECOMMENDED" in out


def test_print_report_handles_zero_streaks(capsys):
    """If no bars qualified we should still render cleanly, not crash."""
    bear = cg.summarise_streaks("bear", [])
    bull = cg.summarise_streaks("bull", [])
    report = cg.CalibrationReport(
        coin="XYZ",
        period_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        period_end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        entry_proba=0.85, max_chop_proba=0.50,
        bull=bull, bear=bear,
        recommended_long_bars=1, recommended_short_bars=1,
        state_means=(-1e-4, 0.0, 1e-4),
    )
    cg.print_report(report)
    out = capsys.readouterr().out
    assert "no qualifying bars" in out


def test_parse_date_returns_utc():
    dt = cg.parse_date("2024-03-15")
    assert dt.year == 2024 and dt.month == 3 and dt.day == 15
    assert dt.tzinfo == timezone.utc
