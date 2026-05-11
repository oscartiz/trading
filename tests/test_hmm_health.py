"""Tests for HMM health telemetry on refit."""
from __future__ import annotations

import numpy as np

from risk import RiskConfig, RiskManager
from runtime import StateStore, TradeJournal
from strategies.configs import RegimeSwitchingConfig
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager, regime_prices


def _build(tmp_path, cfg: RegimeSwitchingConfig | None = None):
    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    state = StateStore("regime_switching", "BTC", root=tmp_path)
    journal = TradeJournal("BTC", root=tmp_path)
    s = RegimeSwitchingStrategy(   # type: ignore[arg-type]
        "BTC", om, risk,
        cfg or RegimeSwitchingConfig(),
        state_store=state, journal=journal,
    )
    return s


def _seed_closes_from_regime_prices(s: RegimeSwitchingStrategy, n_warmup=80) -> None:
    """Populate _closes with a well-separated regime price series."""
    df = regime_prices(n_warmup=n_warmup, bull_bars=60, bear_bars=60, chop_bars=20, seed=0)
    for c in df["close"].tolist():
        s._closes.append(float(c))


# --------------------------------------------------------------------------- #
#  Classifier exposes log_likelihood + n_iter                                  #
# --------------------------------------------------------------------------- #


def test_classifier_exposes_log_likelihood_and_iters(tmp_path):
    s = _build(tmp_path)
    _seed_closes_from_regime_prices(s)
    s._refit()

    assert s._classifier.log_likelihood is not None
    assert np.isfinite(s._classifier.log_likelihood)
    assert s._classifier.n_iter_run >= 1


# --------------------------------------------------------------------------- #
#  Health dict populated on successful refit                                   #
# --------------------------------------------------------------------------- #


def test_refit_populates_health_dict(tmp_path):
    s = _build(tmp_path)
    _seed_closes_from_regime_prices(s)
    s._refit()

    h = s._last_fit_health
    assert h is not None
    assert h["ok"] is True
    assert "log_likelihood" in h and np.isfinite(h["log_likelihood"])
    assert h["n_iter"] >= 1
    assert h["fit_ms"] >= 0.0
    assert h["bear_mu"] <= h["chop_mu"] <= h["bull_mu"]   # sorted by classifier
    assert h["separation"] == h["bull_mu"] - h["bear_mu"]
    assert h["n_returns"] > 0


# --------------------------------------------------------------------------- #
#  Degraded separation warning                                                 #
# --------------------------------------------------------------------------- #


def test_degraded_separation_warning_does_not_raise(tmp_path):
    """A flat-return series collapses the regimes; refit must still complete
    successfully but record the degraded separation in health and (we just
    sanity-check that the run didn't crash)."""
    cfg = RegimeSwitchingConfig(min_regime_separation=1e-2)   # absurdly high
    s = _build(tmp_path, cfg)
    _seed_closes_from_regime_prices(s)
    s._refit()
    # With min_separation=1e-2 (1% per bar — much larger than the synthetic
    # series produces), health must still report ok=True, and the operator
    # gets warned via the WARNING-level log line.
    h = s._last_fit_health
    assert h is not None and h["ok"] is True
    assert h["separation"] < 1e-2


# --------------------------------------------------------------------------- #
#  Fit failure path                                                            #
# --------------------------------------------------------------------------- #


def test_refit_failure_is_caught_and_recorded(tmp_path):
    """Classifier.fit raising must NOT propagate; previous params survive."""
    s = _build(tmp_path)
    _seed_closes_from_regime_prices(s)
    s._refit()                                # successful first fit
    prior_means = s._classifier.state_means.copy()

    # Monkeypatch the classifier's fit() to raise the next time.
    def _boom(_returns):
        raise RuntimeError("singular matrix")
    s._classifier.fit = _boom    # type: ignore[assignment]

    s._refit()    # must not raise

    h = s._last_fit_health
    assert h is not None
    assert h["ok"] is False
    assert "singular" in h["error"]
    # Previous fit's params remain — inference can keep running on stale weights.
    np.testing.assert_array_equal(s._classifier.state_means, prior_means)
