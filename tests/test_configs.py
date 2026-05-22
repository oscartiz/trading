"""Tests for strategies.configs.RegimeSwitchingConfig friction helpers."""
from __future__ import annotations

from execution.fees import TAKER_FEE_RATE
from strategies.configs import RegimeSwitchingConfig


def test_derive_min_expected_return_covers_two_legs_plus_slippage():
    cfg = RegimeSwitchingConfig(
        min_expected_return_per_bar=None,
        expected_slippage_bps=2.0,
    )
    expected = 2.0 * TAKER_FEE_RATE + 2.0 / 10_000.0
    assert cfg.derive_min_expected_return() == expected


def test_derive_min_expected_return_uses_explicit_override():
    cfg = RegimeSwitchingConfig(min_expected_return_per_bar=0.005)
    assert cfg.derive_min_expected_return() == 0.005


def test_default_min_expected_return_is_friction_aware():
    """The default config (min_expected_return_per_bar=None) must derive a
    floor strictly greater than the per-trade fee load — otherwise the
    expected-return gate could let through trades that lose to fees alone."""
    cfg = RegimeSwitchingConfig()
    derived = cfg.derive_min_expected_return()
    assert derived > 2.0 * TAKER_FEE_RATE
