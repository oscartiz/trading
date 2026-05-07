"""Tests for the paper-trading order manager."""
from __future__ import annotations

import asyncio

import pytest

from execution import PaperOrderManager, Side
from risk import RiskConfig, RiskManager
from runtime import equity_watchdog

from tests.conftest import FakeInfo


def _paper(starting_equity: float = 1000.0, mid: float = 100.0,
           fee_rate: float = 0.0, slippage_bps: float = 0.0) -> tuple[PaperOrderManager, FakeInfo]:
    info = FakeInfo(mid=mid)
    pm = PaperOrderManager(info, starting_equity=starting_equity, fee_rate=fee_rate, slippage_bps=slippage_bps)
    return pm, info


def test_open_long_records_position():
    pm, _ = _paper(mid=100.0)
    res = pm.market_order("BTC", Side.BUY, 0.5)
    assert res.success is True
    pos = pm.get_position("BTC")
    assert pos is not None
    assert float(pos["szi"]) == pytest.approx(0.5)
    assert float(pos["entryPx"]) == pytest.approx(100.0)


def test_open_short_uses_negative_szi():
    pm, _ = _paper(mid=100.0)
    pm.market_order("BTC", Side.SELL, 0.5)
    pos = pm.get_position("BTC")
    assert float(pos["szi"]) == pytest.approx(-0.5)


def test_get_position_returns_none_when_flat():
    pm, _ = _paper()
    assert pm.get_position("BTC") is None


def test_close_realises_profit():
    pm, info = _paper(mid=100.0)
    pm.market_order("BTC", Side.BUY, 1.0)        # long 1 BTC @ 100
    info.mid = 110.0
    pm.market_order("BTC", Side.SELL, 1.0)        # close at 110
    assert pm.get_position("BTC") is None
    assert pm.realised_pnl == pytest.approx(10.0)


def test_close_realises_loss_on_short():
    pm, info = _paper(mid=100.0)
    pm.market_order("BTC", Side.SELL, 1.0)        # short 1 BTC @ 100
    info.mid = 110.0
    pm.market_order("BTC", Side.BUY, 1.0)         # cover at 110 — loss of 10
    assert pm.realised_pnl == pytest.approx(-10.0)


def test_partial_close_realises_proportional_pnl():
    pm, info = _paper(mid=100.0)
    pm.market_order("BTC", Side.BUY, 2.0)        # long 2 BTC @ 100
    info.mid = 110.0
    pm.market_order("BTC", Side.SELL, 1.0)        # close half at 110 → +10
    pos = pm.get_position("BTC")
    assert float(pos["szi"]) == pytest.approx(1.0)
    assert float(pos["entryPx"]) == pytest.approx(100.0)   # entry price unchanged
    assert pm.realised_pnl == pytest.approx(10.0)


def test_flip_position_realises_and_reopens():
    pm, info = _paper(mid=100.0)
    pm.market_order("BTC", Side.BUY, 1.0)        # long 1 @ 100
    info.mid = 110.0
    pm.market_order("BTC", Side.SELL, 1.5)        # close 1 (+10), open short 0.5 @ 110
    pos = pm.get_position("BTC")
    assert float(pos["szi"]) == pytest.approx(-0.5)
    assert float(pos["entryPx"]) == pytest.approx(110.0)
    assert pm.realised_pnl == pytest.approx(10.0)


def test_adding_to_position_weights_entry():
    pm, info = _paper(mid=100.0)
    pm.market_order("BTC", Side.BUY, 1.0)        # long 1 @ 100
    info.mid = 200.0
    pm.market_order("BTC", Side.BUY, 1.0)        # long 1 more @ 200 → avg 150
    pos = pm.get_position("BTC")
    assert float(pos["szi"]) == pytest.approx(2.0)
    assert float(pos["entryPx"]) == pytest.approx(150.0)


def test_slippage_pushes_buy_price_up():
    pm, _ = _paper(mid=100.0, slippage_bps=10.0)   # 10bps = 0.1%
    pm.market_order("BTC", Side.BUY, 1.0)
    pos = pm.get_position("BTC")
    assert float(pos["entryPx"]) == pytest.approx(100.10)


def test_slippage_pushes_sell_price_down():
    pm, _ = _paper(mid=100.0, slippage_bps=10.0)
    pm.market_order("BTC", Side.SELL, 1.0)
    pos = pm.get_position("BTC")
    assert float(pos["entryPx"]) == pytest.approx(99.90)


def test_fees_reduce_realised_pnl():
    pm, info = _paper(mid=100.0, fee_rate=0.001)
    pm.market_order("BTC", Side.BUY, 1.0)        # buy fee = 0.1
    info.mid = 110.0
    pm.market_order("BTC", Side.SELL, 1.0)        # sell fee = 0.11; gross +10
    expected_fees = 100.0 * 0.001 + 110.0 * 0.001  # 0.21
    assert pm.fees_paid == pytest.approx(expected_fees)
    assert pm.realised_pnl == pytest.approx(10.0 - expected_fees)


def test_get_equity_includes_unrealised():
    pm, info = _paper(starting_equity=1000.0, mid=100.0)
    pm.market_order("BTC", Side.BUY, 1.0)        # long 1 @ 100
    assert pm.get_equity() == pytest.approx(1000.0)
    info.mid = 110.0
    assert pm.get_equity() == pytest.approx(1010.0)
    info.mid = 90.0
    assert pm.get_equity() == pytest.approx(990.0)


def test_get_equity_after_close_only_reflects_realised():
    pm, info = _paper(starting_equity=1000.0, mid=100.0)
    pm.market_order("BTC", Side.BUY, 1.0)
    info.mid = 110.0
    pm.market_order("BTC", Side.SELL, 1.0)
    info.mid = 200.0   # post-close mid changes shouldn't affect equity
    assert pm.get_equity() == pytest.approx(1010.0)


def test_market_order_rejected_when_no_mid():
    info = FakeInfo()
    info.mid = 0.0   # invalid
    pm = PaperOrderManager(info, starting_equity=1000.0)
    res = pm.market_order("BTC", Side.BUY, 1.0)
    assert res.success is False


def test_set_leverage_is_noop():
    pm, _ = _paper()
    pm.set_leverage("BTC", 5, is_cross=True)
    # Just confirms it doesn't raise — paper mode ignores leverage.


def test_paper_drives_drawdown_watchdog():
    """End-to-end: paper position takes a loss, watchdog reads paper equity,
    risk halts. This is the integration we'll rely on for week-long test runs."""
    info = FakeInfo(mid=100.0)
    pm = PaperOrderManager(info, starting_equity=100.0, fee_rate=0.0, slippage_bps=0.0)
    risk = RiskManager(RiskConfig(max_drawdown_pct=0.05))

    pm.market_order("BTC", Side.BUY, 1.0)   # long 1 @ 100, equity unchanged
    risk.update_equity(pm.get_equity())
    assert risk.is_halted() is False

    info.mid = 90.0   # 10% drop on a 1.0 BTC position = $10 loss on $100 equity = 10% drawdown
    risk.update_equity(pm.get_equity())
    assert risk.is_halted() is True


def test_paper_watchdog_loop_halts_on_loss():
    """Run the actual watchdog coroutine against a paper book that loses money."""
    info = FakeInfo(mid=100.0)
    pm = PaperOrderManager(info, starting_equity=100.0, fee_rate=0.0, slippage_bps=0.0)
    risk = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    pm.market_order("BTC", Side.BUY, 1.0)

    async def driver():
        task = asyncio.create_task(equity_watchdog(pm.get_equity, risk, poll_seconds=0.001))
        # First tick captures HWM = 100; flatten then move price down to trigger drawdown.
        await asyncio.sleep(0.005)
        info.mid = 80.0   # 20% drop on $100 equity → drawdown breached
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    assert risk.is_halted() is True
