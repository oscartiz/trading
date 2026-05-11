"""Tests for runtime.journal.TradeJournal and the regime-strategy integration."""
from __future__ import annotations

import asyncio
import json

import numpy as np

from execution import Side
from risk import RiskConfig, RiskManager
from runtime import StateStore, TradeJournal
from strategies.configs import RegimeSwitchingConfig
from strategies.regime import Regime, RegimeSnapshot
from strategies.regime_switching import RegimeSwitchingStrategy

from tests.conftest import FakeInfo, FakeOrderManager, FakeOrderResult


# --------------------------------------------------------------------------- #
#  TradeJournal                                                                #
# --------------------------------------------------------------------------- #


def test_journal_creates_file_and_appends(tmp_path):
    j = TradeJournal("BTC", root=tmp_path)
    j.record({"coin": "BTC", "event": "entry", "size": 0.001})
    j.record({"coin": "BTC", "event": "exit", "size": 0.001})
    rows = j.read_all()
    assert [r["event"] for r in rows] == ["entry", "exit"]
    assert all("ts" in r for r in rows)


def test_journal_read_empty_when_missing(tmp_path):
    j = TradeJournal("BTC", root=tmp_path)
    assert j.read_all() == []


def test_journal_skips_corrupt_lines(tmp_path):
    j = TradeJournal("BTC", root=tmp_path)
    j.path.parent.mkdir(parents=True, exist_ok=True)
    j.path.write_text('{"event":"entry"}\nnot-json\n{"event":"exit"}\n')
    rows = j.read_all()
    assert [r["event"] for r in rows] == ["entry", "exit"]


def test_journal_isolates_by_coin(tmp_path):
    btc = TradeJournal("BTC", root=tmp_path)
    eth = TradeJournal("ETH", root=tmp_path)
    btc.record({"event": "entry"})
    eth.record({"event": "exit"})
    assert [r["event"] for r in btc.read_all()] == ["entry"]
    assert [r["event"] for r in eth.read_all()] == ["exit"]


def test_journal_writes_jsonl_format(tmp_path):
    j = TradeJournal("BTC", root=tmp_path)
    j.record({"event": "entry", "size": 1.5})
    raw = j.path.read_text()
    assert raw.endswith("\n")
    assert len(raw.strip().splitlines()) == 1
    assert json.loads(raw.strip())["size"] == 1.5


# --------------------------------------------------------------------------- #
#  Regime strategy → journal integration                                       #
# --------------------------------------------------------------------------- #


def _build_strategy(tmp_path):
    info = FakeInfo(mid=100.0)
    om = FakeOrderManager(info=info)
    risk = RiskManager(RiskConfig(max_order_usd=10_000, max_position_usd=10_000, max_drawdown_pct=1.0))
    state = StateStore("regime_switching", "BTC", root=tmp_path)
    journal = TradeJournal("BTC", root=tmp_path)
    cfg = RegimeSwitchingConfig()
    s = RegimeSwitchingStrategy(   # type: ignore[arg-type]
        "BTC", om, risk, cfg, state_store=state, journal=journal,
    )
    return s, om, journal


def _bull_snapshot() -> RegimeSnapshot:
    return RegimeSnapshot(
        regime=Regime.BULL,
        proba=np.array([0.05, 0.10, 0.85]),
        expected_return=0.001,
        expected_vol=0.01,
    )


def test_journal_records_entry_with_fill_price_from_order_result(tmp_path):
    s, om, journal = _build_strategy(tmp_path)
    om.next_result = FakeOrderResult(success=True, order_id=42, fill_price=100.5, fee=0.035)
    s._streak_target = Regime.BULL
    s._streak_bars = 100

    asyncio.run(s._maybe_enter(_bull_snapshot(), None, mid=100.0))

    rows = journal.read_all()
    assert len(rows) == 1
    r = rows[0]
    assert r["event"] == "entry"
    assert r["side"] == Side.BUY.value
    assert r["intended_price"] == 100.0
    assert r["fill_price"] == 100.5
    assert r["fee"] == 0.035
    assert r["regime"] == "bull"
    assert r["order_id"] == 42
    assert r["success"] is True


def test_journal_records_failed_entry_with_error(tmp_path):
    s, om, journal = _build_strategy(tmp_path)
    om.next_result = FakeOrderResult(success=False, error="rate limited")
    s._streak_target = Regime.BULL
    s._streak_bars = 100

    asyncio.run(s._maybe_enter(_bull_snapshot(), None, mid=100.0))

    rows = journal.read_all()
    assert len(rows) == 1
    assert rows[0]["success"] is False
    assert rows[0]["error"] == "rate limited"


def test_journal_records_exit_with_pnl_and_reason(tmp_path):
    s, om, journal = _build_strategy(tmp_path)
    # Simulate already-open long at $100, now closing at $110 (= +$1 on 0.1 size).
    s._in_position = True
    s._entry_price = 100.0
    s._position_side = Side.BUY
    s._target_regime = Regime.BULL
    om.position = {"coin": "BTC", "szi": "0.1"}
    om.next_result = FakeOrderResult(success=True, order_id=7, fill_price=110.0, fee=0.011)

    asyncio.run(s._close_position(exit_reason="regime_weakening", intended_price=109.8))

    rows = journal.read_all()
    assert len(rows) == 1
    r = rows[0]
    assert r["event"] == "exit"
    assert r["side"] == Side.SELL.value           # close direction
    assert r["intended_price"] == 109.8
    assert r["fill_price"] == 110.0
    assert r["entry_price"] == 100.0
    assert r["exit_reason"] == "regime_weakening"
    # P&L: long, (110 - 100) * 0.1 = +1.0
    assert r["pnl_usd"] == 1.0


def test_journal_exit_pnl_handles_short(tmp_path):
    s, om, journal = _build_strategy(tmp_path)
    s._in_position = True
    s._entry_price = 100.0
    s._position_side = Side.SELL
    s._target_regime = Regime.BEAR
    om.position = {"coin": "BTC", "szi": "-0.1"}
    om.next_result = FakeOrderResult(success=True, order_id=8, fill_price=90.0, fee=0.009)

    asyncio.run(s._close_position(exit_reason="stop loss hit (-12.00%)", intended_price=90.5))

    r = journal.read_all()[0]
    # Short closed at 90 from entry 100, size 0.1: (-1) * (90 - 100) * 0.1 = +1.0
    assert r["pnl_usd"] == 1.0
    assert r["side"] == Side.BUY.value
