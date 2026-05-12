import asyncio

from risk import RiskConfig, RiskManager
from runtime import StateStore, equity_watchdog


def test_order_size_limit():
    rm = RiskManager(RiskConfig(max_order_usd=100.0))
    assert rm.check_order("B", 50.0, 0.0) is True
    assert rm.check_order("B", 150.0, 0.0) is False


def test_position_limit():
    rm = RiskManager(RiskConfig(max_position_usd=500.0, max_order_usd=1000.0))
    assert rm.check_order("B", 400.0, 0.0) is True
    assert rm.check_order("B", 600.0, 0.0) is False


def test_drawdown_halt():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.10))
    assert rm.update_equity(1000.0) is True
    assert rm.update_equity(950.0) is True   # 5% — within limit
    assert rm.update_equity(890.0) is False  # 11% — halt


def test_update_equity_ignores_non_positive_values():
    """A zero or negative equity reading should not move the HWM or trigger
    a halt — otherwise a transient API hiccup that returns 0 would silently
    seed an unrecoverable broken state (NaN drawdowns)."""
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    assert rm.update_equity(1000.0) is True
    # Zero equity — must be ignored
    assert rm.update_equity(0.0) is True
    assert rm.is_halted() is False
    # Negative equity — also ignored
    assert rm.update_equity(-50.0) is True
    assert rm.is_halted() is False
    # HWM should still be 1000, so a 5% drop from there should still halt
    assert rm.update_equity(940.0) is False
    assert rm.is_halted() is True


def test_update_equity_initial_zero_does_not_seed_zero_hwm():
    """First call with equity=0 must not pin hwm=0 — otherwise every subsequent
    drawdown calc divides by zero and the halt never fires."""
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    assert rm.update_equity(0.0) is True
    # Now a real positive equity should seed the HWM cleanly.
    rm.update_equity(1000.0)
    rm.update_equity(900.0)
    assert rm.is_halted() is True


def test_halt_sets_flag_and_blocks_orders():
    rm = RiskManager(RiskConfig(max_order_usd=1000.0, max_position_usd=10_000.0, max_drawdown_pct=0.05))
    assert rm.is_halted() is False
    rm.update_equity(1000.0)
    rm.update_equity(900.0)  # 10% drawdown, breaches 5% limit
    assert rm.is_halted() is True
    assert rm.check_order("B", 100.0, 0.0) is False


def test_halt_stays_set_on_subsequent_updates():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    rm.update_equity(1000.0)
    rm.update_equity(900.0)  # halts
    assert rm.is_halted() is True
    rm.update_equity(910.0)  # still under HWM
    assert rm.is_halted() is True


def test_reset_halt_clears_flag_and_hwm():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    rm.update_equity(1000.0)
    rm.update_equity(900.0)
    assert rm.is_halted() is True
    rm.reset_halt()
    assert rm.is_halted() is False
    # New equity establishes new HWM
    assert rm.update_equity(800.0) is True
    assert rm.update_equity(850.0) is True


def test_halt_does_not_block_existing_position_logic():
    """check_order returns False, but exits don't go through check_order, so a
    halted strategy can still close out. We assert the asymmetry: register/release
    still work."""
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    rm.register_order()
    rm.update_equity(1000.0)
    rm.update_equity(900.0)
    assert rm.is_halted() is True
    # Closing the position still releases the slot
    rm.release_order()
    assert rm._open_order_count == 0


# --------------------------------------------------------------------------- #
#  equity_watchdog                                                             #
# --------------------------------------------------------------------------- #


def test_watchdog_halts_on_drawdown_sequence():
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    sequence = iter([1000.0, 950.0, 900.0, 1000.0])  # last one shouldn't unhalt

    def get_equity() -> float:
        return next(sequence, 0.0)

    async def driver():
        task = asyncio.create_task(equity_watchdog(get_equity, rm, poll_seconds=0.001))
        # Give it enough wall time to consume all four values.
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    assert rm.is_halted() is True


def test_watchdog_survives_poll_errors():
    """A flaky equity source must not crash the watchdog — it should keep retrying."""
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    calls = {"n": 0}

    def get_equity() -> float:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient API error")
        return 1000.0

    async def driver():
        task = asyncio.create_task(equity_watchdog(get_equity, rm, poll_seconds=0.001))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    assert calls["n"] >= 3   # at least one successful poll happened
    assert rm._equity_hwm == 1000.0


def test_watchdog_skips_zero_or_negative_equity():
    """A zero equity reading shouldn't reset the HWM — it usually means the API
    returned no data."""
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05))
    rm.update_equity(1000.0)
    sequence = iter([0.0, 0.0, 0.0])

    def get_equity() -> float:
        return next(sequence, 0.0)

    async def driver():
        task = asyncio.create_task(equity_watchdog(get_equity, rm, poll_seconds=0.001))
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(driver())
    assert rm._equity_hwm == 1000.0
    assert rm.is_halted() is False


# --------------------------------------------------------------------------- #
#  Persistence                                                                 #
# --------------------------------------------------------------------------- #


def test_hwm_persists_across_instances(tmp_path):
    store = StateStore("risk", "global", root=tmp_path)
    rm1 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store)
    rm1.update_equity(1000.0)
    rm1.update_equity(1100.0)
    assert rm1._equity_hwm == 1100.0

    store2 = StateStore("risk", "global", root=tmp_path)
    rm2 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store2)
    assert rm2._equity_hwm == 1100.0
    assert rm2.is_halted() is False


def test_halt_persists_across_instances(tmp_path):
    """The whole point: a bot halted on Monday is still halted Tuesday."""
    store = StateStore("risk", "global", root=tmp_path)
    rm1 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store)
    rm1.update_equity(1000.0)
    rm1.update_equity(900.0)   # 10% drawdown — halts
    assert rm1.is_halted() is True

    store2 = StateStore("risk", "global", root=tmp_path)
    rm2 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store2)
    assert rm2.is_halted() is True
    assert rm2._equity_hwm == 1000.0


def test_persisted_halt_blocks_orders_on_restart(tmp_path):
    """Restored halt must actually block check_order on the new instance."""
    store = StateStore("risk", "global", root=tmp_path)
    rm1 = RiskManager(
        RiskConfig(max_order_usd=1000.0, max_position_usd=10_000.0, max_drawdown_pct=0.05),
        state_store=store,
    )
    rm1.update_equity(1000.0)
    rm1.update_equity(900.0)

    store2 = StateStore("risk", "global", root=tmp_path)
    rm2 = RiskManager(
        RiskConfig(max_order_usd=1000.0, max_position_usd=10_000.0, max_drawdown_pct=0.05),
        state_store=store2,
    )
    assert rm2.check_order("B", 100.0, 0.0) is False


def test_reset_halt_clears_persisted_state(tmp_path):
    store = StateStore("risk", "global", root=tmp_path)
    rm1 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store)
    rm1.update_equity(1000.0)
    rm1.update_equity(900.0)
    assert rm1.is_halted() is True

    rm1.reset_halt()

    store2 = StateStore("risk", "global", root=tmp_path)
    rm2 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store2)
    assert rm2.is_halted() is False
    assert rm2._equity_hwm is None


def test_persisted_halt_survives_recovering_equity(tmp_path):
    """Even if equity recovers above the HWM after a halt, the halt stays —
    only an explicit reset_halt clears it."""
    store = StateStore("risk", "global", root=tmp_path)
    rm1 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store)
    rm1.update_equity(1000.0)
    rm1.update_equity(900.0)   # halt set

    store2 = StateStore("risk", "global", root=tmp_path)
    rm2 = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store2)
    rm2.update_equity(1500.0)  # equity recovers — but halt sticks
    assert rm2.is_halted() is True
    assert rm2._equity_hwm == 1500.0   # HWM updates normally


def test_pristine_state_does_not_set_halt(tmp_path):
    """No prior state file → fresh RiskManager starts unhalted with no HWM."""
    store = StateStore("risk", "global", root=tmp_path)
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store)
    assert rm.is_halted() is False
    assert rm._equity_hwm is None


def test_corrupt_risk_state_is_ignored(tmp_path):
    """A corrupt JSON file must not crash startup — load defaults instead."""
    store = StateStore("risk", "global", root=tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("{not valid json")
    rm = RiskManager(RiskConfig(max_drawdown_pct=0.05), state_store=store)
    assert rm.is_halted() is False
    assert rm._equity_hwm is None
