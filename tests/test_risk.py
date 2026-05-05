from risk import RiskConfig, RiskManager


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
