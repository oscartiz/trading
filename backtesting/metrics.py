"""Compute and display backtest performance metrics."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import BacktestResult


def compute_metrics(result: BacktestResult) -> dict:
    trades = result.trades
    equity = result.equity_curve

    if not trades:
        return {"error": "no trades executed"}

    pnls = [t.total_pnl for t in trades]
    total_pnl = sum(pnls)
    funding_total = sum(t.funding_collected for t in trades)
    price_pnl_total = sum(t.price_pnl for t in trades)
    n_trades = len(trades)
    win_rate = sum(1 for p in pnls if p > 0) / n_trades
    avg_hold_h = sum(t.hold_hours for t in trades) / n_trades

    cummax = equity.cummax()
    drawdown = equity - cummax
    max_dd_usd = float(drawdown.min())
    peak = float(cummax.max())
    max_dd_pct = (max_dd_usd / peak * 100) if peak > 0 else 0.0

    hourly_delta = equity.diff().dropna()
    std = float(hourly_delta.std())
    sharpe = (float(hourly_delta.mean()) / std * math.sqrt(8760)) if std > 0 else 0.0

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    duration_days = (result.end - result.start).total_seconds() / 86400

    return {
        "coin": result.coin,
        "period_days": round(duration_days, 1),
        "n_trades": n_trades,
        "total_pnl_usd": round(total_pnl, 4),
        "funding_collected_usd": round(funding_total, 4),
        "price_pnl_usd": round(price_pnl_total, 4),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_pnl_per_trade_usd": round(total_pnl / n_trades, 4),
        "best_trade_usd": round(max(pnls), 4),
        "worst_trade_usd": round(min(pnls), 4),
        "avg_hold_hours": round(avg_hold_h, 1),
        "max_drawdown_usd": round(max_dd_usd, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 3),
        "exit_reasons": reasons,
    }


def print_metrics(result: BacktestResult) -> None:
    m = compute_metrics(result)
    cfg = result.config

    if "error" in m:
        print(f"\nBacktest complete — {m['error']}\n")
        return

    print("\n" + "=" * 58)
    print(f"  Funding Rate Backtest — {m['coin']}")
    print("=" * 58)
    print(f"  Period       : {result.start:%Y-%m-%d} → {result.end:%Y-%m-%d}  ({m['period_days']} days)")
    print(f"  Entry ≥      : {cfg.entry_threshold:.4%}/hr")
    print(f"  Exit  ≤      : {cfg.exit_threshold:.5%}/hr")
    print(f"  Stop loss    : {cfg.stop_loss_pct:.0%}   Max hold: {cfg.max_hold_hours}h")
    print(f"  Position     : ${cfg.position_size_usd}")
    print()
    print(f"  Trades       : {m['n_trades']}")
    print(f"  Win rate     : {m['win_rate_pct']}%")
    print(f"  Avg hold     : {m['avg_hold_hours']}h")
    print()
    print(f"  Total P&L    : ${m['total_pnl_usd']:+.4f}")
    print(f"    Funding    : ${m['funding_collected_usd']:+.4f}")
    print(f"    Price      : ${m['price_pnl_usd']:+.4f}")
    print(f"  Per trade    : ${m['avg_pnl_per_trade_usd']:+.4f}")
    print(f"  Best / Worst : ${m['best_trade_usd']:+.4f}  /  ${m['worst_trade_usd']:+.4f}")
    print()
    print(f"  Max drawdown : ${m['max_drawdown_usd']:.4f}  ({m['max_drawdown_pct']:.2f}%)")
    print(f"  Sharpe ratio : {m['sharpe_ratio']:.3f}")
    print()
    reasons_str = "  ".join(f"{k}={v}" for k, v in m["exit_reasons"].items())
    print(f"  Exit reasons : {reasons_str}")
    print("=" * 58 + "\n")
