"""Generate a backtest results chart: price + funding + equity curve."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engine import BacktestResult

CHART_DIR = Path("data/charts")


def plot_results(
    result: BacktestResult,
    prices_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    save_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """
    Three-panel chart:
      1. Price (OHLC line) with entry/exit markers
      2. Funding rate with entry/exit thresholds
      3. Cumulative P&L (equity curve)
    """
    trades = result.trades
    cfg = result.config

    # ── data prep ──────────────────────────────────────────────────────────
    prices = prices_df.set_index("timestamp").sort_index()
    funding = funding_df.set_index("timestamp").sort_index()

    entry_times = [t.entry_time for t in trades]
    entry_prices = [t.entry_price for t in trades]
    exit_times = [t.exit_time for t in trades]
    exit_prices = [t.exit_price for t in trades]
    sides = [t.side for t in trades]
    pnls = [t.total_pnl for t in trades]
    reasons = [t.exit_reason for t in trades]

    # ── layout ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1,
        figsize=(16, 11),
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=9)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── panel 1: price + trade markers ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(prices.index, prices["close"], color="#58a6ff", linewidth=0.9, label="BTC close")

    for i, t in enumerate(trades):
        marker = "^" if t.side == "long" else "v"
        color = "#3fb950" if t.side == "long" else "#f85149"
        ax1.scatter(t.entry_time, t.entry_price, marker=marker, color=color,
                    s=80, zorder=5, linewidths=0)

        # Exit marker colour: green=profit, red=loss
        exit_color = "#3fb950" if t.total_pnl >= 0 else "#f85149"
        ax1.scatter(t.exit_time, t.exit_price, marker="x", color=exit_color,
                    s=60, zorder=5, linewidths=1.5)

        # Connector line for the trade duration
        ax1.plot(
            [t.entry_time, t.exit_time],
            [t.entry_price, t.exit_price],
            color="#8b949e", linewidth=0.6, linestyle="--", alpha=0.5, zorder=4,
        )

    # Legend proxies for entry markers
    ax1.scatter([], [], marker="^", color="#3fb950", s=60, label="Long entry")
    ax1.scatter([], [], marker="v", color="#f85149", s=60, label="Short entry")
    ax1.scatter([], [], marker="x", color="#3fb950", s=60, linewidths=1.5, label="Profitable exit")
    ax1.scatter([], [], marker="x", color="#f85149", s=60, linewidths=1.5, label="Loss exit")

    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", framealpha=0.9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    title = (
        f"Funding Rate Backtest — {result.coin}  "
        f"({result.start:%Y-%m-%d} → {result.end:%Y-%m-%d})  "
        f"| {len(trades)} trades  |  Net P&L ${sum(pnls):+.2f}"
    )
    ax1.set_title(title, color="#e6edf3", fontsize=11, pad=8)

    # ── panel 2: funding rate ─────────────────────────────────────────────
    ax2 = axes[1]
    funding_pct = funding["funding_rate"] * 100
    ax2.bar(funding.index, funding_pct, width=0.003, color="#58a6ff", alpha=0.6, label="Funding rate")
    ax2.axhline(cfg.entry_threshold * 100, color="#3fb950", linewidth=0.8, linestyle="--",
                label=f"Entry threshold ({cfg.entry_threshold:.4%})")
    ax2.axhline(-cfg.entry_threshold * 100, color="#f85149", linewidth=0.8, linestyle="--")
    ax2.axhline(cfg.exit_threshold * 100, color="#e3b341", linewidth=0.6, linestyle=":",
                label=f"Exit threshold ({cfg.exit_threshold:.5%})")
    ax2.axhline(-cfg.exit_threshold * 100, color="#e3b341", linewidth=0.6, linestyle=":")
    ax2.axhline(0, color="#8b949e", linewidth=0.5)

    ax2.set_ylabel("Funding rate (%)", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", framealpha=0.9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}%"))

    # ── panel 3: equity curve ─────────────────────────────────────────────
    ax3 = axes[2]
    equity = result.equity_curve
    color_line = "#3fb950" if float(equity.iloc[-1]) >= 0 else "#f85149"
    ax3.plot(equity.index, equity.values, color=color_line, linewidth=1.0)
    ax3.fill_between(equity.index, equity.values, 0,
                     where=(equity.values >= 0), color="#3fb950", alpha=0.15)
    ax3.fill_between(equity.index, equity.values, 0,
                     where=(equity.values < 0), color="#f85149", alpha=0.15)
    ax3.axhline(0, color="#8b949e", linewidth=0.5)

    ax3.set_ylabel("Cumulative P&L (USD)", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+.2f}"))

    # ── x-axis formatting ─────────────────────────────────────────────────
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 1])

    if save_path is None:
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHART_DIR / f"backtest_{result.coin}_{result.start:%Y%m%d}_{result.end:%Y%m%d}.png"

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return save_path
