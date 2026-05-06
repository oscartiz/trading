"""Generate a regime-switching backtest chart: price+regimes / probabilities / equity."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

if TYPE_CHECKING:
    from .regime_engine import RegimeBacktestResult

CHART_DIR = Path("data/charts")

_REGIME_COLOR = {
    "bear": "#f85149",
    "chop": "#8b949e",
    "bull": "#3fb950",
}


def plot_regime_results(
    result: "RegimeBacktestResult",
    prices_df: pd.DataFrame,
    save_path: Path | None = None,
    show: bool = True,
) -> Path:
    """
    Three-panel chart:
      1. Price with entry/exit markers and regime-shaded background
      2. Regime posterior probabilities (P(bear), P(chop), P(bull))
      3. Cumulative P&L (equity curve)
    """
    trades = result.trades
    cfg = result.config
    regimes = result.regimes

    prices = prices_df.set_index("timestamp").sort_index()
    pnls = [t.total_pnl for t in trades]

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

    ax1 = axes[0]
    ax1.plot(prices.index, prices["close"], color="#58a6ff", linewidth=0.9, label=f"{result.coin} close")

    if not regimes.empty:
        _shade_regimes(ax1, regimes)

    for t in trades:
        marker = "^" if t.side == "long" else "v"
        entry_color = _REGIME_COLOR["bull"] if t.side == "long" else _REGIME_COLOR["bear"]
        ax1.scatter(t.entry_time, t.entry_price, marker=marker, color=entry_color,
                    s=80, zorder=5, linewidths=0)

        exit_color = _REGIME_COLOR["bull"] if t.total_pnl >= 0 else _REGIME_COLOR["bear"]
        ax1.scatter(t.exit_time, t.exit_price, marker="x", color=exit_color,
                    s=60, zorder=5, linewidths=1.5)

        ax1.plot(
            [t.entry_time, t.exit_time],
            [t.entry_price, t.exit_price],
            color="#8b949e", linewidth=0.6, linestyle="--", alpha=0.5, zorder=4,
        )

    ax1.scatter([], [], marker="^", color=_REGIME_COLOR["bull"], s=60, label="Long entry")
    ax1.scatter([], [], marker="v", color=_REGIME_COLOR["bear"], s=60, label="Short entry")
    ax1.scatter([], [], marker="x", color=_REGIME_COLOR["bull"], s=60, linewidths=1.5, label="Profitable exit")
    ax1.scatter([], [], marker="x", color=_REGIME_COLOR["bear"], s=60, linewidths=1.5, label="Loss exit")

    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", framealpha=0.9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    title = (
        f"Regime-Switching Backtest — {result.coin}  "
        f"({result.start:%Y-%m-%d} → {result.end:%Y-%m-%d})  "
        f"| {len(trades)} trades  |  Net P&L ${sum(pnls):+.2f}"
    )
    ax1.set_title(title, color="#e6edf3", fontsize=11, pad=8)

    ax2 = axes[1]
    if not regimes.empty:
        ax2.plot(regimes.index, regimes["p_bull"], color=_REGIME_COLOR["bull"],
                 linewidth=0.9, label="P(bull)")
        ax2.plot(regimes.index, regimes["p_chop"], color=_REGIME_COLOR["chop"],
                 linewidth=0.9, label="P(chop)")
        ax2.plot(regimes.index, regimes["p_bear"], color=_REGIME_COLOR["bear"],
                 linewidth=0.9, label="P(bear)")
        ax2.axhline(cfg.entry_proba, color="#e3b341", linewidth=0.7, linestyle="--",
                    label=f"Entry P ≥ {cfg.entry_proba:.2f}")
        ax2.axhline(cfg.exit_proba, color="#e3b341", linewidth=0.5, linestyle=":",
                    label=f"Exit P < {cfg.exit_proba:.2f}")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Posterior", fontsize=9)
    ax2.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", framealpha=0.9, ncol=5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

    ax3 = axes[2]
    equity = result.equity_curve
    if len(equity) > 0:
        color_line = _REGIME_COLOR["bull"] if float(equity.iloc[-1]) >= 0 else _REGIME_COLOR["bear"]
        ax3.plot(equity.index, equity.values, color=color_line, linewidth=1.0)
        ax3.fill_between(equity.index, equity.values, 0,
                         where=(equity.values >= 0), color=_REGIME_COLOR["bull"], alpha=0.15)
        ax3.fill_between(equity.index, equity.values, 0,
                         where=(equity.values < 0), color=_REGIME_COLOR["bear"], alpha=0.15)
    ax3.axhline(0, color="#8b949e", linewidth=0.5)
    ax3.set_ylabel("Cumulative P&L (USD)", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+.2f}"))

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.tight_layout(rect=[0, 0, 1, 1])

    if save_path is None:
        CHART_DIR.mkdir(parents=True, exist_ok=True)
        save_path = CHART_DIR / (
            f"regime_{result.coin}_{result.start:%Y%m%d}_{result.end:%Y%m%d}.png"
        )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _shade_regimes(ax, regimes: pd.DataFrame) -> None:
    """Shade the price panel with light bands of the dominant regime per bar."""
    labels = regimes["regime"].to_numpy()
    times = regimes.index.to_numpy()
    if len(labels) == 0:
        return

    start_idx = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            ax.axvspan(times[start_idx], times[i],
                       color=_REGIME_COLOR.get(current, "#30363d"), alpha=0.06, zorder=0)
            current = labels[i]
            start_idx = i
    ax.axvspan(times[start_idx], times[-1],
               color=_REGIME_COLOR.get(current, "#30363d"), alpha=0.06, zorder=0)
