"""Simulate FundingRateStrategy against historical data."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd

from strategies.configs import FundingConfig

Side = Literal["long", "short"]


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: Side
    entry_price: float
    exit_price: float
    position_size_usd: float
    funding_collected: float  # net (positive = received, negative = paid); includes fees
    exit_reason: str

    @property
    def price_pnl(self) -> float:
        direction = -1.0 if self.side == "short" else 1.0
        return direction * (self.exit_price - self.entry_price) / self.entry_price * self.position_size_usd

    @property
    def total_pnl(self) -> float:
        return self.price_pnl + self.funding_collected

    @property
    def hold_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series  # DatetimeIndex → running P&L (realised + unrealised)
    config: FundingConfig
    coin: str
    start: datetime
    end: datetime


def _build_sim_df(funding_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Merge funding and price DataFrames on hourly timestamps."""
    funding = (
        funding_df
        .assign(timestamp=lambda d: d["timestamp"].dt.floor("h"))
        .set_index("timestamp")
        .rename(columns={"funding_rate": "funding_raw"})
    )
    prices = prices_df.set_index("timestamp")
    df = prices.join(funding, how="left")
    # Forward-fill so each hour carries the last known funding rate
    df["funding_rate"] = df["funding_raw"].ffill()
    df["is_settlement"] = df["funding_raw"].notna()
    return df.reset_index()


def _check_exit(
    cfg: FundingConfig,
    ts: datetime,
    close: float,
    high: float,
    low: float,
    entry_price: float,
    entry_time: datetime,
    side: Side,
    funding: float,
) -> str:
    # Use high/low for intrabar stop detection (more realistic than close-only)
    adverse_price = high if side == "short" else low
    direction = -1.0 if side == "short" else 1.0
    pnl_pct = direction * (adverse_price - entry_price) / entry_price
    if pnl_pct < -cfg.stop_loss_pct:
        return "stop_loss"

    if abs(funding) < cfg.exit_threshold:
        return "funding_normalised"

    if (side == "short" and funding < 0) or (side == "long" and funding > 0):
        return "funding_flipped"

    if (ts - entry_time).total_seconds() / 3600 >= cfg.max_hold_hours:
        return "max_hold_time"

    return ""


def _stop_price(entry_price: float, side: Side, stop_pct: float) -> float:
    """Approximate fill price for a stop loss (worst-case for the holder)."""
    return entry_price * (1 + stop_pct) if side == "short" else entry_price * (1 - stop_pct)


def run_backtest(
    funding_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    coin: str,
    config: FundingConfig | None = None,
    fee_rate: float = 0.00035,
) -> BacktestResult:
    """
    Simulate the funding rate carry strategy on historical data.

    fee_rate: taker fee per side (Hyperliquid default ~0.035% = 0.00035).
    funding_rate units must match live strategy (per-8h period from Hyperliquid API).
    """
    cfg = config or FundingConfig()
    df = _build_sim_df(funding_df, prices_df)
    start = df["timestamp"].iloc[0].to_pydatetime()
    end = df["timestamp"].iloc[-1].to_pydatetime()

    trades: list[Trade] = []
    equity_points: list[tuple[datetime, float]] = []

    in_position = False
    entry_price = 0.0
    entry_time = start
    side: Side = "long"
    funding_accumulated = 0.0

    for row in df.itertuples(index=False):
        ts: datetime = row.timestamp.to_pydatetime()
        close: float = row.close
        funding: float = 0.0 if (isinstance(row.funding_rate, float) and np.isnan(row.funding_rate)) else float(row.funding_rate)
        is_settlement: bool = bool(row.is_settlement)

        if in_position:
            # Collect (or pay) funding at each 8-hour settlement
            if is_settlement and funding != 0.0:
                receive = (side == "short" and funding > 0) or (side == "long" and funding < 0)
                funding_accumulated += abs(funding) * cfg.position_size_usd * (1 if receive else -1)

            reason = _check_exit(cfg, ts, close, row.high, row.low, entry_price, entry_time, side, funding)
            if reason:
                exit_price = _stop_price(entry_price, side, cfg.stop_loss_pct) if reason == "stop_loss" else close
                fees = 2 * fee_rate * cfg.position_size_usd
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position_size_usd=cfg.position_size_usd,
                    funding_collected=funding_accumulated - fees,
                    exit_reason=reason,
                ))
                in_position = False
                funding_accumulated = 0.0

        else:
            if funding != 0.0 and abs(funding) >= cfg.entry_threshold:
                side = "short" if funding > 0 else "long"
                in_position = True
                entry_price = close
                entry_time = ts
                funding_accumulated = 0.0

        realised = sum(t.total_pnl for t in trades)
        unrealised = 0.0
        if in_position:
            direction = -1.0 if side == "short" else 1.0
            unrealised = (direction * (close - entry_price) / entry_price * cfg.position_size_usd
                          + funding_accumulated)
        equity_points.append((ts, realised + unrealised))

    # Close any open position at end of backtest
    if in_position:
        last_price = float(df["close"].iloc[-1])
        fees = 2 * fee_rate * cfg.position_size_usd
        trades.append(Trade(
            entry_time=entry_time,
            exit_time=end,
            side=side,
            entry_price=entry_price,
            exit_price=last_price,
            position_size_usd=cfg.position_size_usd,
            funding_collected=funding_accumulated - fees,
            exit_reason="backtest_end",
        ))

    equity_curve = pd.Series(
        [p for _, p in equity_points],
        index=pd.DatetimeIndex([t for t, _ in equity_points]),
        name="pnl",
    )
    return BacktestResult(trades=trades, equity_curve=equity_curve, config=cfg, coin=coin, start=start, end=end)
