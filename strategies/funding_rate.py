import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from execution import OrderManager, Side
from risk import RiskManager
from .base import Strategy


@dataclass
class FundingConfig:
    # Entry: only trade when annualised rate exceeds this (0.02%/hr = ~175%/yr)
    entry_threshold: float = 0.0002
    # Exit: close when rate drops below this (0.005%/hr — no longer worth holding)
    exit_threshold: float = 0.00005
    # Hard stop: close if position moves this much against us
    stop_loss_pct: float = 0.02
    # Hard time cap — never hold longer than this regardless of funding
    max_hold_hours: int = 48
    # Notional size in USD — kept small and conservative
    position_size_usd: float = 50.0
    # How often to poll the REST API
    poll_interval_seconds: int = 600


class FundingRateStrategy(Strategy):
    """
    Collects funding payments by sitting on the receiving side when rates are extreme.
    Positive funding → short (longs pay us). Negative funding → long (shorts pay us).
    Exits when funding normalises, flips, stop loss fires, or max hold time is reached.
    """

    def __init__(self, coin: str, order_manager: OrderManager, risk: RiskManager,
                 config: FundingConfig | None = None) -> None:
        super().__init__(coin, order_manager, risk)
        self.cfg = config or FundingConfig()
        self._in_position = False
        self._entry_price: float | None = None
        self._entry_time: datetime | None = None
        self._position_side: Side | None = None

    def name(self) -> str:
        return "funding_rate"

    async def run(self) -> None:
        self.orders.set_leverage(self.coin, leverage=1, is_cross=True)
        self._check_existing_position()

        logger.info(
            "{} | {} started | entry≥{:.4%}/hr exit≤{:.4%}/hr stop={:.0%} size=${} poll={}s",
            self.name(), self.coin,
            self.cfg.entry_threshold, self.cfg.exit_threshold,
            self.cfg.stop_loss_pct, self.cfg.position_size_usd,
            self.cfg.poll_interval_seconds,
        )

        while True:
            try:
                await self._tick()
            except Exception as exc:
                logger.error("{} tick error: {}", self.name(), exc)
            await asyncio.sleep(self.cfg.poll_interval_seconds)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _get_funding_rate(self) -> float | None:
        try:
            meta, asset_ctxs = self.orders.info.meta_and_asset_ctxs()
            for i, asset in enumerate(meta["universe"]):
                if asset["name"] == self.coin:
                    return float(asset_ctxs[i]["funding"])
        except Exception as exc:
            logger.warning("Failed to fetch funding rate: {}", exc)
        return None

    def _get_mid_price(self) -> float | None:
        try:
            mids = self.orders.info.all_mids()
            raw = mids.get(self.coin)
            return float(raw) if raw is not None else None
        except Exception as exc:
            logger.warning("Failed to fetch mid price: {}", exc)
        return None

    def _check_existing_position(self) -> None:
        pos = self.orders.get_position(self.coin)
        if pos and float(pos.get("szi", 0)) != 0:
            logger.warning(
                "{} | open position detected on startup (szi={}) — strategy will monitor but not enter until closed",
                self.coin, pos["szi"],
            )

    async def _tick(self) -> None:
        funding = self._get_funding_rate()
        mid = self._get_mid_price()

        if funding is None or mid is None:
            logger.warning("{} | skipping tick — could not fetch market data", self.coin)
            return

        annualised = funding * 8760
        logger.info(
            "{} | funding={:+.5%}/hr ({:+.1%}/yr) mid={:.2f} in_position={}",
            self.coin, funding, annualised, mid, self._in_position,
        )

        if not self._in_position:
            await self._check_entry(funding, mid)
        else:
            await self._check_exit(funding, mid)

    async def _check_entry(self, funding: float, mid: float) -> None:
        if abs(funding) < self.cfg.entry_threshold:
            return

        # Receive funding by being on the paying side's opposite
        side = Side.SELL if funding > 0 else Side.BUY
        size = round(self.cfg.position_size_usd / mid, 6)

        if not self.risk.check_order(side, self.cfg.position_size_usd, 0.0):
            return

        logger.info(
            "{} | entering {} | funding={:+.5%}/hr ({:+.1%}/yr) size={} @ {:.2f}",
            self.coin, side, funding, funding * 8760, size, mid,
        )

        result = self.orders.market_order(self.coin, side, size)
        if result.success:
            self._in_position = True
            self._entry_price = mid
            self._entry_time = datetime.now(timezone.utc)
            self._position_side = side
            self.risk.register_order()

    async def _check_exit(self, funding: float, mid: float) -> None:
        reasons: list[str] = []

        # Rate normalised — no longer worth holding
        if abs(funding) < self.cfg.exit_threshold:
            reasons.append(f"funding normalised ({funding:+.5%}/hr)")

        # Funding flipped — we'd now be paying instead of receiving
        if self._position_side == Side.SELL and funding < 0:
            reasons.append(f"funding flipped ({funding:+.5%}/hr, now paying)")
        elif self._position_side == Side.BUY and funding > 0:
            reasons.append(f"funding flipped ({funding:+.5%}/hr, now paying)")

        # Stop loss
        if self._entry_price:
            if self._position_side == Side.SELL:
                pnl_pct = (self._entry_price - mid) / self._entry_price
            else:
                pnl_pct = (mid - self._entry_price) / self._entry_price

            if pnl_pct < -self.cfg.stop_loss_pct:
                reasons.append(f"stop loss hit ({pnl_pct:.2%})")

        # Max hold time
        if self._entry_time:
            held_hours = (datetime.now(timezone.utc) - self._entry_time).total_seconds() / 3600
            if held_hours >= self.cfg.max_hold_hours:
                reasons.append(f"max hold time ({held_hours:.1f}h)")

        if not reasons:
            return

        logger.info("{} | exiting | reasons: {}", self.coin, " | ".join(reasons))
        await self._close_position()

    async def _close_position(self) -> None:
        pos = self.orders.get_position(self.coin)
        if not pos or float(pos.get("szi", 0)) == 0:
            logger.warning("{} | tried to close but no open position found", self.coin)
            self._reset()
            return

        size = abs(float(pos["szi"]))
        close_side = Side.BUY if self._position_side == Side.SELL else Side.SELL
        result = self.orders.market_order(self.coin, close_side, size)
        if result.success:
            self._reset()

    def _reset(self) -> None:
        self._in_position = False
        self._entry_price = None
        self._entry_time = None
        self._position_side = None
        self.risk.release_order()
