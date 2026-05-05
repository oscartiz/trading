from dataclasses import dataclass

from loguru import logger


@dataclass
class RiskConfig:
    max_position_usd: float = 1_000.0
    max_drawdown_pct: float = 0.05      # 5% from equity high-water mark
    max_order_usd: float = 500.0
    max_open_orders: int = 10


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()
        self._equity_hwm: float | None = None
        self._open_order_count: int = 0

    def check_order(self, side: str, size_usd: float, current_position_usd: float) -> bool:
        if size_usd > self.cfg.max_order_usd:
            logger.warning("Order rejected: size ${:.2f} > max ${:.2f}", size_usd, self.cfg.max_order_usd)
            return False

        projected = abs(current_position_usd + size_usd)
        if projected > self.cfg.max_position_usd:
            logger.warning(
                "Order rejected: projected position ${:.2f} > max ${:.2f}", projected, self.cfg.max_position_usd
            )
            return False

        if self._open_order_count >= self.cfg.max_open_orders:
            logger.warning("Order rejected: too many open orders ({})", self._open_order_count)
            return False

        return True

    def update_equity(self, equity: float) -> bool:
        """Return False (halt) if drawdown limit breached."""
        if self._equity_hwm is None or equity > self._equity_hwm:
            self._equity_hwm = equity

        drawdown = (self._equity_hwm - equity) / self._equity_hwm
        if drawdown >= self.cfg.max_drawdown_pct:
            logger.error(
                "Drawdown {:.2%} >= limit {:.2%} — trading halted",
                drawdown,
                self.cfg.max_drawdown_pct,
            )
            return False
        return True

    def register_order(self) -> None:
        self._open_order_count += 1

    def release_order(self) -> None:
        self._open_order_count = max(0, self._open_order_count - 1)
