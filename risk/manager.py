from dataclasses import dataclass

from loguru import logger

from runtime import StateStore


@dataclass
class RiskConfig:
    max_position_usd: float = 1_000.0
    max_drawdown_pct: float = 0.05      # 5% from equity high-water mark
    max_order_usd: float = 500.0
    max_open_orders: int = 10


class RiskManager:
    def __init__(
        self,
        config: RiskConfig | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        self.cfg = config or RiskConfig()
        self._equity_hwm: float | None = None
        self._open_order_count: int = 0
        self._halted: bool = False
        self._state = state_store or StateStore("risk", "global")
        self._load_state()

    def check_order(self, side: str, size_usd: float, current_position_usd: float) -> bool:
        if self._halted:
            logger.warning("Order rejected: trading halted (drawdown breach)")
            return False

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
        """Return False (and halt) if drawdown limit breached.

        Once halted, check_order returns False until reset_halt() is called.
        Existing positions are unaffected — strategies still close them via
        their normal exit paths (which don't go through check_order).
        """
        # Non-positive equity is meaningless for the drawdown calc — the watchdog
        # already filters it but defend the manager too. A zero hwm would otherwise
        # produce NaN drawdowns that silently never trigger the halt.
        if equity <= 0:
            return not self._halted
        if self._equity_hwm is None or equity > self._equity_hwm:
            self._equity_hwm = equity

        drawdown = (self._equity_hwm - equity) / self._equity_hwm
        breached = drawdown >= self.cfg.max_drawdown_pct
        if breached and not self._halted:
            logger.error(
                "Drawdown {:.2%} >= limit {:.2%} — trading halted",
                drawdown, self.cfg.max_drawdown_pct,
            )
            self._halted = True

        self._save_state()
        return not breached

    def is_halted(self) -> bool:
        return self._halted

    def reset_halt(self) -> None:
        """Manually clear the halt and high-water mark (operator override)."""
        self._halted = False
        self._equity_hwm = None
        self._save_state()
        logger.warning("Risk halt manually cleared")

    def register_order(self) -> None:
        self._open_order_count += 1

    def release_order(self) -> None:
        self._open_order_count = max(0, self._open_order_count - 1)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #
    def _save_state(self) -> None:
        self._state.save({
            "halted": self._halted,
            "equity_hwm": self._equity_hwm,
        })

    def _load_state(self) -> None:
        s = self._state.load()
        if not s:
            return
        self._halted = bool(s.get("halted", False))
        hwm = s.get("equity_hwm")
        self._equity_hwm = float(hwm) if hwm is not None else None
        if self._halted:
            logger.error(
                "Risk halt restored from disk | hwm={} — entries blocked until reset_halt()",
                self._equity_hwm,
            )
