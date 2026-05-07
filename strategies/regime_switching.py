"""3-state regime-switching strategy for Hyperliquid perpetuals.

Workflow:
    1. Fetch the last N hourly candles for the coin.
    2. Compute log returns and fit a 3-state Gaussian HMM (states sorted by mean
       so state 0=bear, 1=chop, 2=bull).
    3. On every newly-closed bar, run smoothing on the rolling window to obtain
       posterior probabilities for the most recent bar.
    4. If flat: enter long when P(bull) ≥ entry_proba (and chop is not
       dominant); enter short when P(bear) ≥ entry_proba.
    5. If in a position: exit when the held regime's probability drops below
       exit_proba, or on a hard stop, take profit, or max-hold timeout.
    6. Refit the HMM every `refit_every_bars` bars; in between we do inference
       only against the existing parameters.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone

import numpy as np
from loguru import logger

from execution import OrderManager, Side
from risk import RiskManager

from .base import Strategy
from .configs import RegimeSwitchingConfig
from .regime import Regime, RegimeClassifier, log_returns_from_close


class RegimeSwitchingStrategy(Strategy):
    def __init__(
        self,
        coin: str,
        order_manager: OrderManager,
        risk: RiskManager,
        config: RegimeSwitchingConfig | None = None,
    ) -> None:
        super().__init__(coin, order_manager, risk)
        self.cfg = config or RegimeSwitchingConfig()

        self._closes: deque[float] = deque(maxlen=self.cfg.train_window_bars)
        self._last_bar_open_ms: int | None = None
        self._bars_since_fit: int = 0
        self._classifier = RegimeClassifier(
            n_iter=self.cfg.hmm_max_iter,
            tol=self.cfg.hmm_tol,
            random_state=self.cfg.hmm_random_state,
        )

        self._in_position = False
        self._entry_price: float | None = None
        self._entry_bar_index: int = 0
        self._bar_index: int = 0
        self._position_side: Side | None = None
        self._target_regime: Regime | None = None
        self._last_exit_regime: Regime | None = None
        self._last_exit_bar_index: int = -10**9
        # Per-bar candidate streak (mirrors backtest engine logic).
        self._streak_target: Regime | None = None
        self._streak_bars: int = 0

    def name(self) -> str:
        return "regime_switching"

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        self.orders.set_leverage(self.coin, leverage=self.cfg.leverage, is_cross=self.cfg.is_cross)
        self._check_existing_position()
        await self._warm_up()

        logger.info(
            "{} | {} started | window={} bars refit_every={} entry_p≥{:.2f} exit_p<{:.2f} "
            "stop={:.0%} tp={} max_hold={}b size=${}",
            self.name(), self.coin, self.cfg.train_window_bars, self.cfg.refit_every_bars,
            self.cfg.entry_proba, self.cfg.exit_proba, self.cfg.stop_loss_pct,
            f"{self.cfg.take_profit_pct:.0%}" if self.cfg.take_profit_pct else "off",
            self.cfg.max_hold_bars, self.cfg.position_size_usd,
        )

        while True:
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("{} tick error: {}", self.name(), exc)
            await asyncio.sleep(self.cfg.poll_interval_seconds)

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #
    async def _warm_up(self) -> None:
        candles = self._fetch_recent_candles(self.cfg.train_window_bars)
        if not candles:
            raise RuntimeError(f"{self.coin} | no candles returned during warm-up")

        for c in candles:
            self._closes.append(float(c["c"]))
        self._last_bar_open_ms = int(candles[-1]["t"])
        self._refit()

    def _check_existing_position(self) -> None:
        pos = self.orders.get_position(self.coin)
        if pos and float(pos.get("szi", 0)) != 0:
            logger.warning(
                "{} | existing position on startup (szi={}) — strategy will not open new positions until flat",
                self.coin, pos["szi"],
            )

    # ------------------------------------------------------------------ #
    #  Per-bar logic                                                       #
    # ------------------------------------------------------------------ #
    async def _tick(self) -> None:
        new_bars = self._poll_new_bars()
        if not new_bars:
            return

        for bar in new_bars:
            self._closes.append(float(bar["c"]))
            self._bars_since_fit += 1
            self._bar_index += 1

        if self._bars_since_fit >= self.cfg.refit_every_bars:
            self._refit()

        snap, viterbi_label = self._classify_now()
        if snap is None:
            return

        # Streak counter must advance every bar — even while in a position —
        # otherwise re-entries after exit would skip the confirmation gate.
        self._update_streak(self._candidate_target(snap, viterbi_label))

        mid = self._closes[-1]
        regime_label = viterbi_label.label if viterbi_label is not None else snap.label
        logger.info(
            "{} | bar={} regime={} p_bear={:.2f} p_chop={:.2f} p_bull={:.2f} "
            "Er={:+.5f} σ={:.5f} close={:.2f} in_position={}",
            self.coin, self._bar_index, regime_label,
            snap.proba[0], snap.proba[1], snap.proba[2],
            snap.expected_return, snap.expected_vol, mid, self._in_position,
        )

        if not self._in_position:
            await self._maybe_enter(snap, viterbi_label, mid)
        else:
            await self._maybe_exit(snap, viterbi_label, mid)

    def _classify_now(self):
        returns = log_returns_from_close(np.fromiter(self._closes, dtype=np.float64))
        if returns.size < 10:
            return None, None
        snap = self._classifier.snapshot(returns)
        viterbi_label: Regime | None = None
        if self.cfg.signal_mode == "viterbi":
            viterbi_label = Regime(int(self._classifier.predict(returns)[-1]))
        return snap, viterbi_label

    def _refit(self) -> None:
        returns = log_returns_from_close(np.fromiter(self._closes, dtype=np.float64))
        if returns.size < self._classifier.hmm.n_states * 5:
            logger.warning("{} | not enough returns to fit HMM ({})", self.coin, returns.size)
            return
        self._classifier.fit(returns)
        self._bars_since_fit = 0
        means = self._classifier.state_means
        logger.info(
            "{} | HMM refit | bear_µ={:+.5f} chop_µ={:+.5f} bull_µ={:+.5f} (n={})",
            self.coin, means[0], means[1], means[2], returns.size,
        )

    # ------------------------------------------------------------------ #
    #  Entry / exit                                                        #
    # ------------------------------------------------------------------ #
    def _candidate_target(self, snap, viterbi_label: Regime | None) -> Regime | None:
        """Return the regime we'd enter on this bar absent streak/cooldown gates."""
        p_bear, p_chop, p_bull = snap.proba
        if self.cfg.signal_mode == "viterbi":
            if viterbi_label == Regime.BULL and snap.expected_return >= self.cfg.min_expected_return_per_bar:
                return Regime.BULL
            if viterbi_label == Regime.BEAR and snap.expected_return <= -self.cfg.min_expected_return_per_bar:
                return Regime.BEAR
            return None
        if p_chop > self.cfg.max_chop_proba:
            return None
        if p_bull >= self.cfg.entry_proba and snap.expected_return >= self.cfg.min_expected_return_per_bar:
            return Regime.BULL
        if p_bear >= self.cfg.entry_proba and snap.expected_return <= -self.cfg.min_expected_return_per_bar:
            return Regime.BEAR
        return None

    def _update_streak(self, candidate: Regime | None) -> None:
        if candidate is not None and candidate == self._streak_target:
            self._streak_bars += 1
        elif candidate is not None:
            self._streak_target = candidate
            self._streak_bars = 1
        else:
            self._streak_target = None
            self._streak_bars = 0

    async def _maybe_enter(self, snap, viterbi_label: Regime | None, mid: float) -> None:
        target = self._candidate_target(snap, viterbi_label)
        if target is None:
            return

        # Confirmation: candidate must have been consistent for ≥ N bars.
        if self._streak_bars < self.cfg.entry_confirmation_bars:
            return

        # Same-regime cooldown after a recent exit.
        if (self._last_exit_regime == target
                and (self._bar_index - self._last_exit_bar_index) < self.cfg.same_regime_cooldown_bars):
            return

        side: Side = Side.BUY if target == Regime.BULL else Side.SELL
        target_proba: float = float(snap.proba[int(target)])
        scale = max(target_proba, self.cfg.min_size_scale)
        notional = self.cfg.position_size_usd * scale
        size = round(notional / mid, 6)

        if not self.risk.check_order(side, notional, 0.0):
            return

        logger.info(
            "{} | entering {} | regime={} p={:.2f} notional=${:.2f} size={} @ {:.2f}",
            self.coin, side, target.label, target_proba, notional, size, mid,
        )
        result = self.orders.market_order(self.coin, side, size)
        if result.success:
            self._in_position = True
            self._entry_price = mid
            self._entry_bar_index = self._bar_index
            self._position_side = side
            self._target_regime = target
            self.risk.register_order()

    async def _maybe_exit(self, snap, viterbi_label: Regime | None, mid: float) -> None:
        reasons: list[str] = []
        entry_price = self._entry_price
        target = self._target_regime
        if entry_price is None or target is None:
            return

        held_bars = self._bar_index - self._entry_bar_index

        # P&L-based hard stops fire any time, regardless of min_hold_bars.
        direction = 1.0 if self._position_side == Side.BUY else -1.0
        pnl_pct = direction * (mid - entry_price) / entry_price
        if pnl_pct < -self.cfg.stop_loss_pct:
            reasons.append(f"stop loss hit ({pnl_pct:.2%})")
        if self.cfg.take_profit_pct is not None and pnl_pct > self.cfg.take_profit_pct:
            reasons.append(f"take profit hit ({pnl_pct:.2%})")

        # Regime/time exits are gated behind min_hold_bars.
        if not reasons and held_bars >= self.cfg.min_hold_bars:
            if self.cfg.signal_mode == "viterbi":
                if viterbi_label is not None and viterbi_label != target:
                    reasons.append(f"viterbi flipped ({viterbi_label.label})")
            else:
                held_proba = float(snap.proba[int(target)])
                if held_proba < self.cfg.exit_proba:
                    reasons.append(f"regime weakening (P({target.label})={held_proba:.2f})")
                opposite = Regime.BEAR if target == Regime.BULL else Regime.BULL
                if snap.proba[int(opposite)] >= self.cfg.entry_proba:
                    reasons.append(
                        f"opposing regime dominant (P({opposite.label})={snap.proba[int(opposite)]:.2f})"
                    )
            if not reasons and held_bars >= self.cfg.max_hold_bars:
                reasons.append(f"max hold reached ({held_bars} bars)")

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
        close_side = Side.SELL if self._position_side == Side.BUY else Side.BUY
        result = self.orders.market_order(self.coin, close_side, size)
        if result.success:
            self._reset()

    def _reset(self) -> None:
        self._last_exit_regime = self._target_regime
        self._last_exit_bar_index = self._bar_index
        self._in_position = False
        self._entry_price = None
        self._target_regime = None
        self._position_side = None
        self.risk.release_order()

    # ------------------------------------------------------------------ #
    #  Market data                                                         #
    # ------------------------------------------------------------------ #
    def _fetch_recent_candles(self, n_bars: int) -> list[dict]:
        """Pull the most recent n_bars hourly candles from Hyperliquid."""
        seconds_per_bar = _interval_seconds(self.cfg.candle_interval)
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - n_bars * seconds_per_bar * 1000
        return self._candles_snapshot(start_ms, end_ms)

    def _poll_new_bars(self) -> list[dict]:
        """Return any closed bars strictly newer than the last one we processed."""
        if self._last_bar_open_ms is None:
            return []
        seconds_per_bar = _interval_seconds(self.cfg.candle_interval)
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = self._last_bar_open_ms + 1
        candles = self._candles_snapshot(start_ms, end_ms)
        # Drop the still-forming bar (open time too close to now).
        cutoff_ms = end_ms - seconds_per_bar * 1000
        closed = [c for c in candles if int(c["t"]) <= cutoff_ms]
        if closed:
            self._last_bar_open_ms = int(closed[-1]["t"])
        return closed

    def _candles_snapshot(self, start_ms: int, end_ms: int) -> list[dict]:
        try:
            return self.orders.info.candles_snapshot(
                self.coin, self.cfg.candle_interval, start_ms, end_ms,
            ) or []
        except Exception as exc:
            logger.warning("{} | candles_snapshot failed: {}", self.coin, exc)
            return []


def _interval_seconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    multiplier = {"m": 60, "h": 3600, "d": 86400}.get(unit)
    if multiplier is None:
        raise ValueError(f"Unsupported candle interval: {interval}")
    return value * multiplier
