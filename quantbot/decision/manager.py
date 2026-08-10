"""Open-position management — dynamic SL/TP (addendum §B).

Stages as a trade develops:
  0. `max_bars_in_trade`      -> close a trade that has gone nowhere (scalps)
  1. at `breakeven_at_r`      -> stop moves to entry + `breakeven_buffer_r`,
                                 so the worst case becomes a small win, not a
                                 scratch that still pays the spread
  2. at `partial_tp_at_r`     -> close `partial_tp_fraction` of the position
  3. beyond `trail_start_r`   -> trail the stop `trail_atr_mult` ATR behind price

Stops only ever move in the favourable direction. A stop that can loosen is not
a stop.

When several modes run at once each manager only touches the tickets it opened,
so a scalp manager never trails a swing position onto a 5-minute stop.
"""

from __future__ import annotations

import logging

from ..config import RiskConfig
from ..contracts import Direction, Position, SymbolSpec, as_utc, utcnow
from .execution.base import Broker

log = logging.getLogger(__name__)


class TradeManager:
    def __init__(
        self,
        cfg: RiskConfig,
        broker: Broker,
        max_bars_in_trade: int = 0,
        bar_minutes: int = 0,
    ) -> None:
        self.cfg = cfg
        self.broker = broker
        #: ticket -> risk distance at entry, needed to express progress in R.
        self._risk_distance: dict[int, float] = {}
        #: Scalps that go nowhere still pay spread and tie up risk budget.
        self.max_bars_in_trade = max_bars_in_trade
        self.bar_minutes = bar_minutes
        #: Tickets this manager opened, so one mode never manages another's.
        self._owned: set[int] = set()

    def register(self, ticket: int, risk_distance: float) -> None:
        self._risk_distance[ticket] = risk_distance
        self._owned.add(ticket)

    def forget(self, ticket: int) -> None:
        self._risk_distance.pop(ticket, None)
        self._owned.discard(ticket)

    def owns(self, ticket: int) -> bool:
        """With several modes running, only manage positions this mode opened."""
        return ticket in self._owned

    def adopt(self, pos: Position) -> bool:
        """Take over a position this process did not open.

        Without this, any position that survives a restart belongs to no
        manager and is silently never trailed again — the stop just sits where
        it was placed. Risk is recovered from the existing stop distance.
        """
        if pos.ticket in self._owned:
            return False
        if pos.sl is None:
            # No stop to infer risk from; the caller must repair that first.
            self._owned.add(pos.ticket)
            log.warning("adopted #%s (%s) with NO stop loss", pos.ticket, pos.symbol)
            return True
        risk = abs(pos.entry_price - pos.sl)
        if risk <= 0:
            return False
        self._risk_distance[pos.ticket] = risk
        self._owned.add(pos.ticket)
        # A stop already better than entry means breakeven has been done before.
        moved = (
            pos.sl >= pos.entry_price
            if pos.side is Direction.LONG
            else pos.sl <= pos.entry_price
        )
        pos.breakeven_done = moved
        log.info(
            "adopted #%s %s %s: risk=%.5f, breakeven_done=%s",
            pos.ticket, pos.symbol, pos.side.value, risk, moved,
        )
        return True

    # -- main entry point --------------------------------------------------
    def manage(self, positions: list[Position], atr_by_symbol: dict[str, float]) -> list[str]:
        actions: list[str] = []
        for pos in positions:
            if self._owned and not self.owns(pos.ticket):
                continue
            try:
                actions.extend(self._manage_one(pos, atr_by_symbol.get(pos.symbol)))
            except Exception as exc:
                log.error("manage %s failed: %s", pos.ticket, exc)
        return actions

    def _manage_one(self, pos: Position, atr_value: float | None) -> list[str]:
        spec = self.broker.symbol_spec(pos.symbol)
        tick = self.broker.tick(pos.symbol)
        price = tick.bid if pos.side is Direction.LONG else tick.ask

        risk = self._risk_distance.get(pos.ticket)
        if risk is None or risk <= 0:
            # Recover the original risk from the stop if we lost our bookkeeping
            # (e.g. process restart, or a position adopted from a previous run).
            risk = abs(pos.entry_price - pos.sl) if pos.sl else None
        if not risk or risk <= 0:
            return []

        r_now = (price - pos.entry_price) * pos.side.sign / risk
        actions: list[str] = []

        # 0. time stop: a scalp that hasn't worked within its horizon is not
        # going to; holding it just pays spread and blocks the risk budget.
        if self.max_bars_in_trade and self.bar_minutes:
            held_min = (utcnow() - as_utc(pos.opened_at)).total_seconds() / 60
            limit_min = self.max_bars_in_trade * self.bar_minutes
            if held_min >= limit_min and r_now < self.cfg.breakeven_at_r:
                fill = self.broker.close_position(
                    pos.ticket, reason="time_stop"
                )
                if fill.status.value == "closed":
                    self.forget(pos.ticket)
                    return [
                        f"{pos.ticket}: time stop after {held_min:.0f} min "
                        f"({r_now:+.2f}R) — went nowhere"
                    ]

        # 1. breakeven + buffer — lock in a small profit, not a scratch.
        if not pos.breakeven_done and r_now >= self.cfg.breakeven_at_r:
            buffer = risk * self.cfg.breakeven_buffer_r * pos.side.sign
            be = spec.round_price(pos.entry_price + buffer)
            if self._tighten(pos, be, spec):
                pos.breakeven_done = True
                actions.append(
                    f"{pos.ticket}: stop -> breakeven+{self.cfg.breakeven_buffer_r:.2f}R "
                    f"({be}) at {r_now:.2f}R — trade is now risk-free"
                )

        # 2. partial scale-out
        if (
            not pos.partial_done
            and self.cfg.partial_tp_fraction > 0
            and r_now >= self.cfg.partial_tp_at_r
        ):
            volume = spec.round_volume(pos.volume * self.cfg.partial_tp_fraction)
            if volume >= spec.volume_min and volume < pos.volume:
                fill = self.broker.close_position(pos.ticket, volume=volume, reason="partial_tp")
                if fill.status.value == "closed":
                    pos.partial_done = True
                    actions.append(f"{pos.ticket}: scaled out {volume} at {r_now:.2f}R")

        # 3. volatility trail
        if atr_value and atr_value > 0 and r_now >= self.cfg.trail_start_r:
            trail = atr_value * self.cfg.trail_atr_mult
            new_sl = price - trail if pos.side is Direction.LONG else price + trail
            new_sl = spec.round_price(new_sl)
            if self._tighten(pos, new_sl, spec):
                actions.append(f"{pos.ticket}: trail -> {new_sl} ({r_now:.2f}R)")

        return actions

    def _tighten(self, pos: Position, new_sl: float, spec: SymbolSpec) -> bool:
        """Apply `new_sl` only if it reduces risk. Never loosens a stop."""
        if pos.sl is not None:
            improves = new_sl > pos.sl if pos.side is Direction.LONG else new_sl < pos.sl
            if not improves:
                return False
        # A stop must stay on the losing side of the current price, or the
        # broker rejects it (and it would be an instant-close order anyway).
        tick = self.broker.tick(pos.symbol)
        price = tick.bid if pos.side is Direction.LONG else tick.ask
        wrong_side = new_sl >= price if pos.side is Direction.LONG else new_sl <= price
        if wrong_side:
            return False
        if self.broker.modify_position(pos.ticket, sl=new_sl, tp=pos.tp):
            pos.sl = new_sl
            return True
        return False
