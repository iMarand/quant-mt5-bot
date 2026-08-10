"""In-process paper broker.

Used for backtests and for running the full loop with no broker connection at
all. Models spread and commission, and — importantly — checks SL/TP against bar
*highs and lows*, not just closes, so stops behave the way they would live.
"""

from __future__ import annotations

import logging
from datetime import datetime
from itertools import count

from ...contracts import (
    Direction,
    Fill,
    OrderStatus,
    Position,
    SymbolSpec,
    Tick,
    utcnow,
)
from .base import Broker

log = logging.getLogger(__name__)


class PaperBroker(Broker):
    name = "paper"
    is_demo = True

    def __init__(
        self,
        balance: float = 10_000.0,
        spread_points: float = 12.0,
        commission_per_lot: float = 3.5,
        specs: dict[str, SymbolSpec] | None = None,
    ) -> None:
        self._balance = balance
        self._start_balance = balance
        self.spread_points = spread_points
        self.commission_per_lot = commission_per_lot
        self._specs: dict[str, SymbolSpec] = dict(specs or {})
        self._positions: dict[int, Position] = {}
        self._tickets = count(1)
        self._prices: dict[str, float] = {}
        self._now: datetime = utcnow()
        self.closed_trades: list[dict] = []

    # -- test/backtest driving --------------------------------------------
    def set_price(self, symbol: str, price: float, ts: datetime | None = None) -> None:
        self._prices[symbol] = price
        if ts:
            self._now = ts

    def feed_bar(
        self, symbol: str, high: float, low: float, close: float, ts: datetime
    ) -> list[Fill]:
        """Advance one bar and trigger any SL/TP touched inside it."""
        self._now = ts
        self._prices[symbol] = close
        fills: list[Fill] = []
        for ticket, pos in list(self._positions.items()):
            if pos.symbol != symbol:
                continue
            hit = self._barrier_hit(pos, high, low)
            if hit is not None:
                price, reason = hit
                fills.append(self.close_position(ticket, reason=reason, price=price))
        return fills

    def _barrier_hit(self, pos: Position, high: float, low: float):
        sl, tp = pos.sl, pos.tp
        if pos.side is Direction.LONG:
            hit_sl = sl is not None and low <= sl
            hit_tp = tp is not None and high >= tp
        else:
            hit_sl = sl is not None and high >= sl
            hit_tp = tp is not None and low <= tp
        # If a bar spans both barriers the intrabar order is unknowable; assume
        # the stop went first. Optimism here is how backtests lie.
        if hit_sl:
            return pos.sl, "stop_loss"
        if hit_tp:
            return pos.tp, "take_profit"
        return None

    # -- Broker interface --------------------------------------------------
    def connect(self) -> None:
        log.info("paper broker: balance %.2f", self._balance)

    def disconnect(self) -> None:
        pass

    def register_spec(self, spec: SymbolSpec) -> None:
        self._specs[spec.symbol] = spec

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        return self._specs.setdefault(symbol, SymbolSpec(symbol=symbol))

    def balance(self) -> float:
        return self._balance

    def equity(self) -> float:
        return self._balance + sum(self._unrealized(p) for p in self._positions.values())

    def _unrealized(self, pos: Position) -> float:
        price = self._prices.get(pos.symbol)
        if price is None:
            return 0.0
        return self._pnl(pos, price)

    def _pnl(self, pos: Position, exit_price: float) -> float:
        spec = self.symbol_spec(pos.symbol)
        ticks = (exit_price - pos.entry_price) / spec.tick_size * pos.side.sign
        return ticks * spec.tick_value * pos.volume

    def tick(self, symbol: str) -> Tick:
        spec = self.symbol_spec(symbol)
        mid = self._prices.get(symbol)
        if mid is None:
            raise RuntimeError(f"paper broker has no price for {symbol}; call set_price/feed_bar")
        half = self.spread_points * spec.point / 2
        return Tick(symbol=symbol, ts=self._now, bid=mid - half, ask=mid + half)

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        out = list(self._positions.values())
        if symbol:
            out = [p for p in out if p.symbol == symbol]
        for p in out:
            p.profit = self._unrealized(p)
        return out

    def place_order(
        self,
        symbol: str,
        side: Direction,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "quantbot",
    ) -> Fill:
        if side is Direction.FLAT:
            return Fill(0, symbol, side, 0, 0, self._now, OrderStatus.REJECTED, "flat side")
        try:
            t = self.tick(symbol)
        except RuntimeError as exc:
            return Fill(0, symbol, side, volume, 0, self._now, OrderStatus.REJECTED, str(exc))
        price = t.ask if side is Direction.LONG else t.bid
        ticket = next(self._tickets)
        self._positions[ticket] = Position(
            ticket=ticket,
            symbol=symbol,
            side=side,
            volume=volume,
            entry_price=price,
            sl=sl,
            tp=tp,
            opened_at=self._now,
            comment=comment,
        )
        self._balance -= self.commission_per_lot * volume
        return Fill(ticket, symbol, side, volume, price, self._now, OrderStatus.OPEN, comment)

    def close_position(
        self,
        ticket: int,
        volume: float | None = None,
        reason: str = "",
        price: float | None = None,
    ) -> Fill:
        pos = self._positions.get(ticket)
        if pos is None:
            return Fill(ticket, "", Direction.FLAT, 0, 0, self._now, OrderStatus.REJECTED, "no position")
        if price is None:
            t = self.tick(pos.symbol)
            price = t.bid if pos.side is Direction.LONG else t.ask
        volume = min(volume or pos.volume, pos.volume)
        closing = Position(
            ticket=pos.ticket,
            symbol=pos.symbol,
            side=pos.side,
            volume=volume,
            entry_price=pos.entry_price,
            sl=pos.sl,
            tp=pos.tp,
            opened_at=pos.opened_at,
            prediction_id=pos.prediction_id,
        )
        profit = self._pnl(closing, price) - self.commission_per_lot * volume
        self._balance += profit
        self.closed_trades.append(
            {
                "ticket": ticket,
                "symbol": pos.symbol,
                "side": pos.side.value,
                "volume": volume,
                "entry_price": pos.entry_price,
                "exit_price": price,
                "opened_at": pos.opened_at,
                "closed_at": self._now,
                "profit": profit,
                "exit_reason": reason or "manual",
                "prediction_id": pos.prediction_id,
            }
        )
        if volume >= pos.volume - 1e-9:
            del self._positions[ticket]
        else:
            pos.volume -= volume  # partial scale-out (addendum §B)
        return Fill(ticket, pos.symbol, pos.side, volume, price, self._now, OrderStatus.CLOSED, reason)

    def modify_position(self, ticket: int, sl: float | None = None, tp: float | None = None) -> bool:
        pos = self._positions.get(ticket)
        if pos is None:
            return False
        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp
        return True
