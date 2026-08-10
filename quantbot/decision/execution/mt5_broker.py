"""MetaTrader 5 execution adapter — demo accounts by default.

`connect()` refuses to proceed on a REAL account unless `allow_live=True` is
passed explicitly. That check is the whole point of §1.1: the safety property is
enforced in code, not in a runbook.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ...connectors.mt5_market import MT5Error, _mt5, mt5_initialize, mt5_shutdown
from ...contracts import Direction, Fill, OrderStatus, Position, SymbolSpec, Tick, utcnow
from .base import Broker

log = logging.getLogger(__name__)


class LiveTradingBlocked(RuntimeError):
    pass


class MT5Broker(Broker):
    name = "mt5"

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        magic: int = 990101,
        deviation: int = 20,
        allow_live: bool = False,
    ) -> None:
        self._creds = dict(login=login, password=password, server=server, path=path)
        self.magic = magic
        self.deviation = deviation
        self.allow_live = allow_live
        self._connected = False
        self.is_demo = True
        self._specs: dict[str, SymbolSpec] = {}

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        mt5_initialize(**self._creds)
        self._connected = True
        acct = _mt5().account_info()
        if acct is None:
            mt5_shutdown()
            self._connected = False
            raise MT5Error("connected to terminal but no account is logged in")
        self.is_demo = acct.trade_mode in (0, 1)  # 0 DEMO, 1 CONTEST, 2 REAL
        if not self.is_demo and not self.allow_live:
            mt5_shutdown()
            self._connected = False
            raise LiveTradingBlocked(
                f"account {acct.login} on {acct.server} is a REAL account. QuantBot is "
                "demo-first: set broker.allow_live=true only after the promotion gate "
                "(architecture §8.3) has actually been evaluated."
            )
        log.info(
            "MT5 broker ready: %s account %s @ %s, equity %.2f %s",
            "DEMO" if self.is_demo else "LIVE",
            acct.login,
            acct.server,
            acct.equity,
            acct.currency,
        )

    def disconnect(self) -> None:
        if self._connected:
            mt5_shutdown()
            self._connected = False

    def _require(self):
        if not self._connected:
            self.connect()
        return _mt5()

    # -- account -----------------------------------------------------------
    def equity(self) -> float:
        acct = self._require().account_info()
        return float(acct.equity) if acct else 0.0

    def balance(self) -> float:
        acct = self._require().account_info()
        return float(acct.balance) if acct else 0.0

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        if symbol in self._specs:
            return self._specs[symbol]
        mt5 = self._require()
        info = mt5.symbol_info(symbol)
        if info is None or not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"symbol {symbol!r} unavailable")
        spec = SymbolSpec(
            symbol=symbol,
            digits=info.digits,
            point=info.point,
            contract_size=info.trade_contract_size,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            tick_value=info.trade_tick_value or 1.0,
            tick_size=info.trade_tick_size or info.point,
        )
        self._specs[symbol] = spec
        return spec

    def tick(self, symbol: str) -> Tick:
        mt5 = self._require()
        self.symbol_spec(symbol)
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise MT5Error(f"no tick for {symbol}")
        return Tick(
            symbol=symbol,
            ts=datetime.fromtimestamp(int(t.time), tz=timezone.utc),
            bid=float(t.bid),
            ask=float(t.ask),
        )

    # -- positions ---------------------------------------------------------
    def get_positions(self, symbol: str | None = None) -> list[Position]:
        mt5 = self._require()
        raw = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        out: list[Position] = []
        for p in raw or []:
            if p.magic and p.magic != self.magic:
                continue  # not ours — never manage a human's manual trade
            out.append(
                Position(
                    ticket=int(p.ticket),
                    symbol=p.symbol,
                    side=Direction.LONG if p.type == mt5.POSITION_TYPE_BUY else Direction.SHORT,
                    volume=float(p.volume),
                    entry_price=float(p.price_open),
                    sl=float(p.sl) or None,
                    tp=float(p.tp) or None,
                    opened_at=datetime.fromtimestamp(int(p.time), tz=timezone.utc),
                    profit=float(p.profit),
                    comment=str(getattr(p, "comment", "") or ""),
                )
            )
        return out

    # -- orders ------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        side: Direction,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "quantbot",
    ) -> Fill:
        mt5 = self._require()
        if side is Direction.FLAT:
            return Fill(0, symbol, side, 0, 0, utcnow(), OrderStatus.REJECTED, "flat side")
        spec = self.symbol_spec(symbol)
        t = self.tick(symbol)
        is_buy = side is Direction.LONG
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": spec.round_volume(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": t.ask if is_buy else t.bid,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment[:31],  # MT5 truncates silently past 31 chars
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(symbol),
        }
        if sl:
            request["sl"] = spec.round_price(sl)
        if tp:
            request["tp"] = spec.round_price(tp)

        result = mt5.order_send(request)
        if result is None:
            code, msg = mt5.last_error()
            return Fill(0, symbol, side, volume, 0, utcnow(), OrderStatus.REJECTED, f"({code}) {msg}")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return Fill(
                0,
                symbol,
                side,
                volume,
                0,
                utcnow(),
                OrderStatus.REJECTED,
                f"retcode {result.retcode}: {result.comment}",
            )
        return Fill(
            ticket=int(result.order),
            symbol=symbol,
            side=side,
            volume=float(result.volume),
            price=float(result.price),
            ts=utcnow(),
            status=OrderStatus.OPEN,
            reason=comment,
        )

    def close_position(self, ticket: int, volume: float | None = None, reason: str = "") -> Fill:
        mt5 = self._require()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return Fill(ticket, "", Direction.FLAT, 0, 0, utcnow(), OrderStatus.REJECTED, "not found")
        p = positions[0]
        is_buy = p.type == mt5.POSITION_TYPE_BUY
        t = self.tick(p.symbol)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": float(volume or p.volume),
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": int(ticket),
            "price": t.bid if is_buy else t.ask,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": (reason or "close")[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(p.symbol),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            detail = "send failed" if result is None else f"retcode {result.retcode}"
            return Fill(
                ticket, p.symbol, Direction.LONG if is_buy else Direction.SHORT,
                0, 0, utcnow(), OrderStatus.REJECTED, detail,
            )
        return Fill(
            ticket=ticket,
            symbol=p.symbol,
            side=Direction.LONG if is_buy else Direction.SHORT,
            volume=float(result.volume),
            price=float(result.price),
            ts=utcnow(),
            status=OrderStatus.CLOSED,
            reason=reason,
        )

    def modify_position(self, ticket: int, sl: float | None = None, tp: float | None = None) -> bool:
        mt5 = self._require()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        p = positions[0]
        spec = self.symbol_spec(p.symbol)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": p.symbol,
            "position": int(ticket),
            "sl": spec.round_price(sl if sl is not None else p.sl),
            "tp": spec.round_price(tp if tp is not None else p.tp),
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            log.warning(
                "modify %s failed: %s", ticket, result.comment if result else mt5.last_error()
            )
        return ok

    def _filling_mode(self, symbol: str):
        """Brokers differ on which filling policies they accept; pick a valid one."""
        mt5 = self._require()
        info = mt5.symbol_info(symbol)
        modes = getattr(info, "filling_mode", 0) if info else 0
        if modes & 1:  # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        if modes & 2:  # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def deals_since(self, since: datetime) -> list[dict]:
        """Closed deals, used to reconcile the journal with broker truth."""
        mt5 = self._require()
        deals = mt5.history_deals_get(since, utcnow())
        return [d._asdict() for d in (deals or []) if d.magic == self.magic]
