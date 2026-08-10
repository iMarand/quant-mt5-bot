"""MetaTrader 5 market-data connector (architecture §3.3).

MT5 is the price feed *and* (via `decision.execution.mt5_broker`) the demo
execution venue, so both share this connection. `mt5.initialize()` is global and
process-wide, hence the module-level refcount.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from ..contracts import Candle, SymbolSpec, Tick, as_utc, tf_minutes

log = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INIT_COUNT = 0
#: The -6 remedy is long. Attach it to the first failure only — repeating it for
#: every symbol/timeframe buries the actual errors.
_HINT_SHOWN = False


class MT5Error(RuntimeError):
    pass


def _mt5():
    try:
        import MetaTrader5 as mt5  # noqa: N813
    except ImportError as exc:  # pragma: no cover
        raise MT5Error(
            "MetaTrader5 package not installed. `pip install MetaTrader5` (Windows only)."
        ) from exc
    return mt5


def mt5_timeframe(tf: str):
    mt5 = _mt5()
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if tf not in mapping:
        raise ValueError(f"timeframe {tf!r} not supported by MT5 adapter")
    return mapping[tf]


def mt5_initialize(
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
) -> None:
    """Idempotent, refcounted terminal init. Raises MT5Error with the real reason."""
    global _INIT_COUNT, _HINT_SHOWN
    mt5 = _mt5()
    with _INIT_LOCK:
        if _INIT_COUNT == 0:
            kwargs: dict = {}
            if path:
                kwargs["path"] = path
            if login:
                kwargs.update(login=int(login), password=password or "", server=server or "")
            ok = mt5.initialize(**kwargs)
            if not ok:
                code, msg = mt5.last_error()
                hint = ""
                if code == -6 and not _HINT_SHOWN:
                    _HINT_SHOWN = True
                    # -6 is an IPC authorization refusal, NOT "not logged in" —
                    # it fires even with an account connected in the terminal.
                    hint = (
                        "\n  This code covers several unrelated causes, so it is not diagnostic\n"
                        "  on its own. Run `python -m quantbot doctor` — it reads the terminal's\n"
                        "  own log and reports the actual reason. Common ones:\n"
                        "   1. The account is not authorized (the Navigator panel can still show\n"
                        "      it). An expired demo logs 'Invalid account'.\n"
                        "   2. 'Algo Trading' is off — toolbar button, or Tools > Options >\n"
                        "      Expert Advisors > 'Allow algorithmic trading'.\n"
                        "   3. MT5 elevated while Python is not, or vice versa."
                    )
                raise MT5Error(f"mt5.initialize failed: ({code}) {msg}{hint}")
        _INIT_COUNT += 1


def mt5_shutdown() -> None:
    global _INIT_COUNT
    with _INIT_LOCK:
        _INIT_COUNT = max(0, _INIT_COUNT - 1)
        if _INIT_COUNT == 0:
            try:
                _mt5().shutdown()
            except Exception:  # pragma: no cover
                pass


class MT5MarketData:
    """Multi-timeframe OHLCV puller."""

    name = "mt5"

    #: Retries for a timeframe the terminal is still downloading.
    max_fetch_attempts = 4
    fetch_retry_delay_s = 3.0

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ) -> None:
        self._creds = dict(login=login, password=password, server=server, path=path)
        self._connected = False
        self._specs: dict[str, SymbolSpec] = {}

    # -- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        if self._connected:
            return
        mt5_initialize(**self._creds)
        self._connected = True
        mt5 = _mt5()
        acct = mt5.account_info()
        if acct is not None:
            log.info(
                "MT5 connected: login=%s server=%s balance=%.2f %s trade_mode=%s",
                acct.login,
                acct.server,
                acct.balance,
                acct.currency,
                _trade_mode_name(acct.trade_mode),
            )

    def disconnect(self) -> None:
        if self._connected:
            mt5_shutdown()
            self._connected = False

    def __enter__(self) -> MT5MarketData:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()

    # -- metadata ----------------------------------------------------------
    def symbol_spec(self, symbol: str) -> SymbolSpec:
        if symbol in self._specs:
            return self._specs[symbol]
        self.connect()
        mt5 = _mt5()
        info = mt5.symbol_info(symbol)
        if info is None:
            # Symbol may exist but be hidden from Market Watch.
            if not mt5.symbol_select(symbol, True):
                raise MT5Error(f"symbol {symbol!r} not available on this account")
            info = mt5.symbol_info(symbol)
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
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

    def account_is_demo(self) -> bool:
        self.connect()
        acct = _mt5().account_info()
        if acct is None:
            raise MT5Error("no account info; terminal not logged in")
        # 0 = DEMO, 1 = CONTEST, 2 = REAL
        return acct.trade_mode in (0, 1)

    def account_info(self) -> dict:
        self.connect()
        acct = _mt5().account_info()
        return acct._asdict() if acct else {}

    # -- data --------------------------------------------------------------
    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
        end: datetime | None = None,
    ) -> list[Candle]:
        self.connect()
        mt5 = _mt5()
        tf = mt5_timeframe(timeframe)
        self.symbol_spec(symbol)  # ensures the symbol is selected
        # The first request for a timeframe the terminal hasn't cached starts a
        # background download and fails outright. Retrying after a moment is
        # the documented way through; without it a fresh account silently
        # ingests nothing for its higher timeframes.
        rates = None
        for attempt in range(self.max_fetch_attempts):
            if end is None:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
            else:
                rates = mt5.copy_rates_from(symbol, tf, as_utc(end), int(count))
            if rates is not None and len(rates) > 0:
                break
            code, msg = mt5.last_error()
            if attempt < self.max_fetch_attempts - 1:
                log.info(
                    "%s %s not ready ((%s) %s); terminal is likely still downloading, "
                    "retrying in %.0fs",
                    symbol, timeframe, code, msg, self.fetch_retry_delay_s,
                )
                time.sleep(self.fetch_retry_delay_s)

        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            log.warning(
                "no rates for %s %s after %d attempts: (%s) %s",
                symbol, timeframe, self.max_fetch_attempts, code, msg,
            )
            return []
        out: list[Candle] = []
        for r in rates:
            out.append(
                Candle(
                    symbol=symbol,
                    timeframe=timeframe,
                    # MT5 bar times are broker-server time expressed as a UTC epoch.
                    ts=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["tick_volume"]),
                    spread=float(r["spread"]),
                )
            )
        # Drop the still-forming last bar: acting on a partial candle is lookahead.
        if out and _is_forming(out[-1], timeframe):
            out = out[:-1]
        return out

    def fetch_tick(self, symbol: str) -> Tick:
        self.connect()
        mt5 = _mt5()
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

    def symbols(self, pattern: str = "*") -> list[str]:
        self.connect()
        syms = _mt5().symbols_get(pattern) or []
        return [s.name for s in syms]


def _is_forming(candle: Candle, timeframe: str) -> bool:
    now = datetime.now(timezone.utc)
    return (now - as_utc(candle.ts)).total_seconds() < tf_minutes(timeframe) * 60


def _trade_mode_name(mode: int) -> str:
    return {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(mode, f"UNKNOWN({mode})")
