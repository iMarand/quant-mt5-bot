"""Connector protocols — every data source is a callable tool (§1.2)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..contracts import Candle, CalendarEvent, SymbolSpec, Tick


@runtime_checkable
class MarketDataConnector(Protocol):
    name: str

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def symbol_spec(self, symbol: str) -> SymbolSpec: ...

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
        end: datetime | None = None,
    ) -> list[Candle]: ...

    def fetch_tick(self, symbol: str) -> Tick: ...


@runtime_checkable
class CalendarConnector(Protocol):
    name: str

    def fetch_events(self, force: bool = False) -> list[CalendarEvent]: ...
