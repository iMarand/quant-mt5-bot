"""Execution abstraction (architecture §6).

One interface, implemented first against paper/demo. Swapping demo -> live must
be a config change, not a rewrite — so nothing above this module may import a
broker implementation directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ...contracts import Direction, Fill, Position, SymbolSpec, Tick


class Broker(ABC):
    name: str = "abstract"

    #: False for anything that can touch real money. The runner refuses to
    #: trade a broker with is_demo=False unless broker.allow_live is set.
    is_demo: bool = True

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def equity(self) -> float: ...

    @abstractmethod
    def balance(self) -> float: ...

    @abstractmethod
    def symbol_spec(self, symbol: str) -> SymbolSpec: ...

    @abstractmethod
    def tick(self, symbol: str) -> Tick: ...

    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[Position]: ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: Direction,
        volume: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "quantbot",
    ) -> Fill: ...

    @abstractmethod
    def close_position(self, ticket: int, volume: float | None = None, reason: str = "") -> Fill: ...

    @abstractmethod
    def modify_position(
        self, ticket: int, sl: float | None = None, tp: float | None = None
    ) -> bool: ...

    def __enter__(self) -> Broker:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
