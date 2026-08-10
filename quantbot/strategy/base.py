"""Strategy layer — discrete, named setups that either trigger or don't.

This is the *primary* decision maker. A trade requires a named setup to fire on
explicit, readable conditions; no setup means no trade, however confident any
model might be. The model's role is downstream and advisory (see `book.py`).

Contrast with a model-primary design, which emits a direction on every bar and
therefore trades constantly, paying spread on noise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from ..contracts import Direction


@dataclass
class Setup:
    """One strategy's verdict on one bar."""

    name: str
    direction: Direction
    #: 0..1 — how well the bar matched the rule, not a probability of profit.
    quality: float
    reasons: list[str] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.quality = float(min(max(self.quality, 0.0), 1.0))

    @property
    def is_news(self) -> bool:
        return "news" in self.tags

    def describe(self) -> str:
        return f"{self.name}[{self.direction.value} q={self.quality:.2f}]"


class StrategyContext:
    """Read-only view of the current bar plus the previous one.

    Strategies need "is this fresh?" (a breakout that already broke three bars
    ago is not a breakout), which needs the prior bar. Feature names are
    timeframe-prefixed, so `f("adx", "H1")` resolves `H1_adx`.
    """

    __slots__ = ("row", "prev", "base_tf")

    def __init__(self, row: pd.Series, prev: pd.Series | None, base_tf: str) -> None:
        self.row = row
        self.prev = prev
        self.base_tf = base_tf

    def f(self, name: str, tf: str | None = None, default: float | None = None) -> float | None:
        """Feature value, or `default` when absent/NaN."""
        key = name if tf is None else f"{tf}_{name}"
        if key not in self.row.index:
            return default
        value = self.row[key]
        if value is None or pd.isna(value):
            return default
        return float(value)

    def b(self, name: str, tf: str | None = None) -> float | None:
        """Same, defaulting to the base timeframe."""
        return self.f(name, tf or self.base_tf)

    def prev_f(self, name: str, tf: str | None = None, default: float | None = None):
        if self.prev is None:
            return default
        key = name if tf is None else f"{tf}_{name}"
        if key not in self.prev.index:
            return default
        value = self.prev[key]
        if value is None or pd.isna(value):
            return default
        return float(value)

    def prev_b(self, name: str, tf: str | None = None):
        return self.prev_f(name, tf or self.base_tf)

    def rising(self, name: str, tf: str | None = None) -> bool | None:
        now = self.b(name, tf)
        before = self.prev_b(name, tf)
        if now is None or before is None:
            return None
        return now > before

    def has(self, *names: str) -> bool:
        """True when every named base-timeframe feature is available."""
        return all(self.b(n) is not None for n in names)


class Strategy(ABC):
    """A named setup. Returns a Setup when its conditions fire, else None."""

    name: str = "unnamed"
    tags: set[str] = frozenset()  # type: ignore[assignment]

    def __init__(self, enabled: bool = True, weight: float = 1.0) -> None:
        self.enabled = enabled
        self.weight = weight

    @abstractmethod
    def evaluate(self, ctx: StrategyContext) -> Setup | None: ...

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name} enabled={self.enabled}>"


def scale(value: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] onto [0, 1], clamped. Used for quality scoring."""
    if hi == lo:
        return 0.0
    return float(min(max((value - lo) / (hi - lo), 0.0), 1.0))
