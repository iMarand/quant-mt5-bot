"""Typed contracts shared by every layer.

These dataclasses are the *only* thing layers exchange. Architecture §1.2:
everything is a tool with typed inputs/outputs, so the orchestration layer can
recompose tools without knowing their internals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------
# Timeframes
# --------------------------------------------------------------------------

#: Canonical timeframe names -> minutes. Order matters (ascending).
TIMEFRAMES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def tf_minutes(tf: str) -> int:
    try:
        return TIMEFRAMES[tf]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown timeframe {tf!r}, expected one of {list(TIMEFRAMES)}") from exc


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(ts: datetime) -> datetime:
    """Normalize any datetime to tz-aware UTC (architecture §3.4)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Impact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"
    UNKNOWN = "unknown"

    @property
    def weight(self) -> float:
        return {"low": 0.25, "medium": 0.6, "high": 1.0, "holiday": 0.0, "unknown": 0.1}[self.value]


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

    @property
    def sign(self) -> int:
        return {"long": 1, "short": -1, "flat": 0}[self.value]


class Regime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    NEWS_WINDOW = "news_window"
    HIGH_VOL = "high_vol"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


# --------------------------------------------------------------------------
# Data records
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Candle:
    symbol: str
    timeframe: str
    ts: datetime  # bar OPEN time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0


@dataclass(slots=True)
class CalendarEvent:
    """One economic-calendar row, upserted in place as forecast -> actual lands."""

    event_id: str
    source: str
    currency: str
    name: str
    ts_utc: datetime
    impact: Impact = Impact.UNKNOWN
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    # -- derived on ingest (architecture §3.1) ------------------------------
    @property
    def surprise(self) -> float | None:
        if self.actual is None or self.forecast is None:
            return None
        return self.actual - self.forecast

    @property
    def revision(self) -> float | None:
        if self.actual is None or self.previous is None:
            return None
        return self.actual - self.previous

    def minutes_to_release(self, now: datetime | None = None) -> float:
        now = as_utc(now or utcnow())
        return (as_utc(self.ts_utc) - now).total_seconds() / 60.0

    @property
    def is_released(self) -> bool:
        return self.actual is not None


@dataclass(slots=True)
class Signal:
    """Output contract of the prediction engine (architecture §5)."""

    instrument: str
    timeframe: str
    ts: datetime
    direction: Direction
    confidence: float  # calibrated-ish P(direction correct), 0..1
    horizon_min: int
    regime: Regime
    driving_features: dict[str, float] = field(default_factory=dict)
    model_version: str = "rules-v0"
    features: dict[str, float] = field(default_factory=dict)  # full snapshot, journaled
    #: Name(s) of the setup(s) that fired, joined by '+'. Empty means no trigger.
    setup: str = ""
    #: Human-readable "why", carried into the journal (§1.6).
    rationale: str = ""
    #: Trading mode that produced this signal ("scalp"/"swing").
    mode: str = ""
    #: Sessions live when this signal was formed, e.g. "london+newyork".
    session: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "ts": as_utc(self.ts).isoformat(),
            "symbol": self.instrument,
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "confidence": float(self.confidence),
            "horizon_min": int(self.horizon_min),
            "regime": self.regime.value,
            "driving_features": json.dumps(self.driving_features),
            "features": json.dumps(self.features),
            "model_version": self.model_version,
            "setup": self.setup,
            "rationale": self.rationale,
            "session": self.session,
            "mode": self.mode,
        }


@dataclass(slots=True)
class OrderRequest:
    symbol: str
    side: Direction
    volume: float  # lots
    sl: float | None = None
    tp: float | None = None
    comment: str = "quantbot"
    prediction_id: int | None = None


@dataclass(slots=True)
class Position:
    ticket: int
    symbol: str
    side: Direction
    volume: float
    entry_price: float
    sl: float | None
    tp: float | None
    opened_at: datetime
    prediction_id: int | None = None
    profit: float = 0.0
    #: Broker-side comment, e.g. 'qb1234-scalp'. Used to re-adopt a
    #: position after a restart and hand it back to the right mode.
    comment: str = ""
    # trailing-stop bookkeeping (addendum §B)
    breakeven_done: bool = False
    partial_done: bool = False


@dataclass(slots=True)
class Fill:
    ticket: int
    symbol: str
    side: Direction
    volume: float
    price: float
    ts: datetime
    status: OrderStatus
    reason: str = ""


@dataclass(slots=True)
class Tick:
    symbol: str
    ts: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(slots=True, frozen=True)
class SymbolSpec:
    """Instrument metadata needed by the risk layer to size correctly."""

    symbol: str
    digits: int = 5
    point: float = 0.00001
    contract_size: float = 100_000.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    tick_value: float = 1.0  # account currency per tick per 1.0 lot
    tick_size: float = 0.00001

    def round_volume(self, volume: float) -> float:
        steps = round(volume / self.volume_step)
        vol = steps * self.volume_step
        vol = min(max(vol, self.volume_min), self.volume_max)
        return round(vol, 8)

    def round_price(self, price: float) -> float:
        return round(price, self.digits)


def dumps(obj: Any) -> str:
    """JSON-dump a dataclass/dict with datetime support."""

    def _default(o: Any) -> Any:
        if isinstance(o, datetime):
            return as_utc(o).isoformat()
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        return str(o)

    return json.dumps(obj, default=_default)
