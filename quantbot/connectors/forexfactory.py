"""Forex Factory economic-calendar connector (architecture §3.1).

Uses the *published weekly JSON feed* rather than scraping the HTML calendar —
it is the access path Forex Factory itself offers for automated consumption, it
is cheap to cache, and it keeps us on the right side of §1.5. Every fetch goes
through `FetchPolicy` (robots + rate limit + cache).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone

from ..contracts import CalendarEvent, Impact
from .policy import FetchPolicy

log = logging.getLogger(__name__)

_IMPACT_MAP = {
    "low": Impact.LOW,
    "medium": Impact.MEDIUM,
    "high": Impact.HIGH,
    "holiday": Impact.HOLIDAY,
    "non-economic": Impact.LOW,
}

# "3.2%", "-1.5K", "215K", "1.25M", "0.5%" ... -> float in base units
_NUM_RE = re.compile(r"^\s*(-?[\d,]*\.?\d+)\s*([KMBT%]?)\s*$", re.IGNORECASE)
_MULT = {"": 1.0, "%": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_number(value: object) -> float | None:
    """Parse Forex Factory's display strings into comparable floats.

    Returns None for empty/unavailable fields — importantly *not* 0.0, since a
    missing forecast and a forecast of zero mean very different things to the
    surprise feature.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in {"-", "--", "n/a", "N/A"}:
        return None
    m = _NUM_RE.match(text)
    if not m:
        return None
    number, suffix = m.groups()
    try:
        return float(number.replace(",", "")) * _MULT[suffix.upper()]
    except (ValueError, KeyError):
        return None


def make_event_id(source: str, currency: str, name: str, ts: datetime) -> str:
    """Stable id so a forecast row and its later actual-bearing row collide."""
    key = f"{source}|{currency}|{name.strip().lower()}|{ts.astimezone(timezone.utc).isoformat()}"
    return hashlib.sha1(key.encode()).hexdigest()[:24]


class ForexFactoryCalendar:
    name = "forexfactory"

    def __init__(
        self,
        url: str,
        policy: FetchPolicy,
        currencies: list[str] | None = None,
        cache_minutes: float = 30.0,
    ) -> None:
        self.url = url
        self.policy = policy
        self.currencies = [c.upper() for c in (currencies or [])]
        self.cache_ttl_s = cache_minutes * 60

    def fetch_events(self, force: bool = False) -> list[CalendarEvent]:
        body = self.policy.fetch(self.url, ttl_s=self.cache_ttl_s, force=force)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"calendar feed at {self.url} was not JSON: {exc}") from exc
        return self.parse(payload)

    def parse(self, payload: list[dict]) -> list[CalendarEvent]:
        events: list[CalendarEvent] = []
        for row in payload:
            ccy = str(row.get("country") or row.get("currency") or "").upper()
            if self.currencies and ccy not in self.currencies:
                continue
            ts = _parse_ts(row.get("date"))
            if ts is None:
                continue
            title = str(row.get("title") or row.get("event") or "").strip()
            if not title:
                continue
            impact = _IMPACT_MAP.get(str(row.get("impact", "")).strip().lower(), Impact.UNKNOWN)
            events.append(
                CalendarEvent(
                    event_id=make_event_id(self.name, ccy, title, ts),
                    source=self.name,
                    currency=ccy,
                    name=title,
                    ts_utc=ts,
                    impact=impact,
                    forecast=parse_number(row.get("forecast")),
                    previous=parse_number(row.get("previous")),
                    actual=parse_number(row.get("actual")),
                    raw=row,
                )
            )
        log.info("forexfactory: parsed %d events", len(events))
        return events


def _parse_ts(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    # Feed emits ISO-8601 with offset, e.g. 2026-08-07T13:30:00-04:00
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
