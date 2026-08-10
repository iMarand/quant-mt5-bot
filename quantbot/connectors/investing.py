"""Investing.com secondary/confirmatory calendar source (architecture §3.2).

**Deliberately inert by default.** Investing.com's Terms of Service prohibit
automated scraping of their site, and their robots.txt disallows the calendar
paths. Design principle §1.5 says compliance is architected in, not retrofitted,
so this connector refuses to run unless you supply credentials for a data feed
you are actually licensed to use (their official/partner API, or a vendor that
redistributes the same calendar).

The *cross-check* logic — the part that has real analytical value — lives in
`cross_check()` below and works against any second source, so nothing downstream
depends on Investing.com specifically.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta

from ..contracts import CalendarEvent, Impact
from .forexfactory import make_event_id, parse_number
from .policy import ComplianceError, FetchPolicy

log = logging.getLogger(__name__)


class InvestingCalendar:
    name = "investing"

    def __init__(
        self,
        policy: FetchPolicy,
        api_key: str | None = None,
        api_url: str | None = None,
        currencies: list[str] | None = None,
        enabled: bool = False,
    ) -> None:
        self.policy = policy
        self.api_key = api_key
        self.api_url = api_url
        self.currencies = [c.upper() for c in (currencies or [])]
        self.enabled = enabled

    def fetch_events(self, force: bool = False) -> list[CalendarEvent]:
        if not self.enabled:
            log.info("investing connector disabled (calendar.investing_enabled=false)")
            return []
        if not (self.api_key and self.api_url):
            raise ComplianceError(
                "Investing.com scraping is disallowed by their ToS/robots.txt. Set "
                "calendar.investing_api_key + an api_url for a feed you are licensed to "
                "use, or leave calendar.investing_enabled=false."
            )
        url = f"{self.api_url}?apikey={self.api_key}"
        payload = json.loads(self.policy.fetch(url, force=force))
        return self._parse(payload)

    def _parse(self, payload: list[dict]) -> list[CalendarEvent]:
        out: list[CalendarEvent] = []
        for row in payload:
            ccy = str(row.get("currency", "")).upper()
            if self.currencies and ccy not in self.currencies:
                continue
            ts = row.get("timestamp_utc")
            if not ts:
                continue
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
            name = str(row.get("event", "")).strip()
            out.append(
                CalendarEvent(
                    event_id=make_event_id(self.name, ccy, name, dt),
                    source=self.name,
                    currency=ccy,
                    name=name,
                    ts_utc=dt,
                    impact=Impact(str(row.get("impact", "unknown")).lower())
                    if str(row.get("impact", "")).lower() in Impact._value2member_map_
                    else Impact.UNKNOWN,
                    forecast=parse_number(row.get("forecast")),
                    previous=parse_number(row.get("previous")),
                    actual=parse_number(row.get("actual")),
                    raw=row,
                )
            )
        return out


def cross_check(
    primary: list[CalendarEvent],
    secondary: list[CalendarEvent],
    time_tolerance_min: int = 30,
) -> list[dict]:
    """Report disagreements between two calendar sources (§3.2).

    Sources routinely disagree on impact rating and occasionally on the exact
    release time; those disagreements are themselves a useful uncertainty signal
    and are surfaced as alerts rather than silently reconciled.
    """
    issues: list[dict] = []
    by_key: dict[tuple[str, str], list[CalendarEvent]] = {}
    for ev in secondary:
        by_key.setdefault((ev.currency, _norm(ev.name)), []).append(ev)

    for ev in primary:
        candidates = by_key.get((ev.currency, _norm(ev.name)), [])
        match = None
        for cand in candidates:
            if abs(cand.ts_utc - ev.ts_utc) <= timedelta(minutes=time_tolerance_min):
                match = cand
                break
        if match is None:
            if ev.impact is Impact.HIGH:
                issues.append(
                    {"kind": "missing_in_secondary", "event": ev.name, "currency": ev.currency}
                )
            continue
        if match.impact != ev.impact:
            issues.append(
                {
                    "kind": "impact_mismatch",
                    "event": ev.name,
                    "currency": ev.currency,
                    "primary": ev.impact.value,
                    "secondary": match.impact.value,
                }
            )
        delta_min = abs((match.ts_utc - ev.ts_utc).total_seconds()) / 60
        if delta_min > 1:
            issues.append(
                {
                    "kind": "time_mismatch",
                    "event": ev.name,
                    "currency": ev.currency,
                    "delta_minutes": round(delta_min, 1),
                }
            )
    return issues


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())
