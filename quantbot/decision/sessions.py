"""Trading sessions — when a given pair is actually worth trading.

A EURUSD signal at 03:00 UTC and the same signal at 13:00 UTC are not the same
trade: the first is in the Tokyo lull with wide spreads and thin liquidity, the
second is in the London/New York overlap. Sessions are how the bot expresses
"trade this pair when its own money centre is awake".

All times are UTC. Session boundaries in UTC shift by an hour when London and
New York observe daylight saving; `dst_shift_hours` applies that adjustment for
the northern-hemisphere summer rather than pretending it doesn't happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from ..contracts import as_utc


@dataclass(frozen=True)
class Session:
    name: str
    start_hour: float  # UTC, winter
    end_hour: float
    #: Currencies whose home market this session is.
    currencies: frozenset[str]
    #: Whether this session follows northern-hemisphere DST.
    observes_dst: bool = True

    def window(self, when: datetime) -> tuple[float, float]:
        shift = -1.0 if (self.observes_dst and _is_northern_dst(when.date())) else 0.0
        return (self.start_hour + shift) % 24, (self.end_hour + shift) % 24

    def contains(self, when: datetime) -> bool:
        when = as_utc(when)
        hour = when.hour + when.minute / 60.0
        start, end = self.window(when)
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # wraps midnight (Sydney/Tokyo)


#: Winter (standard time) UTC windows.
SESSIONS: dict[str, Session] = {
    "sydney": Session("sydney", 21.0, 6.0, frozenset({"AUD", "NZD"}), observes_dst=False),
    "tokyo": Session("tokyo", 0.0, 9.0, frozenset({"JPY", "CNH", "SGD"}), observes_dst=False),
    "london": Session("london", 8.0, 17.0, frozenset({"GBP", "EUR", "CHF"})),
    "newyork": Session("newyork", 13.0, 22.0, frozenset({"USD", "CAD", "MXN"})),
    # The overlap is where FX volume actually concentrates.
    "london_ny_overlap": Session(
        "london_ny_overlap", 13.0, 17.0, frozenset({"EUR", "GBP", "USD", "CHF", "CAD"})
    ),
}


def _is_northern_dst(day: date) -> bool:
    """Rough EU/US daylight-saving window: late March to late October.

    The EU and US switch on different weekends, so a handful of days each year
    are off by an hour. That is a rounding error against a session boundary, not
    something worth a timezone database dependency.
    """
    return 3 < day.month < 11 or (day.month == 3 and day.day >= 25) or (
        day.month == 11 and day.day < 1
    )


def sessions_for_currencies(currencies: list[str]) -> list[str]:
    """Which sessions matter for a pair, e.g. EURUSD -> london, newyork, overlap."""
    wanted = {c.upper() for c in currencies}
    return [
        name
        for name, session in SESSIONS.items()
        if session.currencies & wanted
    ]


def active_sessions(when: datetime) -> list[str]:
    return [name for name, s in SESSIONS.items() if s.contains(when)]


@dataclass
class SessionPolicy:
    """Decides whether a pair may be traded at a given instant."""

    #: Session names this mode is allowed to trade in. Empty = derive from the
    #: pair's own currencies.
    allowed: list[str] = field(default_factory=list)
    #: Block the illiquid hours around the daily rollover regardless.
    avoid_rollover: bool = True
    rollover_hour: float = 21.0
    rollover_buffer_min: float = 30.0
    #: Refuse to open new trades late on Friday (weekend gap risk).
    friday_cutoff_hour: float | None = 19.0

    def evaluate(self, when: datetime, currencies: list[str]) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        when = as_utc(when)
        hour = when.hour + when.minute / 60.0

        # FX is shut from Friday close to Sunday open.
        if when.weekday() == 5:
            return False, "saturday: market closed"
        if when.weekday() == 6 and hour < 21.0:
            return False, "sunday before the Sydney open"
        if when.weekday() == 4 and self.friday_cutoff_hour is not None:
            if hour >= self.friday_cutoff_hour:
                return False, f"friday after {self.friday_cutoff_hour:.0f}:00 UTC (weekend gap risk)"

        if self.avoid_rollover:
            delta = abs(hour - self.rollover_hour) * 60
            if min(delta, 1440 - delta) <= self.rollover_buffer_min:
                return False, "daily rollover: spreads widen, liquidity thins"

        names = self.allowed or sessions_for_currencies(currencies)
        if not names:
            return True, "no session restriction for this instrument"
        live = [n for n in names if n in SESSIONS and SESSIONS[n].contains(when)]
        if not live:
            return False, f"outside {'/'.join(names)} session(s)"
        return True, f"in {'/'.join(live)}"
