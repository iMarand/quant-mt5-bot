"""Pre-trade gate: the ordered checks that run *before* any setup is evaluated.

    1. calendar   — do we have usable calendar data, and is it current?
    2. clock      — is the market open, is the data fresh, is it a sane time?
    3. news       — is a high-impact release imminent, or just released?
    4. session    — is this pair's own money centre awake?
    -> only then are strategies allowed to look for a setup.

Running these first is deliberate and cheap: there is no point computing
indicators and model probabilities for a bar we would never trade. Each check
returns a reason, so a quiet bot can always explain its silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from ..contracts import as_utc, tf_minutes, utcnow
from ..features.events import currencies_of
from .sessions import SessionPolicy, active_sessions

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    allowed: bool
    stage: str = ""
    reason: str = ""
    #: True when a high-impact release just landed — news setups want this.
    news_active: bool = False
    #: Sessions live at decision time — journaled so performance can be
    #: broken down by session later.
    sessions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        verdict = "TRADE" if self.allowed else "STAND DOWN"
        return f"{verdict} [{self.stage}] {self.reason}"


class PreTradeGate:
    def __init__(self, cfg, session_policy: SessionPolicy | None = None) -> None:
        self.cfg = cfg
        self.session_policy = session_policy or SessionPolicy()

    def check(
        self,
        symbol: str,
        now: datetime | None = None,
        events: pd.DataFrame | None = None,
        last_bar_ts: datetime | None = None,
        base_timeframe: str = "M15",
    ) -> GateResult:
        now = as_utc(now or utcnow())
        notes: list[str] = []
        currencies = currencies_of(symbol)

        # -- 1. calendar -------------------------------------------------
        result = self._check_calendar(events, now, currencies, notes)
        if result is not None:
            return result

        # -- 2. clock / data freshness -----------------------------------
        result = self._check_clock(now, last_bar_ts, base_timeframe, notes)
        if result is not None:
            return result

        # -- 3. news -----------------------------------------------------
        news = self._check_news(events, now, currencies, notes)
        if news is not None and not news.allowed:
            return news
        news_active = bool(news and news.news_active)

        # -- 4. session --------------------------------------------------
        ok, reason = self.session_policy.evaluate(now, currencies)
        if not ok:
            # A high-impact release moves a pair regardless of whose session it
            # is, so an active news window overrides the session restriction.
            if news_active and self.cfg.news_overrides_session:
                notes.append(f"session '{reason}' overridden by active news window")
            else:
                return GateResult(False, "session", reason, notes=notes)
        else:
            notes.append(f"session: {reason}")

        return GateResult(
            True,
            "clear",
            "all pre-trade checks passed",
            news_active=news_active,
            sessions=active_sessions(now),
            notes=notes,
        )

    # -- individual stages -------------------------------------------------
    def _check_calendar(self, events, now, currencies, notes) -> GateResult | None:
        if not self.cfg.require_calendar:
            return None
        if events is None or events.empty:
            return GateResult(
                False, "calendar", "no calendar data — run `ingest` (or set "
                "pretrade.require_calendar: false to trade without it)"
            )
        relevant = events[events["currency"].isin(currencies)]
        if relevant.empty:
            notes.append(f"calendar holds nothing for {'/'.join(currencies)}")
            if self.cfg.require_relevant_calendar:
                return GateResult(
                    False, "calendar", f"no events for {'/'.join(currencies)}"
                )
            return None

        newest = relevant["ts_utc"].max()
        age_days = (now - newest).total_seconds() / 86400
        # A calendar that stopped updating is worse than none: the news veto
        # would silently pass and we would trade blind into a release.
        if age_days > self.cfg.max_calendar_age_days:
            return GateResult(
                False,
                "calendar",
                f"calendar is stale — newest event is {age_days:.1f} days old",
            )
        notes.append(f"calendar current to {newest:%Y-%m-%d %H:%M} UTC")
        return None

    def _check_clock(self, now, last_bar_ts, base_timeframe, notes) -> GateResult | None:
        if now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21):
            return GateResult(False, "clock", "forex market is closed for the weekend")

        if last_bar_ts is not None:
            age_min = (now - as_utc(last_bar_ts)).total_seconds() / 60
            tolerance = tf_minutes(base_timeframe) * self.cfg.max_bar_age_multiple
            if age_min > tolerance:
                return GateResult(
                    False,
                    "clock",
                    f"last {base_timeframe} bar is {age_min:.0f} min old "
                    f"(tolerance {tolerance:.0f}) — data feed may be stale",
                )
            notes.append(f"data fresh ({age_min:.0f} min old)")

        if self.cfg.blackout_dates:
            today = now.date().isoformat()
            if today in self.cfg.blackout_dates:
                return GateResult(False, "clock", f"{today} is a configured blackout date")

        if now.month == 12 and now.day >= self.cfg.year_end_blackout_from_day:
            return GateResult(
                False, "clock", "year-end: liquidity is thin and spreads are unreliable"
            )
        return None

    def _check_news(self, events, now, currencies, notes) -> GateResult | None:
        if events is None or events.empty:
            return None
        relevant = events[
            (events["currency"].isin(currencies)) & (events["impact"] == "high")
        ]
        if relevant.empty:
            return None

        before = timedelta(minutes=self.cfg.block_before_news_min)
        after = timedelta(minutes=self.cfg.news_window_after_min)

        imminent = relevant[
            (relevant["ts_utc"] > now) & (relevant["ts_utc"] <= now + before)
        ]
        if not imminent.empty:
            nxt = imminent.iloc[0]
            mins = (nxt["ts_utc"] - now).total_seconds() / 60
            return GateResult(
                False,
                "news",
                f"{nxt['name']} ({nxt['currency']}) in {mins:.0f} min — "
                "no new positions into a high-impact release",
            )

        just_released = relevant[
            (relevant["ts_utc"] <= now) & (relevant["ts_utc"] >= now - after)
        ]
        if not just_released.empty:
            last = just_released.iloc[-1]
            mins = (now - last["ts_utc"]).total_seconds() / 60
            notes.append(f"news window: {last['name']} released {mins:.0f} min ago")
            return GateResult(True, "news", "post-release window", news_active=True)

        upcoming = relevant[relevant["ts_utc"] > now]
        if not upcoming.empty:
            nxt = upcoming.iloc[0]
            mins = (nxt["ts_utc"] - now).total_seconds() / 60
            notes.append(f"next high-impact: {nxt['name']} in {mins:.0f} min")
        return None
