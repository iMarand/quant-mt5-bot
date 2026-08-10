"""Import a historical economic calendar from CSV.

Forex Factory's public feed only publishes the *current week* — `lastweek` and
`nextweek` return 404. That is fine for live trading (the journal accumulates
events as the bot runs) but it means the news setups have nothing to fire on
when backtesting over past months.

This closes that gap: point it at any historical calendar export with columns
for date, currency, event and ideally actual/forecast/previous. Column names are
matched loosely so most vendor exports work unchanged.
"""

from __future__ import annotations

import logging
from datetime import timezone
from pathlib import Path

import pandas as pd

from ..contracts import CalendarEvent, Impact
from .forexfactory import make_event_id, parse_number

log = logging.getLogger(__name__)

_COLUMN_ALIASES = {
    "date": "date",
    "datetime": "datetime",
    "timestamp": "datetime",
    "time": "time",
    "currency": "currency",
    "country": "currency",
    "ccy": "currency",
    "event": "name",
    "title": "name",
    "name": "name",
    "impact": "impact",
    "importance": "impact",
    "volatility": "impact",
    "actual": "actual",
    "forecast": "forecast",
    "consensus": "forecast",
    "estimate": "forecast",
    "previous": "previous",
    "prior": "previous",
}

_IMPACT_ALIASES = {
    "high": Impact.HIGH, "3": Impact.HIGH, "red": Impact.HIGH,
    "medium": Impact.MEDIUM, "moderate": Impact.MEDIUM, "2": Impact.MEDIUM, "orange": Impact.MEDIUM,
    "low": Impact.LOW, "1": Impact.LOW, "yellow": Impact.LOW,
    "holiday": Impact.HOLIDAY, "0": Impact.HOLIDAY,
}


def _norm_impact(value: object) -> Impact:
    """Normalize the many spellings vendors use for impact.

    Exact keys first, then substring matching, because real exports say things
    like "High Impact Expected" or "Non-Economic" rather than a bare "high".
    """
    text = str(value).strip().lower()
    if text in _IMPACT_ALIASES:
        return _IMPACT_ALIASES[text]
    if "non-economic" in text or "holiday" in text:
        return Impact.HOLIDAY
    for key in ("high", "medium", "moderate", "low"):
        if key in text:
            return _IMPACT_ALIASES[key]
    return Impact.UNKNOWN


def read_calendar_csv(path: str | Path, tz_offset_hours: float = 0.0) -> list[CalendarEvent]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns={c: _COLUMN_ALIASES.get(str(c).strip().lower(), c) for c in df.columns})

    if "datetime" in df.columns:
        stamp = df["datetime"].astype(str)
    elif "date" in df.columns and "time" in df.columns:
        stamp = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    elif "date" in df.columns:
        stamp = df["date"].astype(str)
    else:
        raise ValueError(f"{path.name}: no date/datetime column. Found: {list(df.columns)}")

    ts = pd.to_datetime(stamp.str.replace(".", "-", regex=False), errors="coerce", utc=True)
    for required in ("currency", "name"):
        if required not in df.columns:
            raise ValueError(f"{path.name}: missing a '{required}' column")

    events: list[CalendarEvent] = []
    for i, when in enumerate(ts):
        if pd.isna(when):
            continue
        when = when.to_pydatetime().astimezone(timezone.utc)
        if tz_offset_hours:
            when = when - pd.Timedelta(hours=tz_offset_hours).to_pytimedelta()
        ccy = str(df["currency"].iloc[i]).strip().upper()
        name = str(df["name"].iloc[i]).strip()
        if not ccy or not name or name.lower() == "nan":
            continue
        events.append(
            CalendarEvent(
                event_id=make_event_id("import", ccy, name, when),
                source="import",
                currency=ccy,
                name=name,
                ts_utc=when,
                impact=_norm_impact(df["impact"].iloc[i]) if "impact" in df.columns else Impact.UNKNOWN,
                forecast=parse_number(df["forecast"].iloc[i]) if "forecast" in df.columns else None,
                previous=parse_number(df["previous"].iloc[i]) if "previous" in df.columns else None,
                actual=parse_number(df["actual"].iloc[i]) if "actual" in df.columns else None,
            )
        )
    log.info("parsed %d calendar events from %s", len(events), path.name)
    return events


def import_calendar(db, path: str | Path, tz_offset_hours: float = 0.0) -> int:
    events = read_calendar_csv(path, tz_offset_hours)
    return db.upsert_events(events)


def calendar_coverage_end(db, max_gap_days: float = 14.0):
    """Last timestamp before the calendar goes sparse.

    The naive `max(ts_utc)` is wrong: the live weekly feed adds events for the
    *coming* week, so the maximum sits in the future even when the historical
    archive stopped a year ago. Training truncated on that would keep a huge
    news-blind middle section.

    This instead walks back to the end of the last densely-populated run — the
    point after which there is a gap longer than `max_gap_days`.
    """
    df = db.events_df()
    if df.empty:
        return None
    ts = df["ts_utc"].sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return ts.iloc[-1]
    gaps = ts.diff().dt.total_seconds() / 86400
    big = gaps[gaps > max_gap_days]
    if big.empty:
        return ts.iloc[-1]
    # The event immediately before the last oversized gap ends the dense run.
    return ts.iloc[int(big.index[-1]) - 1]


def calendar_coverage(db, start=None, end=None) -> dict:
    """How much of a period the calendar actually covers.

    Used to warn that a backtest's news setups had nothing to fire on, rather
    than letting "0 news trades" read as "news setups don't work".
    """
    df = db.events_df()
    if df.empty:
        return {"events": 0, "high_impact": 0, "covered": False}
    info = {
        "events": int(len(df)),
        "high_impact": int((df["impact"] == "high").sum()),
        "from": df["ts_utc"].min(),
        "to": df["ts_utc"].max(),
    }
    if start is not None and end is not None:
        span = pd.Timestamp(end) - pd.Timestamp(start)
        overlap_start = max(pd.Timestamp(start), df["ts_utc"].min())
        overlap_end = min(pd.Timestamp(end), df["ts_utc"].max())
        overlap = max(overlap_end - overlap_start, pd.Timedelta(0))
        info["coverage_pct"] = round(
            float(overlap / span * 100) if span.total_seconds() else 0.0, 1
        )
        info["covered"] = info["coverage_pct"] >= 50
    return info
