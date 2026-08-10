"""Calendar rows -> model-ready event features (architecture §4).

The headline feature the whole design is built around: at any bar we want to
know *how far we are from the next high-impact release for the currencies in
this pair*, how big the last surprise was, and how much this event type has
historically moved this instrument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import Impact

EVENT_FEATURE_VERSION = "evt-v1"

_IMPACT_WEIGHT = {i.value: i.weight for i in Impact}


def currencies_of(symbol: str) -> list[str]:
    """EURUSD -> [EUR, USD]; XAUUSD -> [XAU, USD]. Falls back to USD."""
    s = "".join(ch for ch in symbol.upper() if ch.isalpha())
    if len(s) >= 6:
        return [s[:3], s[3:6]]
    return ["USD"]


def _dt64(values) -> np.ndarray:
    """Tz-aware timestamps -> naive UTC datetime64[ns].

    Pandas hands back an *object* array of Timestamps for tz-aware data, and
    object arrays don't support timedelta arithmetic. Everything here is already
    UTC, so dropping the tz is lossless and makes the vectorized math work.
    """
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy(dtype="datetime64[ns]")


def _impact_rank(impact: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(impact, 0)


def normalized_surprise(events: pd.DataFrame) -> pd.Series:
    """Surprise in units of that event's own historical surprise dispersion.

    Raw `actual - forecast` is not comparable across events (NFP is in hundreds
    of thousands, CPI in tenths of a percent), so each event name is z-scaled
    against its own history — expanding, so no future information leaks in.
    """
    if events.empty:
        return pd.Series(dtype=float)
    # Unreleased events carry a None surprise, which makes the column object
    # dtype and breaks the arithmetic below; coerce before grouping.
    work = events.assign(_surprise=pd.to_numeric(events["surprise"], errors="coerce"))
    surprise = work["_surprise"]
    grp = work.groupby("name")["_surprise"]
    std = grp.transform(lambda s: s.expanding().std().shift(1))
    mean = grp.transform(lambda s: s.expanding().mean().shift(1))
    # An event whose past surprises were all identical has zero dispersion, and
    # dividing by it would silently zero out a genuine outlier. Fall back to the
    # mean absolute surprise so "first real deviation" still registers as large.
    mad = grp.transform(lambda s: s.abs().expanding().mean().shift(1))
    scale = std.where(std > 0, mad)
    scale = scale.where(scale > 0, np.nan)
    z = (surprise - mean.fillna(0.0)) / scale
    return z.clip(-5, 5).fillna(0.0)


def historical_reaction(
    events: pd.DataFrame, candles: pd.DataFrame, window_min: int = 30
) -> pd.Series:
    """Mean |price move| in the `window_min` after each past instance of an event.

    Expanding mean, shifted by one, so an event never sees its own reaction.
    """
    if events.empty or candles.empty:
        return pd.Series(0.0, index=events.index)
    close = candles["close"]
    moves = []
    for ts in events["ts_utc"]:
        try:
            before_idx = close.index.searchsorted(ts) - 1
            after_idx = close.index.searchsorted(ts + pd.Timedelta(minutes=window_min))
            if before_idx < 0 or after_idx >= len(close):
                moves.append(np.nan)
                continue
            p0, p1 = close.iloc[before_idx], close.iloc[after_idx]
            moves.append(abs(p1 - p0) / p0 if p0 else np.nan)
        except (IndexError, KeyError):
            moves.append(np.nan)
    out = pd.Series(moves, index=events.index, dtype=float)
    return (
        out.groupby(events["name"]).transform(lambda s: s.expanding().mean().shift(1)).fillna(0.0)
    )


def build_event_features(
    bar_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    symbol: str,
    candles: pd.DataFrame | None = None,
    news_window_min: int = 30,
) -> pd.DataFrame:
    """Per-bar event feature frame, aligned to `bar_index`.

    Columns:
      minutes_to_next_high / minutes_since_last_high  (clipped, signed clock)
      in_news_window        : |minutes| <= news_window_min around a high-impact event
      next_event_impact_w   : impact weight of the next scheduled event
      last_surprise_z       : normalized surprise of the most recent release
      last_surprise_signed  : that surprise oriented for the *base* currency
      event_density_24h     : impact-weighted count of events in the last 24h
      expected_reaction     : historical |move| for the next event type
    """
    idx = pd.DatetimeIndex(bar_index)
    cols = [
        "minutes_to_next_high",
        "minutes_since_last_high",
        "in_news_window",
        "next_event_impact_w",
        "last_surprise_z",
        "last_surprise_signed",
        "event_density_24h",
        "expected_reaction",
    ]
    out = pd.DataFrame(0.0, index=idx, columns=cols)
    out["minutes_to_next_high"] = 1440.0
    out["minutes_since_last_high"] = 1440.0
    if events is None or events.empty:
        return out

    base_ccy, quote_ccy = (currencies_of(symbol) + ["USD"])[:2]
    ev = events[events["currency"].isin([base_ccy, quote_ccy])].copy()
    if ev.empty:
        return out
    ev = ev.sort_values("ts_utc").reset_index(drop=True)
    ev["impact_w"] = ev["impact"].map(_IMPACT_WEIGHT).fillna(0.1)
    ev["rank"] = ev["impact"].map(_impact_rank)
    ev["surprise_z"] = normalized_surprise(ev)
    # A positive surprise for the quote currency pushes the pair down.
    ev["surprise_signed"] = ev["surprise_z"] * np.where(ev["currency"] == base_ccy, 1.0, -1.0)
    ev["expected_reaction"] = (
        historical_reaction(ev, candles, news_window_min)
        if candles is not None and not candles.empty
        else 0.0
    )

    high = ev[ev["rank"] >= 3].reset_index(drop=True)
    ts_all = _dt64(ev["ts_utc"])
    ts_high = _dt64(high["ts_utc"]) if not high.empty else np.array([], dtype="datetime64[ns]")
    bars = _dt64(idx)

    # -- distance to the next / previous high-impact release ---------------
    if len(ts_high):
        nxt = np.searchsorted(ts_high, bars, side="left")
        prv = nxt - 1
        to_next = np.where(
            nxt < len(ts_high),
            (ts_high[np.clip(nxt, 0, len(ts_high) - 1)] - bars) / np.timedelta64(1, "m"),
            1440.0,
        )
        since_prev = np.where(
            prv >= 0,
            (bars - ts_high[np.clip(prv, 0, len(ts_high) - 1)]) / np.timedelta64(1, "m"),
            1440.0,
        )
        out["minutes_to_next_high"] = np.clip(to_next.astype(float), 0, 1440)
        out["minutes_since_last_high"] = np.clip(since_prev.astype(float), 0, 1440)
        out["in_news_window"] = (
            (out["minutes_to_next_high"] <= news_window_min)
            | (out["minutes_since_last_high"] <= news_window_min)
        ).astype(float)

    # -- next scheduled event of any impact --------------------------------
    nxt_any = np.searchsorted(ts_all, bars, side="left")
    valid = nxt_any < len(ts_all)
    safe = np.clip(nxt_any, 0, len(ts_all) - 1)
    out["next_event_impact_w"] = np.where(valid, ev["impact_w"].to_numpy()[safe], 0.0)
    out["expected_reaction"] = np.where(valid, ev["expected_reaction"].to_numpy()[safe], 0.0)

    # -- most recent *released* event --------------------------------------
    released = ev[ev["actual"].notna()].reset_index(drop=True)
    if not released.empty:
        ts_rel = _dt64(released["ts_utc"])
        prev_rel = np.searchsorted(ts_rel, bars, side="right") - 1
        ok = prev_rel >= 0
        safe_rel = np.clip(prev_rel, 0, len(ts_rel) - 1)
        # Decay the surprise: a release 6 hours ago is not news any more.
        age_min = np.where(
            ok, (bars - ts_rel[safe_rel]) / np.timedelta64(1, "m"), 1e9
        ).astype(float)
        decay = np.exp(-age_min / 120.0)
        out["last_surprise_z"] = np.where(ok, released["surprise_z"].to_numpy()[safe_rel], 0.0) * decay
        out["last_surprise_signed"] = (
            np.where(ok, released["surprise_signed"].to_numpy()[safe_rel], 0.0) * decay
        )

    # -- 24h impact-weighted event density ---------------------------------
    left = np.searchsorted(ts_all, bars - np.timedelta64(24, "h"), side="left")
    right = np.searchsorted(ts_all, bars, side="right")
    cum_w = np.concatenate([[0.0], np.cumsum(ev["impact_w"].to_numpy())])
    out["event_density_24h"] = cum_w[right] - cum_w[left]

    return out.fillna(0.0)


def next_high_impact_events(events: pd.DataFrame, now: pd.Timestamp, within_min: int = 60):
    """Used by the scheduler for event-triggered runs (§8.1)."""
    if events.empty:
        return events
    upcoming = events[
        (events["ts_utc"] >= now)
        & (events["ts_utc"] <= now + pd.Timedelta(minutes=within_min))
        & (events["impact"] == "high")
    ]
    return upcoming.sort_values("ts_utc")
