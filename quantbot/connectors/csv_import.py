"""Import OHLCV history from CSV — the fallback when the MT5 bridge is blocked.

MetaTrader exports bars from the chart context menu ("Save As...") and from
Tools > History Center. Those files use a tab-separated `<DATE> <TIME> <OPEN>...`
header, which this reads directly; a generic `time,open,high,low,close,volume`
CSV works too.

Timestamps in MT5 exports are **broker server time**, not UTC. Pass the server's
UTC offset with `--tz-offset` or every event-timing feature will be wrong by
those hours.
"""

from __future__ import annotations

import logging
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

from ..contracts import Candle, TIMEFRAMES

log = logging.getLogger(__name__)

#: MT5 export header -> our column name.
_MT5_COLUMNS = {
    "<DATE>": "date",
    "<TIME>": "time",
    "<OPEN>": "open",
    "<HIGH>": "high",
    "<LOW>": "low",
    "<CLOSE>": "close",
    "<TICKVOL>": "volume",
    "<VOL>": "real_volume",
    "<SPREAD>": "spread",
}

_ALIASES = {
    "date": "date",
    "time": "time",
    "datetime": "time",
    "timestamp": "time",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "tickvol": "volume",
    "tick_volume": "volume",
    "vol": "volume",
    "spread": "spread",
}


def read_ohlcv_csv(path: str | Path, tz_offset_hours: float = 0.0) -> pd.DataFrame:
    """Parse an MT5 or generic OHLCV CSV into a UTC-indexed frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # MT5 uses tabs; plenty of other exports use commas or semicolons.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    if any(c in _MT5_COLUMNS for c in df.columns):
        df = df.rename(columns=_MT5_COLUMNS)
    else:
        df = df.rename(
            columns={c: _ALIASES.get(c.strip().lower().lstrip("<").rstrip(">"), c) for c in df.columns}
        )

    if "date" in df.columns and "time" in df.columns:
        stamp = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    elif "time" in df.columns:
        stamp = df["time"].astype(str)
    elif "date" in df.columns:
        stamp = df["date"].astype(str)
    else:
        raise ValueError(
            f"{path.name}: no date/time column found. Columns present: {list(df.columns)}"
        )

    ts = pd.to_datetime(stamp.str.replace(".", "-", regex=False), errors="coerce", format="mixed")
    if ts.isna().all():
        raise ValueError(f"{path.name}: could not parse any timestamps")

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")

    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0.0),
            "spread": pd.to_numeric(df.get("spread", 0), errors="coerce").fillna(0.0),
        }
    )
    out.index = pd.DatetimeIndex(ts)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.isna()]

    # Server time -> UTC, then drop the tz so it matches everything else.
    if tz_offset_hours:
        out.index = out.index - timedelta(hours=tz_offset_hours)
    out.index = out.index.tz_localize(timezone.utc)
    return out.sort_index()[~out.sort_index().index.duplicated(keep="last")]


def infer_timeframe(index: pd.DatetimeIndex) -> str | None:
    """Guess the timeframe from the modal bar spacing."""
    if len(index) < 3:
        return None
    minutes = pd.Series(index).diff().dt.total_seconds().div(60).mode()
    if minutes.empty:
        return None
    spacing = int(round(float(minutes.iloc[0])))
    for name, mins in TIMEFRAMES.items():
        if mins == spacing:
            return name
    return None


def import_csv(
    db,
    path: str | Path,
    symbol: str,
    timeframe: str | None = None,
    tz_offset_hours: float = 0.0,
) -> tuple[int, str]:
    """Read a CSV and upsert it as candles. Returns (rows, timeframe used)."""
    df = read_ohlcv_csv(path, tz_offset_hours)
    tf = timeframe or infer_timeframe(df.index)
    if tf is None:
        raise ValueError(
            f"could not infer the timeframe of {Path(path).name}; pass it explicitly"
        )
    if tf not in TIMEFRAMES:
        raise ValueError(f"unknown timeframe {tf!r}; expected one of {list(TIMEFRAMES)}")

    candles = [
        Candle(
            symbol=symbol,
            timeframe=tf,
            ts=ts.to_pydatetime(),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume),
            spread=float(r.spread),
        )
        for ts, r in df.iterrows()
    ]
    n = db.upsert_candles(candles)
    log.info("imported %d %s %s bars from %s", n, symbol, tf, Path(path).name)
    return n, tf
