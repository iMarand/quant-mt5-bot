"""Multi-timeframe aligner — the core of "analyzing from different angles" (§4).

Joins every timeframe onto the base timeframe's clock with a strict backward
as-of join, so a bar at 10:15 sees the H1 bar that *closed at or before* 10:15
and never the one still forming. Getting this wrong is the single most common
source of backtests that look brilliant and trade terribly.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..contracts import tf_minutes
from .events import EVENT_FEATURE_VERSION, build_event_features
from .indicators import INDICATOR_VERSION, compute_indicators

log = logging.getLogger(__name__)

FEATURE_VERSION = f"{INDICATOR_VERSION}+{EVENT_FEATURE_VERSION}+mtf-v1"

#: Columns worth carrying up from a higher timeframe. Keeping this list short
#: matters: 6 timeframes x 40 indicators is a fast route to overfitting.
HTF_COLUMNS = [
    "ret_5",
    "atr_pct",
    "atr_percentile",
    "ema_10_dist",
    "ema_50_dist",
    "ema_fast_slow",
    "rsi_14",
    "macd_hist_norm",
    "adx",
    "plus_di",
    "minus_di",
    "bb_pct",
    "bb_width",
    "stoch_k",
    "range_position",
    "dist_to_high",
    "dist_to_low",
    "pat_bullish_engulfing",
    "pat_bearish_engulfing",
    "pat_pin_bull",
    "pat_pin_bear",
]


def align_timeframes(
    frames: dict[str, pd.DataFrame],
    base_timeframe: str,
    atr_period: int = 14,
) -> pd.DataFrame:
    """`frames` maps timeframe -> raw OHLCV. Returns one feature frame on base tf."""
    if base_timeframe not in frames or frames[base_timeframe].empty:
        raise ValueError(f"no candles for base timeframe {base_timeframe}")

    base_raw = frames[base_timeframe]
    base = compute_indicators(base_raw, atr_period=atr_period)
    base = base.add_prefix(f"{base_timeframe}_")
    base[f"{base_timeframe}_close"] = base_raw["close"]
    out = base

    base_minutes = tf_minutes(base_timeframe)
    for tf, raw in frames.items():
        if tf == base_timeframe or raw.empty:
            continue
        if tf_minutes(tf) < base_minutes:
            # A lower timeframe would need aggregation, not as-of joining; the
            # base timeframe is intentionally the finest resolution we model on.
            log.debug("skipping %s: finer than base %s", tf, base_timeframe)
            continue
        ind = compute_indicators(raw, atr_period=atr_period)
        cols = [c for c in HTF_COLUMNS if c in ind.columns]
        htf = ind[cols].add_prefix(f"{tf}_")
        # Shift by one bar: the higher-tf bar stamped 10:00 only *closes* at
        # 11:00, so at 10:15 the newest fully-known H1 bar is the 09:00 one.
        htf = htf.shift(1)
        out = pd.merge_asof(
            out.sort_index(),
            htf.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(minutes=tf_minutes(tf) * 3),
        )

    return out


def add_cross_asset(
    base: pd.DataFrame, correlated: dict[str, pd.DataFrame], lookback: int = 60
) -> pd.DataFrame:
    """Correlated-instrument context: DXY, yields, gold, oil (addendum §A)."""
    out = base
    for sym, raw in correlated.items():
        if raw.empty:
            continue
        ret = raw["close"].pct_change()
        frame = pd.DataFrame(
            {
                f"x_{sym}_ret_1": ret,
                f"x_{sym}_ret_5": raw["close"].pct_change(5),
                f"x_{sym}_corr": ret.rolling(lookback, min_periods=lookback // 2).corr(
                    base.iloc[:, 0].pct_change()
                )
                if len(base.columns)
                else 0.0,
            }
        ).shift(1)
        out = pd.merge_asof(
            out.sort_index(),
            frame.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
    return out


def build_feature_frame(
    frames: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    symbol: str,
    base_timeframe: str,
    atr_period: int = 14,
    news_window_min: int = 30,
    correlated: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Full Layer-2 output: MTF indicators + event features, one row per base bar."""
    mtf = align_timeframes(frames, base_timeframe, atr_period=atr_period)
    evt = build_event_features(
        mtf.index,
        events,
        symbol=symbol,
        candles=frames.get(base_timeframe),
        news_window_min=news_window_min,
    )
    out = mtf.join(evt, how="left")
    if correlated:
        out = add_cross_asset(out, correlated)
    out["symbol"] = symbol
    return out


#: Columns produced by the labeler. They describe what happened *after* the bar,
#: so training on them is direct leakage — `bars_held` is literally how long it
#: took the future to resolve.
LABEL_COLUMNS = {"label", "fwd_return", "bars_held", "barrier_width"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric model inputs only — drops raw price levels, metadata and labels."""
    drop = {"symbol", "close"} | LABEL_COLUMNS
    cols = []
    for c in df.columns:
        if c in drop or c.endswith("_close") or c.endswith("_atr"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols
