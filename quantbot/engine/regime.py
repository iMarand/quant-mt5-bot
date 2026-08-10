"""Regime tagging (architecture §5).

A model can be genuinely accurate in one regime and useless in another; tagging
every prediction means the promotion gate can check stability *across* regimes
instead of averaging the failure away.
"""

from __future__ import annotations

import pandas as pd

from ..contracts import Regime


def classify_regime(row: pd.Series, news_window_min: int = 30) -> Regime:
    """Order matters: a news window overrides everything else."""
    to_next = float(row.get("minutes_to_next_high", 1440) or 1440)
    since_last = float(row.get("minutes_since_last_high", 1440) or 1440)
    if min(to_next, since_last) <= news_window_min:
        return Regime.NEWS_WINDOW

    atr_pct_rank = row.get("atr_percentile")
    base_atr_rank = _first_present(row, "atr_percentile")
    rank = atr_pct_rank if pd.notna(atr_pct_rank) else base_atr_rank
    if rank is not None and pd.notna(rank) and rank >= 0.85:
        return Regime.HIGH_VOL

    adx_val = _first_present(row, "adx")
    if adx_val is not None and pd.notna(adx_val) and adx_val >= 25:
        return Regime.TRENDING
    return Regime.RANGING


def classify_series(df: pd.DataFrame, news_window_min: int = 30) -> pd.Series:
    return df.apply(classify_regime, axis=1, news_window_min=news_window_min).map(lambda r: r.value)


def _first_present(row: pd.Series, suffix: str) -> float | None:
    """Feature names are timeframe-prefixed (e.g. `M15_adx`); find the first."""
    if suffix in row.index:
        return row[suffix]
    for name in row.index:
        if name.endswith("_" + suffix):
            return row[name]
    return None
