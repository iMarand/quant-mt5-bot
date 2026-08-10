"""Feature-layer invariants. The lookahead tests are the important ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.features.events import build_event_features, currencies_of, normalized_surprise
from quantbot.features.indicators import atr, bollinger, compute_indicators, rsi
from quantbot.features.mtf import align_timeframes


def make_ohlcv(n=500, freq="15min", seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 1.1 + np.cumsum(rng.normal(0, 0.0004, n))
    high = close + np.abs(rng.normal(0, 0.0003, n))
    low = close - np.abs(rng.normal(0, 0.0003, n))
    return pd.DataFrame(
        {
            "open": np.r_[close[0], close[:-1]],
            "high": np.maximum(high, close),
            "low": np.minimum(low, close),
            "close": close,
            "volume": rng.integers(100, 2000, n).astype(float),
        },
        index=idx,
    )


def test_rsi_is_bounded():
    df = make_ohlcv()
    r = rsi(df["close"]).dropna()
    assert not r.empty
    assert r.between(0, 100).all()


def test_rsi_flat_series_is_neutral():
    flat = pd.Series([1.0] * 60, index=pd.date_range("2026-01-01", periods=60, freq="15min"))
    assert rsi(flat).dropna().eq(50.0).all()


def test_atr_is_positive():
    a = atr(make_ohlcv()).dropna()
    assert (a > 0).all()


def test_bollinger_bands_are_ordered():
    bb = bollinger(make_ohlcv()["close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all()
    assert (bb["bb_mid"] >= bb["bb_lower"]).all()


def test_indicators_use_no_future_data():
    """Truncating the series must not change earlier indicator values.

    This is the test that catches accidental lookahead — anything computed with
    a centered window or a full-series statistic fails it.
    """
    df = make_ohlcv(400)
    full = compute_indicators(df)
    partial = compute_indicators(df.iloc[:300])
    common = [c for c in full.columns if c in partial.columns]
    a = full[common].iloc[:300]
    b = partial[common]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_mtf_alignment_never_uses_an_unclosed_higher_tf_bar():
    m15 = make_ohlcv(400, "15min", seed=1)
    h1 = make_ohlcv(100, "1h", seed=2)
    out = align_timeframes({"M15": m15, "H1": h1}, "M15")
    assert "H1_rsi_14" in out.columns

    h1_ind = compute_indicators(h1)
    for ts in out.index[50:200:17]:
        value = out.loc[ts, "H1_rsi_14"]
        if pd.isna(value):
            continue
        # The value must come from an H1 bar that had already *closed* by ts:
        # bar open + 1h <= ts.
        closed = h1_ind[h1_ind.index + pd.Timedelta(hours=1) <= ts]
        assert not closed.empty
        assert value == pytest.approx(float(closed["rsi_14"].iloc[-1]), abs=1e-9)


def test_currencies_of():
    assert currencies_of("EURUSD") == ["EUR", "USD"]
    assert currencies_of("XAUUSD") == ["XAU", "USD"]
    assert currencies_of("US30") == ["USD"]


def test_event_features_clock_counts_down_to_release():
    idx = pd.date_range("2026-01-01 10:00", periods=8, freq="15min", tz="UTC")
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "currency": ["USD"],
            "name": ["CPI m/m"],
            "ts_utc": [pd.Timestamp("2026-01-01 11:30", tz="UTC")],
            "impact": ["high"],
            "forecast": [0.2],
            "previous": [0.1],
            "actual": [None],
            "surprise": [None],
        }
    )
    out = build_event_features(idx, events, "EURUSD", news_window_min=30)
    before = out.loc[out.index < pd.Timestamp("2026-01-01 11:30", tz="UTC")]
    counts = before["minutes_to_next_high"].tolist()
    assert counts == sorted(counts, reverse=True), "clock must count down"
    assert counts[0] == 90.0
    # Once the release has passed and nothing else is scheduled, the "next
    # event" clock resets to its far-away default.
    assert out["minutes_to_next_high"].iloc[-1] == 1440.0
    assert out["minutes_since_last_high"].iloc[-1] < 1440.0
    # In-window flag turns on within 30 minutes of the release.
    assert out.loc[pd.Timestamp("2026-01-01 11:15", tz="UTC"), "in_news_window"] == 1.0
    assert out.loc[pd.Timestamp("2026-01-01 10:00", tz="UTC"), "in_news_window"] == 0.0


def test_normalized_surprise_excludes_the_current_observation():
    ev = pd.DataFrame(
        {"name": ["CPI"] * 5, "surprise": [1.0, 1.0, 1.0, 1.0, 10.0]}
    )
    z = normalized_surprise(ev)
    # First row has no history to normalize against -> 0, not NaN.
    assert z.iloc[0] == 0.0
    # The outlier is scored against the *prior* rows, so it must stand out.
    assert z.iloc[-1] > 1.0
