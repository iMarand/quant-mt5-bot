"""Technical indicators — pure pandas/numpy, no ta-lib build step (§4).

Every function takes an OHLCV DataFrame indexed by UTC bar-open time and returns
a Series/DataFrame on the same index. Nothing here peeks forward: each value at
index *i* uses only bars <= *i*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Trend / momentum
# --------------------------------------------------------------------------


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # avg_loss == 0 makes rs undefined: an all-gain window is RSI 100, and a
    # perfectly flat window (no gain, no loss either) is a neutral 50.
    flat = (avg_loss == 0) & (avg_gain == 0)
    all_gain = (avg_loss == 0) & (avg_gain > 0)
    out = out.mask(all_gain, 100.0).mask(flat, 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / tr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame(
        {
            "adx": dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean(),
            "plus_di": plus_di,
            "minus_di": minus_di,
        }
    )


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct": pct_b}
    )


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    low_n = df["low"].rolling(k_period, min_periods=k_period).min()
    high_n = df["high"].rolling(k_period, min_periods=k_period).max()
    k = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(d_period, min_periods=d_period).mean()})


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(period, min_periods=period).mean()
    md = (tp - ma).abs().rolling(period, min_periods=period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """Rolling session VWAP, reset daily."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.normalize()
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vol


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def support_resistance(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """Distance to the rolling swing high/low, normalized by price."""
    hi = df["high"].rolling(lookback, min_periods=lookback // 2).max()
    lo = df["low"].rolling(lookback, min_periods=lookback // 2).min()
    rng = (hi - lo).replace(0, np.nan)
    return pd.DataFrame(
        {
            "sr_high": hi,
            "sr_low": lo,
            "dist_to_high": (hi - df["close"]) / df["close"],
            "dist_to_low": (df["close"] - lo) / df["close"],
            "range_position": (df["close"] - lo) / rng,
        }
    )


def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 250) -> pd.Series:
    """Volatility-regime feature (addendum §A): same signal, different meaning."""
    a = atr(df, period)
    return a.rolling(lookback, min_periods=max(20, lookback // 5)).rank(pct=True)


# --------------------------------------------------------------------------
# Candlestick patterns (§5, rule-based half of the pattern analyzer)
# --------------------------------------------------------------------------


def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    body_ratio = body / rng

    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_bull = prev_c > prev_o
    bull = c > o

    bullish_engulfing = (~prev_bull) & bull & (c >= prev_o) & (o <= prev_c)
    bearish_engulfing = prev_bull & (~bull) & (c <= prev_o) & (o >= prev_c)

    pin_bar_bull = (lower_wick > 2 * body) & (upper_wick < body) & (body_ratio < 0.35)
    pin_bar_bear = (upper_wick > 2 * body) & (lower_wick < body) & (body_ratio < 0.35)

    doji = body_ratio < 0.1
    inside_bar = (h < h.shift(1)) & (l > l.shift(1))
    outside_bar = (h > h.shift(1)) & (l < l.shift(1))

    return pd.DataFrame(
        {
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick / rng,
            "lower_wick_ratio": lower_wick / rng,
            "pat_bullish_engulfing": bullish_engulfing.astype(float),
            "pat_bearish_engulfing": bearish_engulfing.astype(float),
            "pat_pin_bull": pin_bar_bull.astype(float),
            "pat_pin_bear": pin_bar_bear.astype(float),
            "pat_doji": doji.astype(float),
            "pat_inside": inside_bar.astype(float),
            "pat_outside": outside_bar.astype(float),
        }
    )


# --------------------------------------------------------------------------
# The per-timeframe bundle
# --------------------------------------------------------------------------

INDICATOR_VERSION = "ind-v1"


def compute_indicators(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Full indicator frame for one symbol/timeframe.

    Features are expressed *relatively* (ratios, distances in ATR/%), not as raw
    price levels — a model trained on raw levels stops working the moment price
    leaves the training range.
    """
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    out["close"] = close
    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)

    a = atr(df, atr_period)
    out["atr"] = a
    out["atr_pct"] = a / close
    out["atr_percentile"] = atr_percentile(df, atr_period)

    for p in (10, 20, 50, 200):
        m = ema(close, p)
        out[f"ema_{p}_dist"] = (close - m) / close
    out["ema_fast_slow"] = (ema(close, 10) - ema(close, 50)) / close

    out["rsi_14"] = rsi(close, 14)
    out = out.join(macd(close))
    out["macd_hist_norm"] = out["macd_hist"] / close
    out = out.join(adx(df))
    out = out.join(bollinger(close)[["bb_width", "bb_pct"]])
    out = out.join(stochastic(df))
    out["cci_20"] = cci(df)
    out["obv_slope"] = obv(df).diff(10) / df["volume"].rolling(10).sum().replace(0, np.nan)
    out = out.join(support_resistance(df)[["dist_to_high", "dist_to_low", "range_position"]])
    out = out.join(candle_patterns(df))

    # Session / clock context — FX behaves differently by session.
    out["hour"] = out.index.hour + out.index.minute / 60.0
    out["dow"] = out.index.dayofweek.astype(float)
    out["session_london"] = ((out.index.hour >= 7) & (out.index.hour < 16)).astype(float)
    out["session_ny"] = ((out.index.hour >= 12) & (out.index.hour < 21)).astype(float)
    out["session_asia"] = ((out.index.hour >= 23) | (out.index.hour < 8)).astype(float)

    # macd/macd_signal are absolute price-scale; drop in favour of the norm.
    return out.drop(columns=["macd", "macd_signal"], errors="ignore")
