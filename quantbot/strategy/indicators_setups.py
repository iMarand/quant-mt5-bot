"""Indicator-family setups: EMA, volume, reversal and pure price action.

Grouped by the question each family answers:

  EMA        — is there a trend, and has it just turned?
  volume     — is the move backed by participation, or is it drift?
  reversal   — is the current move exhausted (divergence, failure)?
  price action — what did the candles themselves do, ignoring indicators?

They deliberately overlap. Confluence between families is what
`strategy.min_confluence` exists to reward: an EMA cross that also shows a
volume surge is a different proposition from either alone.
"""

from __future__ import annotations

from ..contracts import Direction
from .base import Setup, Strategy, StrategyContext, scale


# --------------------------------------------------------------------------
# EMA family
# --------------------------------------------------------------------------


class EmaCross(Strategy):
    """Fast/slow EMA cross, filtered by a higher-timeframe trend.

    A bare cross is famously noisy, so this requires the cross to be *fresh*
    (it flipped sign this bar) and to agree with the higher timeframe.
    """

    name = "ema_cross"
    tags = {"technical", "trend", "ema"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        htf: str = "H1",
        require_htf_agreement: bool = True,
        min_adx: float = 15.0,
    ) -> None:
        super().__init__(enabled, weight)
        self.htf = htf
        self.require_htf_agreement = require_htf_agreement
        self.min_adx = min_adx

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        now = ctx.b("ema_fast_slow")
        before = ctx.prev_b("ema_fast_slow")
        if now is None or before is None:
            return None
        # Fresh cross only: sign changed on this bar.
        if (now > 0) == (before > 0):
            return None

        direction = Direction.LONG if now > 0 else Direction.SHORT
        if self.require_htf_agreement:
            htf_trend = ctx.f("ema_fast_slow", self.htf)
            if htf_trend is None:
                return None
            if (htf_trend > 0) != (direction is Direction.LONG):
                return None

        adx = ctx.b("adx")
        if adx is not None and adx < self.min_adx:
            return None

        strength = scale(abs(now), 0.0, 0.0015)
        trend_q = scale(adx or self.min_adx, self.min_adx, 40.0)
        return Setup(
            name=self.name,
            direction=direction,
            quality=0.55 * strength + 0.45 * trend_q,
            reasons=[
                f"EMA cross {'up' if direction is Direction.LONG else 'down'}",
                f"{self.htf} agrees" if self.require_htf_agreement else "no htf filter",
                f"adx {adx:.0f}" if adx is not None else "adx n/a",
            ],
            tags=set(self.tags),
        )


class EmaRibbon(Strategy):
    """Pullback *to* a stacked EMA ribbon inside an established trend.

    "EMAs are stacked" is a market *state*, not an event — it stays true for
    most of a trend, so firing on it alone would trigger on nearly every bar.
    The tradeable event is price returning to the ribbon and the trend holding:
    this requires a fresh touch (price was outside `touch_distance` last bar and
    is inside it now), which turns the state into a discrete trigger.

    Uses distance-to-EMA features rather than raw EMA levels so it works
    unchanged across instruments at very different price scales.
    """

    name = "ema_ribbon"
    tags = {"technical", "trend", "ema"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        min_adx: float = 22.0,
        max_stretch: float = 0.004,
        touch_distance: float = 0.0008,
    ) -> None:
        super().__init__(enabled, weight)
        self.min_adx = min_adx
        self.max_stretch = max_stretch
        self.touch_distance = touch_distance

    def _stacked(self, d10, d20, d50):
        """Returns direction if the ladder is monotonic, else None."""
        if None in (d10, d20, d50):
            return None
        if d10 > 0 and d20 > d10 and d50 > d20:
            return Direction.LONG
        if d10 < 0 and d20 < d10 and d50 < d10:
            return Direction.SHORT
        return None

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        d10 = ctx.b("ema_10_dist")
        adx = ctx.b("adx")
        direction = self._stacked(d10, ctx.b("ema_20_dist"), ctx.b("ema_50_dist"))
        if direction is None or adx is None or adx < self.min_adx:
            return None

        stretch = abs(d10)
        if stretch > self.max_stretch:
            return None  # too far from the ribbon is a chase, not an entry

        # Fresh touch only: last bar must have been outside the touch band.
        prev_d10 = ctx.prev_b("ema_10_dist")
        if prev_d10 is None or abs(prev_d10) <= self.touch_distance:
            return None
        if stretch > self.touch_distance:
            return None

        return Setup(
            name=self.name,
            direction=direction,
            quality=0.6 * scale(adx, self.min_adx, 45.0)
            + 0.4 * (1.0 - scale(stretch, 0.0, self.touch_distance)),
            reasons=[
                f"EMA ribbon stacked {'up' if direction is Direction.LONG else 'down'}",
                f"adx {adx:.0f}",
                "price just pulled back to the ribbon",
            ],
            tags=set(self.tags),
        )


# --------------------------------------------------------------------------
# Volume family
# --------------------------------------------------------------------------


class VolumeSurge(Strategy):
    """A directional move backed by a jump in participation.

    MT5 gives tick volume rather than true traded volume for FX, which is a
    decent proxy for activity but not for size — so this is treated as a
    confirmation-grade signal, not a standalone thesis.
    """

    name = "volume_surge"
    tags = {"technical", "volume"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        min_body_ratio: float = 0.55,
        min_obv_slope: float = 0.15,
        min_atr_percentile: float = 0.5,
    ) -> None:
        super().__init__(enabled, weight)
        self.min_body_ratio = min_body_ratio
        self.min_obv_slope = min_obv_slope
        self.min_atr_percentile = min_atr_percentile

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        obv = ctx.b("obv_slope")
        body = ctx.b("body_ratio")
        ret = ctx.b("ret_1")
        atr_pct = ctx.b("atr_percentile")
        if None in (obv, body, ret, atr_pct):
            return None
        if body < self.min_body_ratio or atr_pct < self.min_atr_percentile:
            return None
        if abs(obv) < self.min_obv_slope:
            return None
        # Flow and price must point the same way, or it is churn.
        if (obv > 0) != (ret > 0):
            return None

        direction = Direction.LONG if ret > 0 else Direction.SHORT
        return Setup(
            name=self.name,
            direction=direction,
            quality=0.45 * scale(abs(obv), self.min_obv_slope, 1.0)
            + 0.35 * scale(body, self.min_body_ratio, 0.95)
            + 0.20 * scale(atr_pct, self.min_atr_percentile, 0.95),
            reasons=[
                f"volume flow {obv:+.2f} agrees with price",
                f"strong body ({body:.2f})",
            ],
            tags=set(self.tags),
        )


# --------------------------------------------------------------------------
# Reversal family
# --------------------------------------------------------------------------


class DivergenceReversal(Strategy):
    """Momentum failing to confirm a new extreme — classic divergence.

    Approximated from available features: price at a range extreme while RSI
    and the MACD histogram are *not* at a matching extreme, plus a rejection
    candle. Requires a non-trending backdrop, because fading a strong trend on
    divergence alone is how accounts die.
    """

    name = "divergence_reversal"
    tags = {"technical", "reversal"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        max_adx: float = 28.0,
        extreme: float = 0.85,
    ) -> None:
        super().__init__(enabled, weight)
        self.max_adx = max_adx
        self.extreme = extreme

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        pos = ctx.b("range_position")
        rsi = ctx.b("rsi_14")
        macd = ctx.b("macd_hist_norm")
        prev_macd = ctx.prev_b("macd_hist_norm")
        adx = ctx.b("adx")
        if None in (pos, rsi, macd, prev_macd, adx):
            return None
        if adx > self.max_adx:
            return None

        bull_reject = (ctx.b("pat_pin_bull") or 0) + (ctx.b("pat_bullish_engulfing") or 0)
        bear_reject = (ctx.b("pat_pin_bear") or 0) + (ctx.b("pat_bearish_engulfing") or 0)

        if pos >= self.extreme and rsi < 70 and macd < prev_macd and bear_reject > 0:
            direction, conf = Direction.SHORT, scale(70 - rsi, 0, 20)
        elif pos <= 1 - self.extreme and rsi > 30 and macd > prev_macd and bull_reject > 0:
            direction, conf = Direction.LONG, scale(rsi - 30, 0, 20)
        else:
            return None

        return Setup(
            name=self.name,
            direction=direction,
            quality=0.45 * conf
            + 0.30 * scale(abs(pos - 0.5) * 2, self.extreme, 1.0)
            + 0.25 * scale(self.max_adx - adx, 0, self.max_adx),
            reasons=[
                "price at extreme, momentum not confirming",
                f"rsi {rsi:.0f}",
                "reversal candle",
            ],
            tags=set(self.tags),
        )


# --------------------------------------------------------------------------
# Price-action family
# --------------------------------------------------------------------------


class PriceAction(Strategy):
    """Candles only — no oscillators, no moving averages.

    Engulfing or pin-bar rejection, with a meaningful body and a wick pointing
    the right way. Deliberately indicator-free so it fires in conditions where
    smoothed indicators are still catching up.
    """

    name = "price_action"
    tags = {"technical", "price_action"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        min_body_ratio: float = 0.45,
        min_wick_ratio: float = 0.35,
        require_trend_agreement: bool = False,
        htf: str = "H1",
    ) -> None:
        super().__init__(enabled, weight)
        self.min_body_ratio = min_body_ratio
        self.min_wick_ratio = min_wick_ratio
        self.require_trend_agreement = require_trend_agreement
        self.htf = htf

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        engulf_bull = ctx.b("pat_bullish_engulfing") or 0
        engulf_bear = ctx.b("pat_bearish_engulfing") or 0
        pin_bull = ctx.b("pat_pin_bull") or 0
        pin_bear = ctx.b("pat_pin_bear") or 0
        body = ctx.b("body_ratio")
        lower = ctx.b("lower_wick_ratio")
        upper = ctx.b("upper_wick_ratio")
        if body is None or lower is None or upper is None:
            return None

        if engulf_bull > 0 and body >= self.min_body_ratio:
            direction, kind, q = Direction.LONG, "bullish engulfing", scale(body, self.min_body_ratio, 0.95)
        elif engulf_bear > 0 and body >= self.min_body_ratio:
            direction, kind, q = Direction.SHORT, "bearish engulfing", scale(body, self.min_body_ratio, 0.95)
        elif pin_bull > 0 and lower >= self.min_wick_ratio:
            direction, kind, q = Direction.LONG, "bullish pin bar", scale(lower, self.min_wick_ratio, 0.8)
        elif pin_bear > 0 and upper >= self.min_wick_ratio:
            direction, kind, q = Direction.SHORT, "bearish pin bar", scale(upper, self.min_wick_ratio, 0.8)
        else:
            return None

        if self.require_trend_agreement:
            trend = ctx.f("ema_fast_slow", self.htf)
            if trend is None or (trend > 0) != (direction is Direction.LONG):
                return None

        return Setup(
            name=self.name,
            direction=direction,
            quality=q,
            reasons=[kind, f"body {body:.2f}"],
            tags=set(self.tags),
        )


# --------------------------------------------------------------------------
# Session family
# --------------------------------------------------------------------------


class SessionOpenRange(Strategy):
    """Break of the range established at a session open.

    The first hours of London and New York routinely set the day's direction;
    this trades the break of that early range while the session is still young.
    """

    name = "session_open_range"
    tags = {"technical", "session", "breakout"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        open_hours: tuple[float, ...] = (7.0, 12.0),
        window_hours: float = 3.0,
        threshold: float = 0.9,
        min_atr_percentile: float = 0.4,
    ) -> None:
        super().__init__(enabled, weight)
        self.open_hours = open_hours
        self.window_hours = window_hours
        self.threshold = threshold
        self.min_atr_percentile = min_atr_percentile

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        hour = ctx.b("hour")
        pos = ctx.b("range_position")
        atr_pct = ctx.b("atr_percentile")
        if hour is None or pos is None or atr_pct is None:
            return None
        if atr_pct < self.min_atr_percentile:
            return None

        since_open = min((hour - h) % 24 for h in self.open_hours)
        if since_open > self.window_hours:
            return None

        if pos >= self.threshold:
            direction = Direction.LONG
        elif pos <= 1 - self.threshold:
            direction = Direction.SHORT
        else:
            return None

        freshness = 1.0 - scale(since_open, 0.0, self.window_hours)
        return Setup(
            name=self.name,
            direction=direction,
            quality=0.5 * freshness + 0.5 * scale(atr_pct, self.min_atr_percentile, 0.95),
            reasons=[
                f"{since_open:.1f}h into the session open",
                f"range break {pos:.2f}",
            ],
            tags=set(self.tags),
        )
