"""Indicator/price-action setups.

Each is a discrete pattern with explicit entry conditions and a quality score
derived from how strongly the bar matched. They are intentionally readable: if
one of these loses money you can point at the exact condition that fired.

Higher-timeframe context is used for *permission* (trade with the H1 trend) and
the base timeframe for *timing* (enter on the pullback/rejection). A setup
returns None whenever the features it needs are unavailable, so a missing
timeframe silently disables the setups that depend on it rather than guessing.
"""

from __future__ import annotations

from ..contracts import Direction
from .base import Setup, Strategy, StrategyContext, scale


class TrendPullback(Strategy):
    """Trend continuation: HTF trending, base timeframe pulled back, momentum turning.

    The classic "buy the dip in an uptrend". Requires a real trend (ADX), a
    genuine pullback (RSI/Bollinger position off the extreme), and evidence the
    pullback is ending rather than continuing.
    """

    name = "trend_pullback"
    tags = {"technical", "trend"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        htf: str = "H1",
        min_adx: float = 20.0,
        pullback_rsi: tuple[float, float] = (38.0, 58.0),
    ) -> None:
        super().__init__(enabled, weight)
        self.htf = htf
        self.min_adx = min_adx
        self.pullback_rsi = pullback_rsi

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        trend = ctx.f("ema_fast_slow", self.htf)
        adx = ctx.f("adx", self.htf)
        rsi = ctx.b("rsi_14")
        macd = ctx.b("macd_hist_norm")
        if trend is None or adx is None or rsi is None or macd is None:
            return None
        if adx < self.min_adx:
            return None

        up = trend > 0
        lo, hi = self.pullback_rsi
        if up:
            # Pullback in an uptrend: RSI cooled off but not broken down.
            if not (lo <= rsi <= hi):
                return None
            # Momentum must be turning back up, not still falling.
            if ctx.rising("macd_hist_norm") is not True:
                return None
            direction = Direction.LONG
            depth = scale(hi - rsi, 0, hi - lo)
        else:
            if not (100 - hi <= rsi <= 100 - lo):
                return None
            if ctx.rising("macd_hist_norm") is not False:
                return None
            direction = Direction.SHORT
            depth = scale(rsi - (100 - hi), 0, hi - lo)

        trend_strength = scale(adx, self.min_adx, 45.0)
        quality = 0.45 * trend_strength + 0.35 * depth + 0.20 * scale(abs(trend), 0.0, 0.002)
        return Setup(
            name=self.name,
            direction=direction,
            quality=quality,
            reasons=[
                f"{self.htf} trend {'up' if up else 'down'} (adx {adx:.0f})",
                f"pullback rsi {rsi:.0f}",
                "momentum turning",
            ],
            tags=set(self.tags),
        )


class Breakout(Strategy):
    """Range breakout with volatility expansion — and it must be *fresh*.

    `range_position` crossing the threshold this bar (not already above it) is
    what separates a breakout from chasing an extended move.
    """

    name = "breakout"
    tags = {"technical", "breakout"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        threshold: float = 0.95,
        min_adx: float = 18.0,
        min_atr_percentile: float = 0.45,
    ) -> None:
        super().__init__(enabled, weight)
        self.threshold = threshold
        self.min_adx = min_adx
        self.min_atr_percentile = min_atr_percentile

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        pos = ctx.b("range_position")
        prev_pos = ctx.prev_b("range_position")
        adx = ctx.b("adx")
        atr_pct = ctx.b("atr_percentile")
        bb_width = ctx.b("bb_width")
        prev_width = ctx.prev_b("bb_width")
        if None in (pos, prev_pos, adx, atr_pct):
            return None
        if adx < self.min_adx or atr_pct < self.min_atr_percentile:
            return None

        expanding = bb_width is not None and prev_width is not None and bb_width > prev_width

        if pos >= self.threshold and prev_pos < self.threshold:
            direction = Direction.LONG
        elif pos <= 1 - self.threshold and prev_pos > 1 - self.threshold:
            direction = Direction.SHORT
        else:
            return None  # not a fresh break

        quality = (
            0.40 * scale(atr_pct, self.min_atr_percentile, 0.95)
            + 0.35 * scale(adx, self.min_adx, 40.0)
            + 0.25 * (1.0 if expanding else 0.0)
        )
        return Setup(
            name=self.name,
            direction=direction,
            quality=quality,
            reasons=[
                f"fresh break of {self.threshold:.0%} range",
                f"atr percentile {atr_pct:.2f}",
                "bands expanding" if expanding else "bands flat",
            ],
            tags=set(self.tags),
        )


class MeanReversion(Strategy):
    """Fade a Bollinger extreme — but only in a ranging market, with rejection.

    Requires low ADX (no trend to fight) plus a reversal candle, because "price
    is extended" alone is how you get run over in a trend.
    """

    name = "mean_reversion"
    tags = {"technical", "range"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        max_adx: float = 20.0,
        band: float = 0.92,
    ) -> None:
        super().__init__(enabled, weight)
        self.max_adx = max_adx
        self.band = band

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        adx = ctx.b("adx")
        bb = ctx.b("bb_pct")
        rsi = ctx.b("rsi_14")
        if adx is None or bb is None or rsi is None:
            return None
        if adx > self.max_adx:
            return None

        bull_reject = (ctx.b("pat_pin_bull") or 0) + (ctx.b("pat_bullish_engulfing") or 0)
        bear_reject = (ctx.b("pat_pin_bear") or 0) + (ctx.b("pat_bearish_engulfing") or 0)

        if bb >= self.band and bear_reject > 0:
            direction, stretch, rsi_conf = Direction.SHORT, bb, scale(rsi, 60, 85)
        elif bb <= 1 - self.band and bull_reject > 0:
            direction, stretch, rsi_conf = Direction.LONG, 1 - bb, scale(40 - rsi, 0, 25)
        else:
            return None

        quality = (
            0.40 * scale(stretch, self.band, 1.05)
            + 0.35 * rsi_conf
            + 0.25 * scale(self.max_adx - adx, 0, self.max_adx)
        )
        return Setup(
            name=self.name,
            direction=direction,
            quality=quality,
            reasons=[
                f"ranging (adx {adx:.0f})",
                f"band extreme {bb:.2f}",
                "reversal candle",
            ],
            tags=set(self.tags),
        )


class SupportResistanceRejection(Strategy):
    """Rejection candle at a swing high/low — trade away from the level."""

    name = "sr_rejection"
    tags = {"technical", "level"}

    def __init__(
        self, enabled: bool = True, weight: float = 1.0, proximity: float = 0.0015
    ) -> None:
        super().__init__(enabled, weight)
        self.proximity = proximity

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        to_high = ctx.b("dist_to_high")
        to_low = ctx.b("dist_to_low")
        if to_high is None or to_low is None:
            return None

        bull = (ctx.b("pat_pin_bull") or 0) + (ctx.b("pat_bullish_engulfing") or 0)
        bear = (ctx.b("pat_pin_bear") or 0) + (ctx.b("pat_bearish_engulfing") or 0)
        wick_up = ctx.b("upper_wick_ratio") or 0.0
        wick_dn = ctx.b("lower_wick_ratio") or 0.0

        if to_high <= self.proximity and bear > 0:
            direction, closeness, wick = Direction.SHORT, to_high, wick_up
        elif to_low <= self.proximity and bull > 0:
            direction, closeness, wick = Direction.LONG, to_low, wick_dn
        else:
            return None

        quality = 0.5 * scale(self.proximity - closeness, 0, self.proximity) + 0.5 * scale(
            wick, 0.3, 0.7
        )
        return Setup(
            name=self.name,
            direction=direction,
            quality=quality,
            reasons=[
                "at swing level",
                f"rejection wick {wick:.2f}",
            ],
            tags=set(self.tags),
        )
