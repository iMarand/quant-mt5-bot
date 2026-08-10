"""Event-driven setups built on the economic calendar (Forex Factory).

This is the half of the design the whole ingestion layer exists to serve: trade
*because* something happened, not because a model felt like it.

Two distinct behaviours around a release:
  * `NewsReaction`  — after the number lands, trade the surprise direction once
    price confirms it.
  * `NewsBreakout`  — after a release, trade the break of the pre-news range
    regardless of the surprise sign (the market's read matters more than ours).

Both refuse to trade *before* a release. Holding into a high-impact print is a
volatility bet, not a directional edge, and the risk layer vetoes that window
anyway (`news_veto_minutes`).
"""

from __future__ import annotations

from ..contracts import Direction
from .base import Setup, Strategy, StrategyContext, scale


class NewsReaction(Strategy):
    """Trade the direction implied by a surprise, once price agrees.

    `last_surprise_signed` is already oriented for the pair: positive means the
    surprise favours the base currency (see features/events.py). Requiring price
    confirmation avoids the trap of fading the market because our sign
    convention says the number was "good".
    """

    name = "news_reaction"
    tags = {"news", "event"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        min_surprise_z: float = 0.6,
        react_within_min: float = 45.0,
        settle_after_min: float = 2.0,
        require_confirmation: bool = True,
    ) -> None:
        super().__init__(enabled, weight)
        self.min_surprise_z = min_surprise_z
        self.react_within_min = react_within_min
        self.settle_after_min = settle_after_min
        self.require_confirmation = require_confirmation

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        since = ctx.f("minutes_since_last_high")
        surprise = ctx.f("last_surprise_signed")
        if since is None or surprise is None:
            return None
        # Only in the window after the print, and not in the first chaotic
        # seconds where spreads blow out and fills are unreliable.
        if not (self.settle_after_min <= since <= self.react_within_min):
            return None
        if abs(surprise) < self.min_surprise_z:
            return None

        direction = Direction.LONG if surprise > 0 else Direction.SHORT
        momentum = ctx.b("macd_hist_norm")
        ret = ctx.b("ret_5")
        confirmed = True
        if self.require_confirmation:
            signals = [s for s in (momentum, ret) if s is not None]
            if not signals:
                return None
            confirmed = all((s > 0) == (direction is Direction.LONG) for s in signals)
            if not confirmed:
                return None

        recency = 1.0 - scale(since, self.settle_after_min, self.react_within_min)
        quality = 0.55 * scale(abs(surprise), self.min_surprise_z, 3.0) + 0.45 * recency
        return Setup(
            name=self.name,
            direction=direction,
            quality=quality,
            reasons=[
                f"surprise z={surprise:+.2f}",
                f"{since:.0f} min after high-impact release",
                "price confirms" if confirmed else "unconfirmed",
            ],
            tags=set(self.tags),
        )


class NewsBreakout(Strategy):
    """After a high-impact release, trade the break of the prior range.

    Deliberately agnostic about whether the number was "good": what matters is
    which way the market resolved it.
    """

    name = "news_breakout"
    tags = {"news", "event", "breakout"}

    def __init__(
        self,
        enabled: bool = True,
        weight: float = 1.0,
        react_within_min: float = 60.0,
        settle_after_min: float = 2.0,
        threshold: float = 0.9,
        min_atr_percentile: float = 0.55,
    ) -> None:
        super().__init__(enabled, weight)
        self.react_within_min = react_within_min
        self.settle_after_min = settle_after_min
        self.threshold = threshold
        self.min_atr_percentile = min_atr_percentile

    def evaluate(self, ctx: StrategyContext) -> Setup | None:
        since = ctx.f("minutes_since_last_high")
        if since is None or not (self.settle_after_min <= since <= self.react_within_min):
            return None
        pos = ctx.b("range_position")
        atr_pct = ctx.b("atr_percentile")
        if pos is None or atr_pct is None:
            return None
        # A release that didn't move anything is not a tradeable event.
        if atr_pct < self.min_atr_percentile:
            return None

        if pos >= self.threshold:
            direction = Direction.LONG
        elif pos <= 1 - self.threshold:
            direction = Direction.SHORT
        else:
            return None

        recency = 1.0 - scale(since, self.settle_after_min, self.react_within_min)
        quality = (
            0.45 * scale(atr_pct, self.min_atr_percentile, 0.95)
            + 0.30 * recency
            + 0.25 * scale(abs(pos - 0.5) * 2, self.threshold, 1.0)
        )
        return Setup(
            name=self.name,
            direction=direction,
            quality=quality,
            reasons=[
                f"{since:.0f} min after release",
                f"range break {pos:.2f}",
                f"vol expanded (atr pct {atr_pct:.2f})",
            ],
            tags=set(self.tags),
        )
