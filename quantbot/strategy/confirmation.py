"""Entry timing — wait for the market to confirm a setup before acting.

A setup says "conditions are right". It does not say "the move has started".
Entering on the trigger bar means paying for every setup that immediately
fizzles; waiting for confirmation trades fewer of them, later and worse-priced,
but only after price has actually moved your way.

Which is better is an empirical question, not a settled one — hence the modes:

  off      enter on the trigger bar (default; fastest, most trades)
  momentum wait until price moves `confirm_atr_mult` x ATR in the signal's
           favour before entering
  pullback wait for price to come *back* toward the trigger price, for a
           better entry at the cost of missing runaway moves

Pending entries expire after `max_wait_bars`, because a setup nobody confirmed
within its own horizon is not a setup any more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ..contracts import Direction, as_utc

log = logging.getLogger(__name__)


@dataclass
class PendingEntry:
    key: str
    symbol: str
    direction: Direction
    trigger_price: float
    atr: float
    setup_names: str
    confidence: float
    created_at: datetime
    bars_waited: int = 0
    reasons: list[str] = field(default_factory=list)


class ConfirmationPolicy:
    """Holds setups that have fired but not yet earned an entry."""

    def __init__(
        self,
        mode: str = "off",
        confirm_atr_mult: float = 0.25,
        max_wait_bars: int = 3,
    ) -> None:
        self.mode = mode
        self.confirm_atr_mult = confirm_atr_mult
        self.max_wait_bars = max_wait_bars
        self._pending: dict[str, PendingEntry] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def pending_for(self, key: str) -> PendingEntry | None:
        return self._pending.get(key)

    def clear(self, key: str) -> None:
        self._pending.pop(key, None)

    def register(
        self,
        key: str,
        symbol: str,
        direction: Direction,
        price: float,
        atr: float,
        setup_names: str,
        confidence: float,
        now: datetime,
        reasons: list[str] | None = None,
    ) -> PendingEntry:
        entry = PendingEntry(
            key=key,
            symbol=symbol,
            direction=direction,
            trigger_price=price,
            atr=atr,
            setup_names=setup_names,
            confidence=confidence,
            created_at=as_utc(now),
            reasons=list(reasons or []),
        )
        self._pending[key] = entry
        return entry

    def check(self, key: str, price: float) -> tuple[bool, str]:
        """Has a pending entry been confirmed, or should it expire?

        Returns (confirmed, note). An expired entry is dropped and reported.
        """
        entry = self._pending.get(key)
        if entry is None:
            return False, ""

        entry.bars_waited += 1
        if entry.bars_waited > self.max_wait_bars:
            self.clear(key)
            return False, (
                f"pending {entry.setup_names} expired unconfirmed after "
                f"{self.max_wait_bars} bars"
            )

        distance = self.confirm_atr_mult * max(entry.atr, 0.0)
        moved = (price - entry.trigger_price) * entry.direction.sign

        if self.mode == "momentum":
            if moved >= distance:
                self.clear(key)
                return True, f"confirmed: moved {moved:.5f} in favour"
            return False, (
                f"waiting for {entry.setup_names}: {moved:.5f}/{distance:.5f} confirmation"
            )

        if self.mode == "pullback":
            # Favourable retrace: price came back against the signal by the
            # confirmation distance, giving a better entry on the same thesis.
            if -moved >= distance:
                self.clear(key)
                return True, f"confirmed on pullback of {-moved:.5f}"
            # Ran away without us — abandon rather than chase.
            if moved >= distance * 3:
                self.clear(key)
                return False, f"{entry.setup_names} ran away before the pullback"
            return False, f"waiting for a pullback on {entry.setup_names}"

        # Unknown mode: fail open rather than silently blocking all trading.
        log.warning("unknown confirmation mode %r; entering immediately", self.mode)
        self.clear(key)
        return True, "confirmation disabled (unknown mode)"

    def prune(self, live_keys: set[str]) -> None:
        """Drop pending entries whose symbol/mode is no longer being evaluated."""
        for key in list(self._pending):
            if key not in live_keys:
                self._pending.pop(key, None)
