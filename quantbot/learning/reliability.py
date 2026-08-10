"""Per-setup reliability learned from the journal.

The model's confidence adjustment answers "does the data agree with this
direction right now?". It does not answer "has this *particular setup* ever
worked?". This module answers the second question from the bot's own recorded
outcomes, and feeds it back as a multiplier on setup quality.

Two properties matter more than sophistication here:

  * **Shrinkage.** A setup that is 3-for-4 is not a 75% setup. Every estimate is
    pulled toward the 0.5 prior with a strength of `prior_strength` pseudo-
    observations, so a young setup is treated as unproven rather than brilliant.
  * **Bounded influence.** The multiplier is clipped, so no amount of history
    lets one setup dominate or silences another entirely. Reliability tilts the
    book; it does not replace the operator's `enabled` switches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..storage import Database

log = logging.getLogger(__name__)


@dataclass
class SetupStats:
    name: str
    n: int
    wins: int
    raw_accuracy: float
    shrunk_accuracy: float
    weight: float

    def describe(self) -> str:
        return (
            f"{self.name:<28} n={self.n:<5} acc={self.raw_accuracy:.3f} "
            f"-> {self.shrunk_accuracy:.3f}  weight x{self.weight:.2f}"
        )


@dataclass
class SetupReliability:
    """Multipliers on setup quality, measured from realized outcomes."""

    stats: dict[str, SetupStats] = field(default_factory=dict)
    min_weight: float = 0.6
    max_weight: float = 1.4

    def weight(self, setup_name: str) -> float:
        """Multiplier for a setup. Unknown setups get 1.0 — no opinion."""
        stat = self.stats.get(setup_name)
        return stat.weight if stat is not None else 1.0

    def describe(self) -> list[str]:
        return [s.describe() for s in sorted(self.stats.values(), key=lambda s: -s.n)]

    @classmethod
    def from_journal(
        cls,
        db: Database,
        prior_strength: float = 40.0,
        min_weight: float = 0.6,
        max_weight: float = 1.4,
        min_samples: int = 20,
    ) -> SetupReliability:
        """Build from scored predictions. Empty journal -> no opinions at all."""
        rel = cls(min_weight=min_weight, max_weight=max_weight)
        df = db.predictions_df()
        if df.empty or "setup" not in df.columns:
            return rel

        scored = df[(df["correct"].notna()) & (df["direction"] != "flat")]
        scored = scored[scored["setup"].fillna("") != ""]
        if scored.empty:
            return rel

        # A composite trigger like "breakout+volume_surge" credits both setups,
        # so each one accumulates evidence from every context it fired in.
        exploded = scored.assign(setup=scored["setup"].str.split("+")).explode("setup")
        exploded["setup"] = exploded["setup"].str.strip()

        for name, group in exploded.groupby("setup"):
            n = int(len(group))
            wins = int(group["correct"].sum())
            raw = wins / n if n else 0.5
            # Beta-binomial posterior mean with a 0.5-centred prior.
            shrunk = (wins + 0.5 * prior_strength) / (n + prior_strength)
            weight = 1.0
            if n >= min_samples:
                # Map accuracy onto a multiplier around 1.0: 0.5 -> 1.0,
                # better -> up, worse -> down, then clip.
                weight = min(max(shrunk / 0.5, min_weight), max_weight)
            rel.stats[name] = SetupStats(
                name=name,
                n=n,
                wins=wins,
                raw_accuracy=round(raw, 4),
                shrunk_accuracy=round(shrunk, 4),
                weight=round(weight, 4),
            )

        log.info("setup reliability from %d scored predictions", len(scored))
        return rel


def setup_quality_columns(book, df, base_tf: str, prefix: str = "sq_"):
    """Per-bar setup qualities, as model features.

    This is what makes the model *aware of the strategies*: instead of seeing
    only raw indicators, it gets a column per setup holding that setup's quality
    on that bar (0 when it did not fire, signed by direction). The learner can
    then discover setup-conditional edges — "breakout works in this regime,
    mean_reversion does not" — rather than being told to trust them equally.
    """
    import pandas as pd

    from ..strategy.base import StrategyContext

    names = [s.name for s in book.strategies]
    out = {f"{prefix}{n}": [0.0] * len(df) for n in names}
    prev = None
    for i, (_ts, row) in enumerate(df.iterrows()):
        ctx = StrategyContext(row, prev, base_tf)
        for strat in book.strategies:
            if not strat.enabled:
                continue
            try:
                setup = strat.evaluate(ctx)
            except Exception:
                continue
            if setup is not None:
                # Signed so direction is part of the signal, not just presence.
                out[f"{prefix}{strat.name}"][i] = setup.quality * setup.direction.sign
        prev = row
    return pd.DataFrame(out, index=df.index)
