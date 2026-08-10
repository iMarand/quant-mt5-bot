"""StrategyBook — runs every setup, resolves conflicts, applies the model.

The order of operations encodes the design decision:

    1. strategies vote        -> no trigger means no trade, full stop
    2. conflicts resolved     -> opposing setups cancel rather than net out
    3. confluence scored      -> agreeing setups raise conviction
    4. model *adjusts*        -> can damp or veto, can never create a trade

Step 4 is what "the model is an assistant" means concretely: it never appears
before step 1, so it cannot manufacture a signal on a bar where nothing fired.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..contracts import Direction
from .base import Setup, Strategy, StrategyContext
from .indicators_setups import (
    DivergenceReversal,
    EmaCross,
    EmaRibbon,
    PriceAction,
    SessionOpenRange,
    VolumeSurge,
)
from .news import NewsBreakout, NewsReaction
from .technical import Breakout, MeanReversion, SupportResistanceRejection, TrendPullback

log = logging.getLogger(__name__)

#: name -> factory. Config enables/tunes by name.
REGISTRY: dict[str, type[Strategy]] = {
    TrendPullback.name: TrendPullback,
    Breakout.name: Breakout,
    MeanReversion.name: MeanReversion,
    SupportResistanceRejection.name: SupportResistanceRejection,
    NewsReaction.name: NewsReaction,
    NewsBreakout.name: NewsBreakout,
    EmaCross.name: EmaCross,
    EmaRibbon.name: EmaRibbon,
    VolumeSurge.name: VolumeSurge,
    DivergenceReversal.name: DivergenceReversal,
    PriceAction.name: PriceAction,
    SessionOpenRange.name: SessionOpenRange,
}


class Decision:
    """What the book concluded for one bar."""

    __slots__ = ("direction", "confidence", "setups", "reasons", "model_note", "vetoed_by")

    def __init__(
        self,
        direction: Direction = Direction.FLAT,
        confidence: float = 0.0,
        setups: list[Setup] | None = None,
        reasons: list[str] | None = None,
        model_note: str = "",
        vetoed_by: str = "",
    ) -> None:
        self.direction = direction
        self.confidence = confidence
        self.setups = setups or []
        self.reasons = reasons or []
        self.model_note = model_note
        self.vetoed_by = vetoed_by

    @property
    def triggered(self) -> bool:
        return self.direction is not Direction.FLAT

    @property
    def setup_names(self) -> str:
        return "+".join(s.name for s in self.setups) if self.setups else ""


def build_strategies(cfg) -> list[Strategy]:
    """Instantiate the enabled setups from config."""
    out: list[Strategy] = []
    for name, params in (cfg.setups or {}).items():
        cls = REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"unknown setup {name!r}; known: {sorted(REGISTRY)}")
        params = dict(params or {})
        if not params.pop("enabled", True):
            continue
        try:
            out.append(cls(enabled=True, **params))
        except TypeError as exc:
            raise ValueError(f"bad parameters for setup {name!r}: {exc}") from exc
    return out


class StrategyBook:
    def __init__(self, cfg, strategies: list[Strategy] | None = None) -> None:
        self.cfg = cfg
        self.strategies = strategies if strategies is not None else build_strategies(cfg)
        if not self.strategies:
            log.warning("no setups enabled — nothing will ever trigger")

    # -- steps 1-3 ---------------------------------------------------------
    def triggered_setups(self, ctx: StrategyContext) -> list[Setup]:
        found: list[Setup] = []
        for strat in self.strategies:
            if not strat.enabled:
                continue
            try:
                setup = strat.evaluate(ctx)
            except Exception as exc:  # a broken setup must not kill the cycle
                log.error("setup %s raised: %s", strat.name, exc)
                continue
            if setup is not None and setup.quality >= self.cfg.min_setup_quality:
                setup.quality *= strat.weight
                found.append(setup)
        return found

    def combine(self, setups: list[Setup]) -> Decision:
        if not setups:
            return Decision(reasons=["no setup triggered"])

        longs = [s for s in setups if s.direction is Direction.LONG]
        shorts = [s for s in setups if s.direction is Direction.SHORT]

        # Opposing setups cancel. Netting them out would invent a weak signal
        # from genuine disagreement, which is the opposite of what we want.
        if longs and shorts:
            return Decision(
                setups=setups,
                reasons=[
                    "conflicting setups: "
                    + ", ".join(s.describe() for s in setups)
                ],
                vetoed_by="conflict",
            )

        winners = longs or shorts
        if len(winners) < self.cfg.min_confluence:
            return Decision(
                setups=winners,
                reasons=[f"only {len(winners)} setup(s), need {self.cfg.min_confluence}"],
                vetoed_by="insufficient_confluence",
            )

        if self.cfg.require_news and not any(s.is_news for s in winners):
            return Decision(
                setups=winners,
                reasons=["require_news is on and no event setup fired"],
                vetoed_by="no_news_trigger",
            )

        direction = winners[0].direction
        best = max(s.quality for s in winners)
        # Confluence bonus is deliberately sublinear: two setups agreeing is
        # meaningfully better than one, four is not twice as good as two.
        bonus = self.cfg.confluence_bonus * np.log1p(len(winners) - 1)
        conviction = float(min(best + bonus, 1.0))

        confidence = self.cfg.base_confidence + (
            self.cfg.max_confidence - self.cfg.base_confidence
        ) * conviction
        reasons = [r for s in winners for r in s.reasons]
        return Decision(
            direction=direction,
            confidence=float(confidence),
            setups=winners,
            reasons=reasons,
        )

    # -- step 4 ------------------------------------------------------------
    def apply_model(self, decision: Decision, proba: np.ndarray | None) -> Decision:
        """Let the model adjust a decision the strategies already made.

        `proba` is [P(down), P(flat), P(up)]. The model may:
          * damp or boost confidence in proportion to its agreement, and
          * veto outright when it strongly disagrees.
        It may never flip the direction or create one.
        """
        if proba is None or not decision.triggered or self.cfg.effective_model_role() == "off":
            return decision

        p_down, _p_flat, p_up = (float(x) for x in proba)
        denom = p_down + p_up
        if denom <= 0:
            return decision
        agree = (p_up if decision.direction is Direction.LONG else p_down) / denom

        if agree < self.cfg.model_veto_below:
            decision.vetoed_by = "model_disagrees"
            decision.model_note = f"model agreement {agree:.2f} < {self.cfg.model_veto_below}"
            decision.direction = Direction.FLAT
            return decision

        # agree=0.5 is neutral -> no change; the scale is capped by assist_weight.
        adjust = 1.0 + self.cfg.model_assist_weight * (2 * agree - 1.0)
        decision.confidence = float(
            min(max(decision.confidence * adjust, 0.0), self.cfg.max_confidence)
        )
        decision.model_note = f"model agreement {agree:.2f} (x{adjust:.2f})"
        return decision

    # -- convenience -------------------------------------------------------
    def decide(
        self,
        row: pd.Series,
        prev: pd.Series | None,
        base_tf: str,
        proba: np.ndarray | None = None,
    ) -> Decision:
        ctx = StrategyContext(row, prev, base_tf)
        decision = self.combine(self.triggered_setups(ctx))
        return self.apply_model(decision, proba)
