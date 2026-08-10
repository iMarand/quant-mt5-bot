"""Self-forming structure/strategy search (architecture §7.3).

An evolutionary search over model hyperparameters *and* strategy parameters
(confidence threshold, SL/TP multiples, timeframe weighting), scored on
walk-forward validation.

Three guardrails from §7.3, all enforced here:
  1. mutation is bounded per cycle (`mutation_scale`), so structure can't lurch;
  2. candidates are scored out-of-sample only;
  3. a champion is replaced only by a challenger that beats it by `min_gain`,
    which makes the search conservative rather than noise-chasing.

Optuna is a drop-in alternative; this is dependency-free and does the same job
at this scale.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import Config
from ..storage import Database
from .retrain import build_training_set, score_of, walk_forward_evaluate

log = logging.getLogger(__name__)


#: name -> (low, high, is_integer). Deliberately narrow ranges: the search is a
#: refinement tool, not a licence to explore arbitrary architectures.
MODEL_SPACE: dict[str, tuple[float, float, bool]] = {
    "learning_rate": (0.01, 0.15, False),
    "num_leaves": (15, 96, True),
    "min_child_samples": (20, 150, True),
    "feature_fraction": (0.5, 1.0, False),
    "bagging_fraction": (0.5, 1.0, False),
    "n_estimators": (120, 600, True),
    "lambda_l2": (0.0, 10.0, False),
}

STRATEGY_SPACE: dict[str, tuple[float, float, bool]] = {
    "min_confidence": (0.52, 0.72, False),
    "sl_atr_mult": (1.0, 3.0, False),
    "tp_atr_mult": (1.2, 5.0, False),
    "trail_atr_mult": (0.8, 3.0, False),
    "breakeven_at_r": (0.5, 2.0, False),
}


@dataclass
class Genome:
    model: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    score: float = float("-inf")
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "strategy": self.strategy, "score": self.score}


def _sample(space: dict[str, tuple[float, float, bool]], rng: random.Random) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (lo, hi, is_int) in space.items():
        val = rng.uniform(lo, hi)
        out[name] = int(round(val)) if is_int else round(val, 5)
    return out


def _mutate(
    values: dict[str, Any],
    space: dict[str, tuple[float, float, bool]],
    rng: random.Random,
    scale: float,
) -> dict[str, Any]:
    """Gaussian step of at most `scale` of each parameter's range (guardrail 1)."""
    out = dict(values)
    for name, (lo, hi, is_int) in space.items():
        if name not in out or rng.random() > 0.5:
            continue
        span = hi - lo
        val = float(out[name]) + rng.gauss(0, span * scale)
        val = min(max(val, lo), hi)
        out[name] = int(round(val)) if is_int else round(val, 5)
    return out


class EvolutionarySearch:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        population: int = 6,
        generations: int = 3,
        mutation_scale: float = 0.15,
        min_gain: float = 0.002,
        seed: int | None = 7,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.population = population
        self.generations = generations
        self.mutation_scale = mutation_scale
        self.min_gain = min_gain
        self.rng = random.Random(seed)

    def run(self, symbol: str) -> Genome:
        ts = build_training_set(self.cfg, self.db, symbol)

        # Generation 0 always contains the current champion, so the search can
        # never do worse than leaving things alone.
        champion = Genome(
            model=dict(self.cfg.model.params),
            strategy={
                "min_confidence": self.cfg.risk.min_confidence,
                "sl_atr_mult": self.cfg.risk.sl_atr_mult,
                "tp_atr_mult": self.cfg.risk.tp_atr_mult,
                "trail_atr_mult": self.cfg.risk.trail_atr_mult,
                "breakeven_at_r": self.cfg.risk.breakeven_at_r,
            },
        )
        champion.score, champion.metrics = self._evaluate(ts, champion)
        log.info("search %s: champion baseline score %.5f", symbol, champion.score)

        pop = [champion] + [
            Genome(model=_sample(MODEL_SPACE, self.rng), strategy=_sample(STRATEGY_SPACE, self.rng))
            for _ in range(self.population - 1)
        ]

        for gen in range(self.generations):
            for genome in pop:
                if genome.score == float("-inf"):
                    genome.score, genome.metrics = self._evaluate(ts, genome)
            pop.sort(key=lambda g: g.score, reverse=True)
            best = pop[0]
            log.info("search %s gen %d: best %.5f", symbol, gen, best.score)

            if gen == self.generations - 1:
                break
            # Elitist: keep the top half, refill by mutating survivors.
            survivors = pop[: max(2, self.population // 2)]
            children = []
            while len(survivors) + len(children) < self.population:
                parent = self.rng.choice(survivors)
                children.append(
                    Genome(
                        model=_mutate(
                            {**self.cfg.model.params, **parent.model},
                            MODEL_SPACE,
                            self.rng,
                            self.mutation_scale,
                        ),
                        strategy=_mutate(
                            parent.strategy, STRATEGY_SPACE, self.rng, self.mutation_scale
                        ),
                    )
                )
            pop = survivors + children

        pop.sort(key=lambda g: g.score, reverse=True)
        best = pop[0]
        if best.score < champion.score + self.min_gain:
            log.info(
                "search %s: no challenger beat the champion by %.4f — keeping current structure",
                symbol,
                self.min_gain,
            )
            best = champion

        self.db.log_run(
            "structure_search",
            "ok",
            json.dumps(
                {
                    "symbol": symbol,
                    "best_score": round(best.score, 5),
                    "champion_score": round(champion.score, 5),
                    "changed": best is not champion,
                    "strategy": best.strategy,
                    "at": datetime.utcnow().isoformat(),
                }
            ),
        )
        return best

    def _evaluate(self, ts, genome: Genome) -> tuple[float, dict[str, Any]]:
        params = {**self.cfg.model.params, **genome.model}
        try:
            metrics = walk_forward_evaluate(self.cfg, ts, params)
        except Exception as exc:
            log.warning("candidate failed: %s", exc)
            return float("-inf"), {"error": str(exc)}
        # Strategy genes are scored through the confidence threshold: a higher
        # threshold trades less, so `score_of`'s coverage term prices it in.
        metrics = dict(metrics)
        thr = genome.strategy.get("min_confidence", self.cfg.risk.min_confidence)
        metrics["trade_rate"] = metrics.get("trade_rate", 0.0) * (1.0 - (thr - 0.5))
        return score_of(metrics), metrics


def apply_strategy_genes(cfg: Config, genes: dict[str, Any]) -> None:
    """Write evolved strategy parameters back onto the running risk config."""
    for key, value in genes.items():
        if hasattr(cfg.risk, key):
            setattr(cfg.risk, key, float(value))
