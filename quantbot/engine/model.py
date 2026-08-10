"""Model ensemble (architecture §5): rule baseline + gradient-boosted trees.

Start simple and add complexity only as data justifies it. The rule model always
works (zero training data required) and is what runs on day one; the LightGBM
model takes over per-symbol once it has beaten the baseline on walk-forward
validation. Both expose the same `predict_proba` contract, so the predictor
doesn't care which is active.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labeling import CLASS_ORDER, CLASS_TO_IDX

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Baseline: transparent rules
# --------------------------------------------------------------------------


@dataclass
class RuleModel:
    """Trend + momentum + mean-reversion votes, blended into a probability.

    Deliberately boring. Its job is to be a floor the learned model has to beat,
    and to keep the system tradeable while the journal fills up.
    """

    version: str = "rules-v1"
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "trend": 1.0,
            "momentum": 0.8,
            "meanrev": 0.5,
            "pattern": 0.6,
            "event": 0.7,
        }
    )
    feature_names_: list[str] = field(default_factory=list)

    def votes(self, row: pd.Series) -> dict[str, float]:
        """Each vote is in [-1, 1]; the names double as the explanation (§1.6)."""
        v: dict[str, float] = {}

        # Trend: higher-timeframe EMA alignment.
        trend_cols = [c for c in row.index if c.endswith("_ema_fast_slow")]
        if trend_cols:
            vals = [row[c] for c in trend_cols if pd.notna(row[c])]
            v["trend"] = float(np.tanh(np.mean(vals) * 2000)) if vals else 0.0

        # Momentum: MACD histogram + DI spread.
        macd_cols = [c for c in row.index if c.endswith("_macd_hist_norm")]
        # nanmean over an all-NaN warm-up row warns and returns NaN; filter first.
        macd_vals = [row[c] for c in macd_cols if pd.notna(row[c])]
        macd_v = float(np.tanh(float(np.mean(macd_vals)) * 5000)) if macd_vals else 0.0
        plus = _first(row, "plus_di")
        minus = _first(row, "minus_di")
        di_v = 0.0
        if plus is not None and minus is not None and pd.notna(plus) and pd.notna(minus):
            di_v = float(np.tanh((plus - minus) / 25.0))
        v["momentum"] = (macd_v + di_v) / 2.0

        # Mean reversion: stretched RSI / Bollinger position fades.
        rsi_v = _first(row, "rsi_14")
        bb = _first(row, "bb_pct")
        mr = 0.0
        if rsi_v is not None and pd.notna(rsi_v):
            mr += -np.tanh((rsi_v - 50) / 20.0)
        if bb is not None and pd.notna(bb):
            mr += -np.tanh((bb - 0.5) * 3.0)
        v["meanrev"] = float(mr / 2.0)

        # Candlestick patterns on the base timeframe.
        bull = _sum_suffix(row, ("_pat_bullish_engulfing", "_pat_pin_bull"))
        bear = _sum_suffix(row, ("_pat_bearish_engulfing", "_pat_pin_bear"))
        v["pattern"] = float(np.tanh(bull - bear))

        # Event: signed, decayed surprise, damped outside a news window.
        surprise = row.get("last_surprise_signed", 0.0)
        surprise = 0.0 if pd.isna(surprise) else float(surprise)
        v["event"] = float(np.tanh(surprise / 2.0))

        return v

    def score(self, row: pd.Series) -> tuple[float, dict[str, float]]:
        v = self.votes(row)
        total_w = sum(self.weights.get(k, 0.0) for k in v)
        if total_w == 0:
            return 0.0, v
        s = sum(self.weights.get(k, 0.0) * val for k, val in v.items()) / total_w
        return float(np.clip(s, -1, 1)), v

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Map the blended score onto P(down), P(flat), P(up)."""
        out = np.zeros((len(X), 3))
        for i, (_, row) in enumerate(X.iterrows()):
            s, _ = self.score(row)
            # Rules are weak evidence: cap the directional edge at ~+/-0.20.
            edge = 0.20 * s
            flat = 0.34
            out[i] = [(1 - flat) / 2 - edge, flat, (1 - flat) / 2 + edge]
        return out

    def explain(self, row: pd.Series) -> dict[str, float]:
        _, v = self.score(row)
        return {f"rule_{k}": round(val, 4) for k, val in v.items()}


def _first(row: pd.Series, suffix: str) -> float | None:
    for name in row.index:
        if name == suffix or name.endswith("_" + suffix):
            return row[name]
    return None


def _sum_suffix(row: pd.Series, suffixes: tuple[str, ...]) -> float:
    total = 0.0
    for name in row.index:
        if name.endswith(suffixes):
            val = row[name]
            if pd.notna(val):
                total += float(val)
    return total


# --------------------------------------------------------------------------
# Learned model
# --------------------------------------------------------------------------


class GBMModel:
    """LightGBM 3-class classifier over the engineered feature set."""

    def __init__(self, params: dict[str, Any] | None = None, version: str | None = None) -> None:
        self.params = dict(params or {})
        self.version = version or f"gbm-{datetime.utcnow():%Y%m%d%H%M%S}"
        self.model = None
        self.feature_names_: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.calibration_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None) -> GBMModel:
        import lightgbm as lgb

        self.feature_names_ = list(X.columns)
        y_idx = y.map(CLASS_TO_IDX).to_numpy()
        params = dict(self.params)
        params.setdefault("objective", "multiclass")
        params["num_class"] = 3
        n_estimators = params.pop("n_estimators", 300)
        self.model = lgb.LGBMClassifier(n_estimators=n_estimators, **params)
        self.model.fit(X, y_idx, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fitted")
        # A single row sliced from a DataFrame comes back as object dtype via
        # Series.to_frame().T; LightGBM rejects that, so coerce at the boundary.
        X = X.reindex(columns=self.feature_names_, fill_value=np.nan).astype(float)
        proba = self.model.predict_proba(X)
        # LGBMClassifier may drop classes absent from training data.
        if proba.shape[1] != 3:
            full = np.full((len(X), 3), 1e-6)
            for col, cls in enumerate(self.model.classes_):
                full[:, int(cls)] = proba[:, col]
            proba = full / full.sum(axis=1, keepdims=True)
        return proba

    def feature_importance(self, top: int = 15) -> dict[str, float]:
        if self.model is None:
            return {}
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return {}
        pairs = sorted(zip(self.feature_names_, imp), key=lambda kv: -kv[1])[:top]
        total = float(sum(imp)) or 1.0
        return {k: round(float(v) / total, 4) for k, v in pairs}

    def explain(self, row: pd.Series) -> dict[str, float]:
        """Top contributing features for one prediction, via LightGBM SHAP."""
        if self.model is None:
            return {}
        try:
            X = row.reindex(self.feature_names_).to_frame().T.astype(float)
            contrib = self.model.booster_.predict(X, pred_contrib=True)[0]
            n_feat = len(self.feature_names_)
            # Layout is [feat..., bias] repeated per class; use the up-class block.
            up_block = contrib[2 * (n_feat + 1) : 3 * (n_feat + 1) - 1]
            pairs = sorted(
                zip(self.feature_names_, up_block), key=lambda kv: -abs(kv[1])
            )[:8]
            return {k: round(float(v), 5) for k, v in pairs}
        except Exception as exc:  # explanation must never break inference
            log.debug("shap explain failed: %s", exc)
            return self.feature_importance(8)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "features": self.feature_names_,
                    "params": self.params,
                    "version": self.version,
                    "metrics": self.metrics,
                },
                fh,
            )
        path.with_suffix(".json").write_text(
            json.dumps(
                {"version": self.version, "metrics": self.metrics, "params": self.params},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> GBMModel:
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        m = cls(params=blob.get("params"), version=blob.get("version"))
        m.model = blob["model"]
        m.feature_names_ = blob["features"]
        m.metrics = blob.get("metrics", {})
        return m


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------


class Ensemble:
    """Blends the rule baseline and the learned model.

    `gbm_weight` starts low and is raised by the retraining job only when
    walk-forward metrics justify it (learning/retrain.py).
    """

    def __init__(self, rules: RuleModel, gbm: GBMModel | None = None, gbm_weight: float = 0.7):
        self.rules = rules
        self.gbm = gbm
        self.gbm_weight = gbm_weight if gbm is not None else 0.0

    @property
    def version(self) -> str:
        return f"{self.rules.version}" if self.gbm is None else f"{self.gbm.version}+{self.rules.version}"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_rules = self.rules.predict_proba(X)
        if self.gbm is None:
            return p_rules
        p_gbm = self.gbm.predict_proba(X)
        w = self.gbm_weight
        return w * p_gbm + (1 - w) * p_rules

    def explain(self, row: pd.Series) -> dict[str, float]:
        out = self.rules.explain(row)
        if self.gbm is not None:
            out.update(self.gbm.explain(row))
        return out


def proba_to_direction(proba: np.ndarray) -> tuple[int, float]:
    """(-1/0/+1, confidence). Confidence is P(chosen class), not P(up)."""
    idx = int(np.argmax(proba))
    return CLASS_ORDER[idx], float(proba[idx])


def directional_confidence(proba: np.ndarray) -> tuple[int, float]:
    """Direction from P(up) vs P(down) only, ignoring the flat class.

    This is what the risk layer wants: "given that I trade, which way, and how
    sure am I" — the flat class is a *reason not to trade*, handled separately.
    """
    p_down, p_flat, p_up = proba
    directional = p_up + p_down
    if directional <= 0:
        return 0, 0.5
    if p_up >= p_down:
        return 1, float(p_up / directional)
    return -1, float(p_down / directional)
