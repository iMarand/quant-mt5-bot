"""Setup selector — "which of my strategies works in *this* context?"

This is a better-posed learning problem than the direction model, and it is the
one the study data actually supports.

The direction model asks "up or down?" of every bar. Most bars have no
answer, so it sits at the majority-class baseline. The selector asks a narrower
question, only of bars where a setup fired:

    given that `trend_pullback` just triggered long, in the London/NY overlap,
    at 14:00, in a trending regime, with ATR in the 70th percentile and no news
    window — has this combination historically won?

One model over all setups, with the setup name as a feature, so evidence is
shared: knowing that breakouts fail in thin sessions informs the estimate for
other breakout-like triggers rather than being learned twelve separate times.

Trained on counterfactual study rows (learning/counterfactual.py), which number
in the tens of thousands rather than the handful the live journal holds. At
runtime it scales a triggered setup's quality by how well that setup has done
in the context it just fired in.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Context the selector reasons over. Everything here is known at decision time.
NUMERIC_FEATURES = [
    "hour",
    "dow",
    "in_news_window",
    "minutes_since_last_high",
    "last_surprise_signed",
    "atr_percentile",
    "adx",
    "quality",
    "is_long",
]
CATEGORICAL_FEATURES = ["setup", "session", "regime", "symbol"]


def build_features(rows: pd.DataFrame, categories: dict[str, list[str]] | None = None):
    """Context -> model matrix. `categories` pins one-hot columns for inference."""
    df = pd.DataFrame(index=rows.index)
    for col in NUMERIC_FEATURES:
        if col == "is_long" and "direction" in rows.columns:
            df[col] = (rows["direction"] == "long").astype(float)
        else:
            df[col] = pd.to_numeric(rows.get(col), errors="coerce")

    cats = categories or {c: sorted(rows[c].dropna().unique()) for c in CATEGORICAL_FEATURES if c in rows}
    for col, values in cats.items():
        series = rows.get(col)
        for value in values:
            df[f"{col}={value}"] = (
                (series == value).astype(float) if series is not None else 0.0
            )
    return df.astype(float), cats


@dataclass
class SetupSelector:
    """P(this setup wins | this context)."""

    model: object | None = None
    categories: dict[str, list[str]] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    version: str = ""

    # -- training ----------------------------------------------------------
    def fit(self, study: pd.DataFrame, params: dict | None = None) -> SetupSelector:
        import lightgbm as lgb

        rows = study.dropna(subset=["won"]).copy()
        if len(rows) < 500:
            raise ValueError(f"only {len(rows)} study rows; need 500+ to fit a selector")

        X, cats = build_features(rows)
        y = rows["won"].astype(int)
        self.categories = cats
        self.feature_names = list(X.columns)

        settings = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 80,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_estimators": 300,
            "verbose": -1,
            **(params or {}),
        }
        n_estimators = settings.pop("n_estimators")
        self.model = lgb.LGBMClassifier(n_estimators=n_estimators, **settings)
        self.model.fit(X, y)
        self.version = f"selector-{datetime.utcnow():%Y%m%d%H%M%S}"
        return self

    def evaluate(self, study: pd.DataFrame, split: float = 0.7) -> dict:
        """Time-split validation: fit on the past, score on the future.

        Study rows are heavily overlapping in time, so a random split would leak.
        """
        rows = study.dropna(subset=["won"]).sort_values("ts")
        cut = rows["ts"].quantile(split)
        early, late = rows[rows["ts"] <= cut], rows[rows["ts"] > cut]
        if len(early) < 500 or len(late) < 200:
            return {"error": "not enough rows for a time split"}

        probe = SetupSelector().fit(early)
        p = probe.predict_proba(late)
        actual = late["won"].astype(int).to_numpy()

        base = float(actual.mean())
        brier = float(np.mean((p - actual) ** 2))
        # Does ranking by predicted win probability actually sort outcomes?
        order = np.argsort(-p)
        top_decile = actual[order[: max(1, len(actual) // 10)]].mean()
        bottom_decile = actual[order[-max(1, len(actual) // 10) :]].mean()

        metrics = {
            "n_train": int(len(early)),
            "n_test": int(len(late)),
            "base_rate": round(base, 4),
            "brier": round(brier, 4),
            "brier_baseline": round(float(np.mean((base - actual) ** 2)), 4),
            "top_decile_win_rate": round(float(top_decile), 4),
            "bottom_decile_win_rate": round(float(bottom_decile), 4),
            "decile_spread": round(float(top_decile - bottom_decile), 4),
            "split_at": str(cut),
        }
        self.metrics = metrics
        return metrics

    # -- inference ---------------------------------------------------------
    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(rows), 0.5)
        X, _ = build_features(rows, self.categories)
        X = X.reindex(columns=self.feature_names, fill_value=0.0)
        return self.model.predict_proba(X)[:, 1]

    def score_context(self, context: dict) -> float:
        """P(win) for one triggered setup in one context."""
        if self.model is None:
            return 0.5
        try:
            return float(self.predict_proba(pd.DataFrame([context]))[0])
        except Exception as exc:  # inference must never break a trading cycle
            log.debug("selector inference failed: %s", exc)
            return 0.5

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "categories": self.categories,
                    "feature_names": self.feature_names,
                    "metrics": self.metrics,
                    "version": self.version,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> SetupSelector | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                blob = pickle.load(fh)
        except Exception as exc:
            log.warning("could not load selector %s: %s", path, exc)
            return None
        return cls(
            model=blob.get("model"),
            categories=blob.get("categories", {}),
            feature_names=blob.get("feature_names", []),
            metrics=blob.get("metrics", {}),
            version=blob.get("version", ""),
        )


def selector_path(cfg) -> Path:
    return cfg.registry_path / "setup_selector.pkl"
