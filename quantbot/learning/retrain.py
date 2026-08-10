"""Periodic retraining with walk-forward validation (architecture §7.2).

Two guardrails make this safe to run unattended:
  * every candidate is scored on *purged, embargoed walk-forward* folds, never
    in-sample, so a model can't be promoted for memorizing;
  * a new model only becomes active if it beats the currently active one on the
    same folds — otherwise the incumbent stays and the run is logged as a
    no-promotion. That is the answer to "did today's retrain improve anything?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..engine.labeling import (
    CLASS_TO_IDX,
    purge_and_embargo,
    triple_barrier_labels,
    walk_forward_splits,
)
from ..engine.model import GBMModel
from ..engine.predictor import Predictor
from ..features import feature_columns
from ..ops.metrics import expected_calibration_error
from ..storage import Database

log = logging.getLogger(__name__)


@dataclass
class TrainingSet:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    features: list[str] = field(default_factory=list)


def build_training_set(cfg: Config, db: Database, symbol: str) -> TrainingSet:
    """Features + triple-barrier labels for one symbol on the base timeframe."""
    predictor = Predictor(cfg, db)
    feats_df = predictor.build_features(symbol, bars=cfg.data.history_bars)
    candles = db.load_candles(symbol, cfg.data.base_timeframe, limit=cfg.data.history_bars)

    labels = triple_barrier_labels(
        candles,
        horizon_bars=cfg.model.horizon_bars,
        atr_mult=cfg.model.label_atr_mult,
        atr_period=cfg.risk.atr_period,
    )
    joined = feats_df.join(labels, how="inner").dropna(subset=["label"])

    # Train only where the calendar reaches. Mixing news-aware rows with
    # news-blind ones teaches the model that event features are usually zero,
    # and puts the blind rows in the latest walk-forward folds where validation
    # actually happens.
    if cfg.model.limit_to_calendar_coverage:
        from ..connectors.calendar_import import calendar_coverage_end

        last_event = calendar_coverage_end(db)
        if last_event is not None:
            covered = joined.index <= last_event
            kept, total = int(covered.sum()), len(joined)
            if kept >= cfg.model.min_train_rows and kept < total:
                log.info(
                    "limiting training to calendar coverage: %d/%d rows (through %s)",
                    kept, total, last_event.date(),
                )
                joined = joined[covered]
            elif kept < cfg.model.min_train_rows:
                log.warning(
                    "calendar covers only %d rows (< min_train_rows=%d); training on "
                    "the full span instead — news features will be mostly empty",
                    kept, cfg.model.min_train_rows,
                )
    cols = feature_columns(joined)
    # Rows where every feature is NaN are warm-up bars, not data.
    joined = joined.dropna(subset=cols, thresh=max(1, int(len(cols) * 0.6)))
    if joined.empty:
        raise ValueError(f"no usable training rows for {symbol}")

    X = joined[cols].astype(float)
    y = joined["label"].astype(int)
    meta = joined[["fwd_return", "bars_held", "barrier_width"]]
    return TrainingSet(X=X, y=y, meta=meta, features=cols)


def _sample_weights(y: pd.Series, meta: pd.DataFrame) -> np.ndarray:
    """Down-weight overlapping labels and up-weight the rarer classes.

    Overlapping label windows mean neighbouring rows are near-duplicates; naive
    training treats them as independent evidence and overstates its confidence.
    """
    counts = y.value_counts()
    class_w = {k: len(y) / (len(counts) * v) for k, v in counts.items()}
    w = y.map(class_w).to_numpy(dtype=float)
    held = meta["bars_held"].to_numpy(dtype=float)
    overlap = np.clip(held, 1, None)
    return w / overlap * overlap.mean()


def walk_forward_evaluate(
    cfg: Config, ts: TrainingSet, params: dict[str, Any]
) -> dict[str, Any]:
    """Out-of-sample metrics across expanding folds. This is the only score that counts."""
    n = len(ts.X)
    splits = walk_forward_splits(n, cfg.model.n_splits, min_train=cfg.model.min_train_rows)
    if not splits:
        raise ValueError(
            f"only {n} rows: need > {cfg.model.min_train_rows} for walk-forward validation"
        )

    accs, edges, briers, preds_all, truth_all, conf_all = [], [], [], [], [], []
    for train_idx, test_idx in splits:
        train_idx = purge_and_embargo(
            ts.X.index, train_idx, test_idx, cfg.model.horizon_bars, cfg.model.embargo_bars
        )
        if len(train_idx) < 100:
            continue
        X_tr, y_tr = ts.X.iloc[train_idx], ts.y.iloc[train_idx]
        X_te, y_te = ts.X.iloc[test_idx], ts.y.iloc[test_idx]
        if y_tr.nunique() < 2:
            continue

        model = GBMModel(params=params).fit(
            X_tr, y_tr, sample_weight=_sample_weights(y_tr, ts.meta.iloc[train_idx])
        )
        proba = model.predict_proba(X_te)
        idx = proba.argmax(axis=1)
        y_true_idx = y_te.map(CLASS_TO_IDX).to_numpy()
        accs.append(float((idx == y_true_idx).mean()))

        # Directional edge: accuracy on the rows where the model was not flat.
        directional = idx != 1
        if directional.sum() > 0:
            pred_dir = np.where(idx[directional] == 2, 1, -1)
            true_dir = y_te.to_numpy()[directional]
            hit = pred_dir == true_dir
            edges.append(float(hit.mean()))
            # Measure calibration on the *same* quantity the live path reports:
            # P(direction | trading), not the raw multiclass max probability.
            p = proba[directional]
            denom = (p[:, 0] + p[:, 2]).clip(min=1e-9)
            conf = np.maximum(p[:, 0], p[:, 2]) / denom
            conf_all.extend(conf.tolist())
            truth_all.extend(hit.astype(float).tolist())
        preds_all.extend(idx.tolist())

        onehot = np.zeros_like(proba)
        onehot[np.arange(len(y_true_idx)), y_true_idx] = 1
        briers.append(float(((proba - onehot) ** 2).sum(axis=1).mean()))

    if not accs:
        raise ValueError("no valid walk-forward folds (too little data after purging)")

    ece = (
        expected_calibration_error(pd.Series(conf_all), pd.Series(truth_all))
        if conf_all
        else float("nan")
    )
    return {
        "folds": len(accs),
        "accuracy": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "directional_accuracy": round(float(np.mean(edges)), 4) if edges else None,
        "multiclass_brier": round(float(np.mean(briers)), 4),
        "calibration_error": None if np.isnan(ece) else round(ece, 4),
        "trade_rate": round(float(np.mean([p != 1 for p in preds_all])), 4),
        "rows": n,
    }


def score_of(metrics: dict[str, Any]) -> float:
    """Single scalar for model comparison: directional edge, penalized for miscalibration."""
    edge = metrics.get("directional_accuracy") or 0.5
    ece = metrics.get("calibration_error") or 0.0
    coverage = min(metrics.get("trade_rate", 0.0) / 0.2, 1.0)  # reward acting sometimes
    return (edge - 0.5) * coverage - 0.5 * ece


def retrain_symbol(
    cfg: Config,
    db: Database,
    symbol: str,
    params: dict[str, Any] | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    """Train a candidate, validate it walk-forward, promote only if it wins."""
    tf = cfg.data.base_timeframe
    ts = build_training_set(cfg, db, symbol)
    params = dict(params or cfg.model.params)

    metrics = walk_forward_evaluate(cfg, ts, params)
    candidate_score = score_of(metrics)

    incumbent = db.active_model(symbol, tf)
    incumbent_score = None
    if incumbent is not None:
        import json

        incumbent_score = score_of(json.loads(incumbent["metrics"]))

    promote = activate and (incumbent_score is None or candidate_score > incumbent_score)

    version = f"gbm-{symbol}-{tf}-{datetime.utcnow():%Y%m%d%H%M%S}"
    model = GBMModel(params=params, version=version)
    model.fit(ts.X, ts.y, sample_weight=_sample_weights(ts.y, ts.meta))
    model.metrics = {
        **metrics,
        "score": round(candidate_score, 5),
        "top_features": model.feature_importance(15),
        "symbol": symbol,
        "timeframe": tf,
    }
    path = cfg.registry_path / f"{version}.pkl"
    model.save(path)
    db.register_model(
        version=version,
        path=str(path),
        symbol=symbol,
        timeframe=tf,
        metrics=model.metrics,
        params=params,
        activate=promote,
    )

    result = {
        "symbol": symbol,
        "version": version,
        "promoted": promote,
        "score": round(candidate_score, 5),
        "incumbent_score": None if incumbent_score is None else round(incumbent_score, 5),
        **metrics,
    }
    db.log_run("retrain", "ok" if promote else "no_promotion", str(result))
    log.info(
        "retrain %s: score=%.5f incumbent=%s promoted=%s",
        symbol,
        candidate_score,
        incumbent_score,
        promote,
    )
    return result


def retrain_all(cfg: Config, db: Database) -> list[dict[str, Any]]:
    out = []
    for symbol in cfg.data.symbols:
        try:
            out.append(retrain_symbol(cfg, db, symbol))
        except Exception as exc:
            log.error("retrain %s failed: %s", symbol, exc)
            db.alert("error", "retrain", f"{symbol}: {exc}")
            out.append({"symbol": symbol, "error": str(exc)})
    return out
