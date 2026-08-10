"""Monitoring & drift detection (architecture §8.2)."""

from __future__ import annotations

import json
import logging

import pandas as pd

from ..config import Config
from ..storage import Database
from .metrics import calibration_table, population_stability_index

log = logging.getLogger(__name__)


def feature_drift(
    db: Database, cfg: Config, current: pd.DataFrame, reference_frac: float = 0.6, top_n: int = 15
) -> dict[str, float]:
    """PSI of each feature: recent window vs the older reference window.

    Compares the tail of the same frame against its own head, so it works
    without needing the exact training snapshot on hand.
    """
    if current.empty or len(current) < 200:
        return {}
    split = int(len(current) * reference_frac)
    ref, cur = current.iloc[:split], current.iloc[split:]
    psi = {}
    for col in current.columns:
        if not pd.api.types.is_numeric_dtype(current[col]):
            continue
        value = population_stability_index(ref[col], cur[col])
        if value > 0:
            psi[col] = round(value, 4)
    worst = dict(sorted(psi.items(), key=lambda kv: -kv[1])[:top_n])
    drifted = {k: v for k, v in worst.items() if v > cfg.ops.drift_psi_threshold}
    if drifted:
        db.alert("warn", "drift", f"PSI above {cfg.ops.drift_psi_threshold}: {drifted}")
        log.warning("feature drift detected: %s", drifted)
    return worst


def calibration_report(db: Database) -> pd.DataFrame:
    preds = db.predictions_df()
    if preds.empty:
        return pd.DataFrame()
    scored = preds[(preds["correct"].notna()) & (preds["direction"] != "flat")]
    if scored.empty:
        return pd.DataFrame()
    return calibration_table(scored["confidence"], scored["correct"])


def health_check(db: Database, cfg: Config, ingestor=None) -> dict:
    """One call the scheduler makes every cycle; everything it finds is alerted."""
    out: dict = {"feed_gaps": [], "alerts": []}
    if ingestor is not None:
        out["feed_gaps"] = ingestor.check_feed_gaps()

    calib = calibration_report(db)
    if not calib.empty:
        out["calibration"] = json.loads(calib.to_json(orient="records"))
        worst_gap = calib["gap"].abs().max()
        if worst_gap > 0.15:
            db.alert(
                "warn",
                "calibration",
                f"confidence off by up to {worst_gap:.2f} in some bucket",
            )

    out["alerts"] = [
        {"ts": r["ts"], "level": r["level"], "source": r["source"], "message": r["message"]}
        for r in db.recent_alerts(10)
    ]
    return out
