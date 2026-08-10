"""Journal resolution: attach realized outcomes to past predictions (§7.1).

Every prediction is logged the moment it is made — including the ones the risk
layer vetoed. Vetoed predictions still get outcomes, which is what makes it
possible later to ask "were my vetoes right?" — the first, cheapest form of the
counterfactual reasoning in §7.4.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from ..contracts import Direction, as_utc, utcnow
from ..storage import Database

log = logging.getLogger(__name__)


def resolve_outcomes(db: Database, timeframe: str, max_rows: int = 5000) -> int:
    """Score every prediction whose horizon has elapsed and which has candles."""
    pending = db.open_predictions()
    if not pending:
        return 0

    now = utcnow()
    resolved = 0
    candle_cache: dict[str, pd.DataFrame] = {}

    for row in pending[:max_rows]:
        ts = as_utc(pd.Timestamp(row["ts"]).to_pydatetime())
        horizon = timedelta(minutes=int(row["horizon_min"]))
        if ts + horizon > now:
            continue  # not yet decidable

        symbol = row["symbol"]
        if symbol not in candle_cache:
            candle_cache[symbol] = db.load_candles(symbol, timeframe, limit=200_000)
        candles = candle_cache[symbol]
        if candles.empty:
            continue

        entry_idx = candles.index.searchsorted(pd.Timestamp(ts))
        exit_idx = candles.index.searchsorted(pd.Timestamp(ts + horizon))
        if entry_idx >= len(candles) or exit_idx >= len(candles):
            continue  # candles for the horizon haven't been ingested yet

        entry = float(candles["close"].iloc[entry_idx])
        exit_price = float(candles["close"].iloc[exit_idx])
        if entry == 0:
            continue

        realized = (exit_price - entry) / entry
        # Label with a dead-band so noise isn't scored as a directional call.
        band = _dead_band(candles, entry_idx)
        label = 1 if realized > band else (-1 if realized < -band else 0)

        direction = Direction(row["direction"])
        correct = bool(direction.sign != 0 and label == direction.sign)

        db.record_outcome(
            prediction_id=int(row["id"]),
            realized_return=realized,
            label=label,
            correct=correct,
            realized_r=realized * direction.sign / band if band else None,
        )
        resolved += 1

    if resolved:
        log.info("resolved %d prediction outcomes", resolved)
        db.log_run("resolve_outcomes", "ok", f"{resolved} resolved")
    return resolved


def _dead_band(candles: pd.DataFrame, idx: int, lookback: int = 50) -> float:
    """Half an average bar range, as a fraction of price — 'meaningfully moved'."""
    lo = max(0, idx - lookback)
    window = candles.iloc[lo : idx + 1]
    if window.empty:
        return 0.0002
    rng = (window["high"] - window["low"]).mean()
    price = float(window["close"].iloc[-1]) or 1.0
    return max(float(rng) / price * 0.5, 1e-5)


def journal_summary(db: Database) -> dict:
    df = db.predictions_df()
    if df.empty:
        return {"predictions": 0}
    scored = df[df["correct"].notna()]
    out = {
        "predictions": int(len(df)),
        "scored": int(len(scored)),
        "acted_on": int(df["acted_on"].sum()),
        "vetoed": int((df["veto_reason"].notna()).sum()),
    }
    if not scored.empty:
        directional = scored[scored["direction"] != "flat"]
        out["by_setup"] = setup_performance(db)
        out["directional_accuracy"] = (
            round(float(directional["correct"].mean()), 4) if not directional.empty else None
        )
        out["mean_confidence"] = round(float(scored["confidence"].mean()), 4)
        by_regime = directional.groupby("regime")["correct"].agg(["mean", "count"])
        out["by_regime"] = {
            k: {"accuracy": round(float(v["mean"]), 4), "n": int(v["count"])}
            for k, v in by_regime.iterrows()
        }
    return out


def setup_performance(db: Database) -> dict:
    """Per-setup hit rate — which named patterns actually work (§7.3).

    This is the table that tells you to disable a setup, and the signal the
    model uses to learn which patterns to trust.
    """
    df = db.predictions_df()
    if df.empty or "setup" not in df.columns:
        return {}
    scored = df[(df["correct"].notna()) & (df["direction"] != "flat")]
    scored = scored[scored["setup"].fillna("") != ""]
    if scored.empty:
        return {}

    out: dict[str, dict] = {}
    for name, group in scored.groupby("setup"):
        out[str(name)] = {
            "n": int(len(group)),
            "accuracy": round(float(group["correct"].mean()), 4),
            "mean_confidence": round(float(group["confidence"].mean()), 4),
            "acted_on": int(group["acted_on"].sum()),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def setup_trade_performance(db: Database) -> pd.DataFrame:
    """Realized PnL grouped by the setup that opened each trade."""
    df = db.query_df(
        """SELECT p.setup AS setup, t.profit AS profit, t.r_multiple AS r
           FROM trades t JOIN predictions p ON p.id = t.prediction_id
           WHERE p.setup IS NOT NULL AND p.setup != ''"""
    )
    if df.empty:
        return df
    grouped = df.groupby("setup")["profit"].agg(
        trades="size", net="sum", win_rate=lambda s: float((s > 0).mean())
    )
    gains = df[df["profit"] > 0].groupby("setup")["profit"].sum()
    losses = -df[df["profit"] < 0].groupby("setup")["profit"].sum()
    grouped["profit_factor"] = (gains / losses.replace(0, float("nan"))).round(3)
    return grouped.sort_values("net", ascending=False).round(4)


def session_performance(db: Database) -> pd.DataFrame:
    """Realized PnL grouped by session and mode.

    With sessions auto-derived per pair the bot trades round the clock; this is
    how you find out whether the thin Asian hours are actually paying for their
    wider spreads, rather than assuming either way.
    """
    df = db.query_df(
        """SELECT COALESCE(p.session,'?') AS session, COALESCE(p.mode,'?') AS mode,
                  t.profit AS profit
           FROM trades t JOIN predictions p ON p.id = t.prediction_id"""
    )
    if df.empty:
        return df
    grouped = df.groupby(["session", "mode"])["profit"].agg(
        trades="size", net="sum", win_rate=lambda s: float((s > 0).mean())
    )
    gains = df[df["profit"] > 0].groupby(["session", "mode"])["profit"].sum()
    losses = -df[df["profit"] < 0].groupby(["session", "mode"])["profit"].sum()
    grouped["profit_factor"] = (gains / losses.replace(0, float("nan"))).round(3)
    return grouped.sort_values("net", ascending=False).round(4)
