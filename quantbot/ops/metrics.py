"""Performance and calibration metrics (architecture §8.2, §8.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(r) < 2 or downside.std(ddof=1) in (0, np.nan) or downside.empty:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Returns (absolute drawdown, drawdown as % of the running peak)."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = equity - peak
    pct = (dd / peak.replace(0, np.nan)) * 100
    return float(dd.min()), float(pct.min() if pct.notna().any() else 0.0)


def profit_factor(profits: pd.Series) -> float:
    gains = profits[profits > 0].sum()
    losses = -profits[profits < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def expectancy(profits: pd.Series) -> float:
    return float(profits.mean()) if len(profits) else 0.0


def win_rate(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    return float((profits > 0).mean())


def brier_score(confidence: pd.Series, correct: pd.Series) -> float:
    """Lower is better; 0.25 is what you get by always saying 50%."""
    c = confidence.astype(float)
    y = correct.astype(float)
    mask = c.notna() & y.notna()
    if mask.sum() == 0:
        return float("nan")
    return float(((c[mask] - y[mask]) ** 2).mean())


def calibration_table(
    confidence: pd.Series, correct: pd.Series, bins: int = 5
) -> pd.DataFrame:
    """When the model says 70%, is it right ~70% of the time? (§8.2)"""
    df = pd.DataFrame({"conf": confidence.astype(float), "hit": correct.astype(float)}).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "n", "mean_confidence", "actual_accuracy", "gap"])
    df["bin"] = pd.cut(df["conf"], bins=np.linspace(0.5, 1.0, bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        n=("hit", "size"), mean_confidence=("conf", "mean"), actual_accuracy=("hit", "mean")
    )
    grouped["gap"] = grouped["actual_accuracy"] - grouped["mean_confidence"]
    return grouped.reset_index()


def expected_calibration_error(confidence: pd.Series, correct: pd.Series, bins: int = 5) -> float:
    table = calibration_table(confidence, correct, bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    """Feature-drift measure (§8.2). >0.25 conventionally means "materially shifted"."""
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def trade_metrics(trades: pd.DataFrame, starting_equity: float = 10_000.0) -> dict:
    """Full risk-adjusted summary of a closed-trade table."""
    if trades.empty:
        return {"trades": 0}
    profits = trades["profit"].astype(float)
    equity = starting_equity + profits.cumsum()
    dd_abs, dd_pct = max_drawdown(equity)
    # Trades aren't daily observations; annualize by the observed trade rate.
    span_days = max(
        (trades["closed_at"].max() - trades["closed_at"].min()).total_seconds() / 86400, 1.0
    )
    trades_per_year = len(trades) / span_days * 365
    return {
        "trades": int(len(trades)),
        "net_profit": round(float(profits.sum()), 2),
        "win_rate": round(win_rate(profits), 4),
        "profit_factor": round(profit_factor(profits), 3),
        "expectancy": round(expectancy(profits), 4),
        "sharpe": round(sharpe(profits / starting_equity, int(max(trades_per_year, 1))), 3),
        "sortino": round(sortino(profits / starting_equity, int(max(trades_per_year, 1))), 3),
        "max_drawdown": round(dd_abs, 2),
        "max_drawdown_pct": round(abs(dd_pct), 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "span_days": round(span_days, 1),
    }
