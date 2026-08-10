"""Promotion gate (architecture §8.3).

The bar is defined in config *before* the demo run starts, so success is judged
against a pre-committed standard rather than whatever the results happen to
support afterwards. This module only reports pass/fail — it never flips
`allow_live` for you. That remains a human decision (§10, phase 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import GateConfig
from ..storage import Database
from .metrics import expected_calibration_error, trade_metrics


@dataclass
class GateCheck:
    name: str
    passed: bool
    actual: float | None
    required: float | None
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        a = "n/a" if self.actual is None else f"{self.actual:.4g}"
        r = "" if self.required is None else f" (required {self.required:.4g})"
        return f"[{mark}] {self.name}: {a}{r}{(' — ' + self.detail) if self.detail else ''}"


@dataclass
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    regime_breakdown: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def render(self) -> str:
        head = "PROMOTION GATE: " + ("PASSED" if self.passed else "NOT PASSED")
        lines = [head, "=" * len(head), ""]
        lines += [str(c) for c in self.checks]
        if self.regime_breakdown:
            lines += ["", "Per-regime directional accuracy:"]
            for regime, stats in sorted(self.regime_breakdown.items()):
                lines.append(
                    f"  {regime:<14} n={stats['n']:<5} accuracy={stats['accuracy']:.4f}"
                    f" {'ok' if stats['passes'] else 'below bar'}"
                )
        lines += [
            "",
            "This is a measurement, not a recommendation. Passing means the",
            "pre-committed bar was met on demo data — nothing more.",
        ]
        return "\n".join(lines)


def evaluate_gate(
    db: Database, cfg: GateConfig, starting_equity: float = 10_000.0, broker: str | None = None
) -> GateReport:
    report = GateReport()
    trades = db.trades_df(broker=broker)
    preds = db.predictions_df()
    scored = preds[preds["correct"].notna()] if not preds.empty else preds
    directional = (
        scored[scored["direction"] != "flat"] if not scored.empty else pd.DataFrame()
    )

    m = trade_metrics(trades, starting_equity) if not trades.empty else {"trades": 0}
    report.metrics = m

    n_trades = int(m.get("trades", 0))
    report.checks.append(
        GateCheck(
            "sample size",
            n_trades >= cfg.min_trades,
            n_trades,
            cfg.min_trades,
            "10 lucky trades is not evidence",
        )
    )

    acc = float(directional["correct"].mean()) if not directional.empty else None
    report.checks.append(
        GateCheck(
            "directional accuracy",
            acc is not None and acc >= cfg.min_directional_accuracy,
            acc,
            cfg.min_directional_accuracy,
            f"n={len(directional)}",
        )
    )

    pf = m.get("profit_factor")
    report.checks.append(
        GateCheck("profit factor", pf is not None and pf >= cfg.min_profit_factor, pf, cfg.min_profit_factor)
    )

    sh = m.get("sharpe")
    report.checks.append(
        GateCheck("sharpe", sh is not None and sh >= cfg.min_sharpe, sh, cfg.min_sharpe)
    )

    dd = m.get("max_drawdown_pct")
    report.checks.append(
        GateCheck(
            "max drawdown %",
            dd is not None and dd <= cfg.max_drawdown_pct,
            dd,
            cfg.max_drawdown_pct,
            "lower is better",
        )
    )

    ece = (
        expected_calibration_error(directional["confidence"], directional["correct"])
        if not directional.empty
        else None
    )
    ece = None if ece is None or pd.isna(ece) else float(ece)
    report.checks.append(
        GateCheck(
            "confidence calibration",
            ece is not None and ece <= cfg.max_calibration_error,
            ece,
            cfg.max_calibration_error,
            "mean |stated confidence - actual accuracy|",
        )
    )

    # Stability across regimes — an edge that only exists in one regime is not
    # a strategy, it's a bet on that regime persisting.
    passing_regimes = 0
    if not directional.empty:
        grouped = directional.groupby("regime")["correct"].agg(["mean", "count"])
        for regime, row in grouped.iterrows():
            ok = bool(row["count"] >= 30 and row["mean"] >= cfg.min_directional_accuracy)
            report.regime_breakdown[str(regime)] = {
                "accuracy": float(row["mean"]),
                "n": int(row["count"]),
                "passes": ok,
            }
            passing_regimes += int(ok)
    report.checks.append(
        GateCheck(
            "regime stability",
            passing_regimes >= cfg.min_regimes_passing,
            passing_regimes,
            cfg.min_regimes_passing,
            "regimes with n>=30 meeting the accuracy bar",
        )
    )

    db.log_run("gate", "pass" if report.passed else "fail", str(report.metrics))
    return report
