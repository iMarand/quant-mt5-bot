"""Event-driven backtester over the paper broker.

Replays base-timeframe bars in order, feeding the *same* predictor, risk manager
and trade manager the live loop uses — so a bug in the risk layer shows up here
rather than only in production. Bars are fed to the broker before a new signal
is evaluated, so SL/TP resolve on the bar they were touched.

Caveat worth stating plainly: intrabar path is unknown, spread is a constant,
and there is no slippage model beyond that. Backtest numbers are a sanity check
on the plumbing, not a forecast.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..contracts import Direction, Signal, SymbolSpec, as_utc, tf_minutes
from ..decision.execution.paper import PaperBroker
from ..decision.manager import TradeManager
from ..decision.risk import RiskManager, Veto
from ..engine.labeling import triple_barrier_labels
from ..engine.model import Ensemble, GBMModel, RuleModel, directional_confidence
from ..engine.predictor import Predictor
from ..engine.regime import classify_regime
from ..features import feature_columns
from ..storage import Database
from .metrics import trade_metrics

log = logging.getLogger(__name__)


def walk_forward_signals(
    cfg: Config,
    db: Database,
    symbol: str,
    df: pd.DataFrame,
    feats: list[str],
    warmup: int,
    retrain_every: int = 500,
) -> list[Signal]:
    """Out-of-sample signals: at every bar, only models fit on *earlier* bars.

    Replaying with the registry's active model would be in-sample — that model
    was trained on these very bars, and the resulting equity curve is fiction.
    Instead the model is refit every `retrain_every` bars on data ending
    `horizon + embargo` bars before the block starts, mirroring §7.2.
    """
    candles = db.load_candles(symbol, cfg.data.base_timeframe, limit=cfg.data.history_bars)
    labels = triple_barrier_labels(
        candles,
        horizon_bars=cfg.model.horizon_bars,
        atr_mult=cfg.model.label_atr_mult,
        atr_period=cfg.risk.atr_period,
    ).reindex(df.index)

    from ..learning.retrain import _sample_weights

    rules = RuleModel()
    X = df[feats].astype(float)
    proba = np.zeros((len(df), 3))
    proba[:warmup] = rules.predict_proba(X.iloc[:warmup])

    gap = cfg.model.horizon_bars + cfg.model.embargo_bars
    start, n_fits = warmup, 0
    while start < len(df):
        end = min(start + retrain_every, len(df))
        train_end = max(0, start - gap)
        y = labels["label"].iloc[:train_end]
        mask = y.notna()
        ens = Ensemble(rules, None)
        if mask.sum() >= cfg.model.min_train_rows and y[mask].nunique() >= 2:
            y_tr = y[mask].astype(int)
            X_tr = X.iloc[:train_end][mask.to_numpy()]
            try:
                model = GBMModel(params=cfg.model.params).fit(
                    X_tr, y_tr, sample_weight=_sample_weights(y_tr, labels.iloc[:train_end][mask])
                )
                ens = Ensemble(rules, model)
                n_fits += 1
            except Exception as exc:
                log.warning("walk-forward fit at bar %d failed: %s", start, exc)
        proba[start:end] = ens.predict_proba(X.iloc[start:end])
        start = end

    log.info("backtest %s: %d out-of-sample model fits", symbol, n_fits)

    # Hand the out-of-sample probabilities to the strategy book, so setups
    # decide and the model assists using only past-fitted models.
    from ..engine.predictor import Predictor

    signals = Predictor(cfg, db).predict_frame(symbol, df, proba=proba)
    for sig in signals:
        sig.model_version = "walkforward"
    return signals


def run_backtest(
    cfg: Config,
    db: Database,
    symbol: str,
    warmup: int = 300,
    spec: SymbolSpec | None = None,
    progress_every: int = 500,
    in_sample: bool = False,
    retrain_every: int = 500,
) -> dict:
    predictor = Predictor(cfg, db)
    df = predictor.build_features(symbol, bars=cfg.data.history_bars)
    feats = feature_columns(df)
    df = df.dropna(subset=feats, how="all")
    candles = db.load_candles(symbol, cfg.data.base_timeframe, limit=cfg.data.history_bars)
    df = df.join(candles[["high", "low", "close"]], how="inner", rsuffix="_bar")
    if len(df) <= warmup + 10:
        raise ValueError(f"only {len(df)} feature rows; need more than warmup={warmup}")

    broker = PaperBroker(
        balance=cfg.broker.paper_balance,
        spread_points=cfg.broker.paper_spread_points,
        commission_per_lot=cfg.broker.paper_commission_per_lot,
    )
    broker.register_spec(spec or SymbolSpec(symbol=symbol))
    broker.connect()

    risk = RiskManager(cfg.risk)
    manager = TradeManager(cfg.risk, broker)
    events = db.events_df()
    equity_curve: list[tuple[pd.Timestamp, float]] = []
    veto_counts: dict[str, int] = {}
    atr_col = f"{cfg.data.base_timeframe}_atr"

    if in_sample:
        log.warning(
            "in_sample=True: replaying with the registry's active model, which was "
            "trained on these bars. The result is diagnostic only, not an estimate of edge."
        )
        signals = predictor.predict_frame(symbol, df[feats])
    else:
        signals = walk_forward_signals(
            cfg, db, symbol, df, feats, warmup=warmup, retrain_every=retrain_every
        )

    day_pnl, current_day, counted = 0.0, None, 0

    for i, (ts, row) in enumerate(df.iterrows()):
        if current_day != ts.date():
            current_day, day_pnl = ts.date(), 0.0

        # 1. advance the broker's clock; this may hit stops/targets.
        broker.feed_bar(symbol, float(row["high"]), float(row["low"]), float(row["close"]), ts)
        # closed_trades is append-only and kept for the final metrics, so track
        # how many have already been folded into today's realized PnL.
        for trade in broker.closed_trades[counted:]:
            day_pnl += float(trade["profit"])
        counted = len(broker.closed_trades)

        if i < warmup:
            equity_curve.append((ts, broker.equity()))
            continue

        # 2. manage open positions on the new bar.
        atr_value = float(row[atr_col]) if atr_col in row and pd.notna(row[atr_col]) else 0.0
        manager.manage(broker.get_positions(), {symbol: atr_value})

        # 3. evaluate the signal for this bar.
        signal = signals[i]
        tick = broker.tick(symbol)
        entry = tick.ask if signal.direction is Direction.LONG else tick.bid
        decision = risk.evaluate(
            signal,
            equity=broker.equity(),
            entry_price=entry,
            atr_value=atr_value,
            spec=broker.symbol_spec(symbol),
            open_positions=broker.get_positions(),
            realized_pnl_today=day_pnl,
            spread=tick.ask - tick.bid,
            upcoming_events=events,
            now=ts,
        )
        if isinstance(decision, Veto):
            veto_counts[decision.reason] = veto_counts.get(decision.reason, 0) + 1
        else:
            fill = broker.place_order(
                symbol=symbol,
                side=decision.side,
                volume=decision.volume,
                sl=decision.sl,
                tp=decision.tp,
                comment="backtest",
            )
            if fill.status.value == "open":
                manager.register(fill.ticket, decision.risk_distance)

        equity_curve.append((ts, broker.equity()))
        if progress_every and i % progress_every == 0:
            log.info("backtest %s: bar %d/%d equity %.2f", symbol, i, len(df), broker.equity())

    # Close anything still open at the end, at the last price.
    for pos in broker.get_positions():
        broker.close_position(pos.ticket, reason="end_of_backtest")

    trades = pd.DataFrame(broker.closed_trades)
    if not trades.empty:
        trades["closed_at"] = pd.to_datetime(trades["closed_at"], utc=True)
        trades["opened_at"] = pd.to_datetime(trades["opened_at"], utc=True)

    metrics = trade_metrics(trades, cfg.broker.paper_balance) if not trades.empty else {"trades": 0}
    equity = pd.Series(dict(equity_curve))

    from ..connectors.calendar_import import calendar_coverage

    setup_counts: dict[str, int] = {}
    for sig in signals[warmup:]:
        if sig.setup:
            setup_counts[sig.setup] = setup_counts.get(sig.setup, 0) + 1

    return {
        "symbol": symbol,
        "setups": dict(sorted(setup_counts.items(), key=lambda kv: -kv[1])),
        "calendar": calendar_coverage(db, df.index[0], df.index[-1]),
        "mode": "in-sample (diagnostic)" if in_sample else "walk-forward (out-of-sample)",
        "bars": len(df),
        "from": str(df.index[0]),
        "to": str(df.index[-1]),
        "metrics": metrics,
        "vetoes": dict(sorted(veto_counts.items(), key=lambda kv: -kv[1])),
        "trades": trades,
        "equity": equity,
    }


def render_backtest(result: dict) -> str:
    m = result["metrics"]
    lines = [
        f"Backtest {result['symbol']}  {result['from']} -> {result['to']}  ({result['bars']} bars)",
        f"Mode: {result.get('mode', 'walk-forward')}",
        "-" * 72,
    ]
    if not m or m.get("trades", 0) == 0:
        lines.append("No trades taken. Check risk.min_confidence and the veto breakdown below.")
    else:
        for key in (
            "trades",
            "net_profit",
            "win_rate",
            "profit_factor",
            "expectancy",
            "sharpe",
            "sortino",
            "max_drawdown_pct",
            "final_equity",
        ):
            if key in m:
                lines.append(f"  {key:<20} {m[key]}")
    if result.get("setups"):
        lines += ["", "Setups that triggered:"]
        for name, n in result["setups"].items():
            lines.append(f"  {name:<32} {n}")

    cal = result.get("calendar") or {}
    if cal and cal.get("coverage_pct", 0) < 50:
        lines += [
            "",
            f"WARNING: the calendar covers only {cal.get('coverage_pct', 0)}% of this period "
            f"({cal.get('events', 0)} events).",
            "  Forex Factory publishes the current week only, so news setups had almost",
            "  nothing to fire on. Their absence here is a DATA gap, not evidence that",
            "  they don't work. Use `quantbot import-calendar <csv>` to backfill history.",
        ]

    if result["vetoes"]:
        lines += ["", "Vetoes (why trades were not taken):"]
        for reason, n in result["vetoes"].items():
            lines.append(f"  {reason:<24} {n}")
    lines += [
        "",
        "Costs modelled: fixed spread + per-lot commission. Intrabar path assumes",
        "the stop is hit first when a bar spans both barriers.",
    ]
    return "\n".join(lines)
