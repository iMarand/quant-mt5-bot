"""Command-line entry point: `python -m quantbot <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config, load_config
from .storage import Database


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _context(args) -> tuple[Config, Database]:
    cfg = load_config(args.config)
    if getattr(args, "broker", None):
        cfg.broker.kind = args.broker
    if getattr(args, "live", False):
        cfg.dry_run = False
    _setup_logging(getattr(args, "log_level", None) or cfg.ops.log_level)
    return cfg, Database(cfg.db_file)


def _market_connector(cfg: Config):
    """MT5 for real data; the paper broker has no data feed of its own."""
    from .connectors.mt5_market import MT5MarketData

    return MT5MarketData(
        login=cfg.broker.mt5_login,
        password=cfg.broker.mt5_password,
        server=cfg.broker.mt5_server,
        path=cfg.broker.mt5_path,
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_doctor(args) -> int:
    cfg, db = _context(args)
    print(f"config      : {args.config or 'config.yaml'}")
    print(f"database    : {cfg.db_file}")
    print(f"symbols     : {', '.join(cfg.data.symbols)}  base tf {cfg.data.base_timeframe}")
    print(f"broker      : {cfg.broker.kind}  dry_run={cfg.dry_run}  allow_live={cfg.broker.allow_live}")

    print("\n-- data --")
    for symbol in cfg.data.symbols:
        for tf in cfg.data.timeframes:
            last = db.last_candle_ts(symbol, tf)
            n = db.query(
                "SELECT COUNT(*) c FROM candles WHERE symbol=? AND timeframe=?", (symbol, tf)
            )[0]["c"]
            print(f"  {symbol} {tf:<4} {n:>7} bars  last={last}")
    n_events = db.query("SELECT COUNT(*) c FROM calendar_events")[0]["c"]
    print(f"  calendar events: {n_events}")

    print("\n-- models --")
    for symbol in cfg.data.symbols:
        row = db.active_model(symbol, cfg.data.base_timeframe)
        if row:
            m = json.loads(row["metrics"])
            print(
                f"  {symbol}: {row['version']} score={m.get('score')} "
                f"dir_acc={m.get('directional_accuracy')}"
            )
        else:
            print(f"  {symbol}: no trained model — rules baseline in use")

    print("\n-- MT5 terminal --")
    try:
        market = _market_connector(cfg)
        market.connect()
        info = market.account_info()
        print(
            f"  connected: login={info.get('login')} server={info.get('server')} "
            f"balance={info.get('balance')} {info.get('currency')} "
            f"mode={'DEMO' if market.account_is_demo() else 'REAL'}"
        )
        market.disconnect()
    except Exception as exc:
        print(f"  NOT AVAILABLE: {exc}")
        # The error code alone isn't diagnostic; the terminal's log is.
        from .connectors.mt5_diag import diagnose

        diag = diagnose()
        if diag.log_file:
            print(f"\n  terminal log: {diag.log_file}")
            if diag.build:
                print(f"  build {diag.build}, algo trading: {diag.algo_trading}")
            for step in diag.explain():
                print(f"  -> {step}")

    print("\n-- recent alerts --")
    from .storage.db import summarize

    rows = db.recent_alerts(5)
    if not rows:
        print("  (none)")
    for row in rows:
        # summarize again on read: rows written before alerts were sanitized
        # still hold multi-line text.
        ts = str(row["ts"])[:19].replace("T", " ")
        print(f"  {ts} [{row['level']}] {row['source']}: {summarize(row['message'], 160)}")
    return 0


def cmd_ingest(args) -> int:
    cfg, db = _context(args)
    from .connectors.ingest import Ingestor

    market = _market_connector(cfg)
    ing = Ingestor(cfg, db, market)
    if not args.no_calendar:
        n = ing.ingest_calendar(force=args.force)
        print(f"calendar: {n} events")
    if not args.no_market:
        market.connect()
        n = ing.ingest_market(count=args.bars)
        print(f"market  : {n} candles")
        market.disconnect()
    for problem in ing.check_feed_gaps():
        print(f"WARN {problem}")
    return 0


def cmd_import(args) -> int:
    cfg, db = _context(args)
    from .connectors.csv_import import import_csv

    paths: list[Path] = []
    for pattern in args.paths:
        p = Path(pattern)
        paths.extend(sorted(p.parent.glob(p.name)) if any(c in pattern for c in "*?") else [p])
    if not paths:
        print("no files matched")
        return 1

    total = 0
    for path in paths:
        try:
            n, tf = import_csv(
                db,
                path,
                symbol=args.symbol,
                timeframe=args.timeframe,
                tz_offset_hours=args.tz_offset,
            )
        except Exception as exc:
            print(f"{path.name}: FAILED — {exc}")
            continue
        total += n
        print(f"{path.name}: {n} bars as {args.symbol} {tf}")
    print(f"\nimported {total} bars total")
    if args.tz_offset == 0:
        print(
            "NOTE: --tz-offset was 0. MT5 exports use broker SERVER time; if your\n"
            "      server is not UTC, re-import with the correct offset or every\n"
            "      news-timing feature will be shifted."
        )
    return 0


def cmd_import_calendar(args) -> int:
    cfg, db = _context(args)
    from .connectors.calendar_import import calendar_coverage, import_calendar

    total = 0
    for pattern in args.paths:
        path = Path(pattern)
        for f in (sorted(path.parent.glob(path.name)) if any(c in pattern for c in "*?") else [path]):
            try:
                n = import_calendar(db, f, tz_offset_hours=args.tz_offset)
            except Exception as exc:
                print(f"{f.name}: FAILED — {exc}")
                continue
            total += n
            print(f"{f.name}: {n} events")
    cov = calendar_coverage(db)
    print(
        f"\ncalendar now holds {cov.get('events', 0)} events "
        f"({cov.get('high_impact', 0)} high-impact)"
    )
    if cov.get("events"):
        print(f"spanning {cov.get('from')} -> {cov.get('to')}")
    return 0


def cmd_predict(args) -> int:
    cfg, db = _context(args)
    from .engine.predictor import Predictor

    predictor = Predictor(cfg, db)
    for symbol in args.symbols or cfg.data.symbols:
        try:
            signal = predictor.predict_latest(symbol)
        except Exception as exc:
            print(f"{symbol}: {exc}")
            continue
        print(
            f"{symbol} @ {signal.ts:%Y-%m-%d %H:%M} UTC\n"
            f"  direction  : {signal.direction.value}\n"
            f"  confidence : {signal.confidence:.4f}\n"
            f"  setup      : {signal.setup or '(none triggered — no trade)'}\n"
            f"  why        : {signal.rationale or '-'}\n"
            f"  regime     : {signal.regime.value}\n"
            f"  horizon    : {signal.horizon_min} min\n"
            f"  model      : {signal.model_version}"
        )
        print("  detail     :")
        for k, v in sorted(
            signal.driving_features.items(), key=lambda kv: -abs(kv[1])
        )[:10]:
            print(f"    {k:<34} {v:+.5f}")
        if args.journal:
            pid = db.record_prediction(signal)
            print(f"  journaled as prediction #{pid}")
    return 0


def cmd_train(args) -> int:
    cfg, db = _context(args)
    from .learning.retrain import retrain_symbol

    for symbol in args.symbols or cfg.data.symbols:
        try:
            result = retrain_symbol(cfg, db, symbol, activate=not args.no_activate)
        except Exception as exc:
            print(f"{symbol}: FAILED — {exc}")
            continue
        print(json.dumps(result, indent=2))
    return 0


def cmd_search(args) -> int:
    cfg, db = _context(args)
    from .learning.retrain import retrain_symbol
    from .learning.search import EvolutionarySearch, apply_strategy_genes

    for symbol in args.symbols or cfg.data.symbols:
        search = EvolutionarySearch(
            cfg, db, population=args.population, generations=args.generations
        )
        best = search.run(symbol)
        print(f"\n{symbol}: best score {best.score:.5f}")
        print(json.dumps(best.as_dict(), indent=2))
        if args.apply:
            apply_strategy_genes(cfg, best.strategy)
            result = retrain_symbol(cfg, db, symbol, params=best.model, activate=True)
            print(f"retrained with evolved structure: promoted={result['promoted']}")
            print(
                "NOTE: evolved strategy genes were applied in-memory only. "
                "Copy them into config.yaml to persist."
            )
    return 0


def cmd_backtest(args) -> int:
    cfg, db = _context(args)
    from .ops.backtest import render_backtest, run_backtest

    for symbol in args.symbols or cfg.data.symbols:
        try:
            result = run_backtest(
                cfg,
                db,
                symbol,
                warmup=args.warmup,
                in_sample=args.in_sample,
                retrain_every=args.retrain_every,
            )
        except Exception as exc:
            print(f"{symbol}: {exc}")
            continue
        print(render_backtest(result))
        if args.out:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            result["trades"].to_csv(out / f"{symbol}_trades.csv", index=False)
            result["equity"].to_csv(out / f"{symbol}_equity.csv")
            print(f"\nwrote {out}/{symbol}_trades.csv and _equity.csv")
    return 0


def cmd_run(args) -> int:
    cfg, db = _context(args)
    from .connectors.ingest import Ingestor
    from .decision.execution import make_broker
    from .ops.runner import Runner
    from .ops.scheduler import Scheduler

    market = _market_connector(cfg)
    broker = make_broker(cfg)
    broker.connect()
    ingestor = Ingestor(cfg, db, market)
    runner = Runner(cfg, db, broker, ingestor)

    blocking, warnings = runner.preflight()
    for w in warnings:
        print(f"WARNING: {w}")
    if blocking and not args.force:
        print("cannot start — preflight failed:")
        for p in blocking:
            print(f"  - {p}")
        print(
            "\nFix these first. Most likely:\n"
            "  python -m quantbot doctor    # diagnoses the MT5 connection\n"
            "  python -m quantbot ingest    # populates candles once MT5 connects\n"
            "\nTo exercise the loop without real data:\n"
            "  python tools/seed_synthetic.py --config config.sandbox.yaml\n"
            "  python -m quantbot --config config.sandbox.yaml run --once --live\n"
            "\nPass --force to start anyway."
        )
        broker.disconnect()
        market.disconnect()
        return 1

    mode = "DRY RUN (no orders)" if cfg.dry_run else f"LIVE ORDERS on {broker.name}"
    print(f"starting: {mode}, broker demo={getattr(broker, 'is_demo', True)}")

    try:
        if args.once:
            result = Scheduler(cfg, db, runner).run_once("manual")
            print(result.render())
        else:
            Scheduler(cfg, db, runner).run_forever(max_cycles=args.max_cycles)
    finally:
        broker.disconnect()
        market.disconnect()
    return 0


def cmd_report(args) -> int:
    cfg, db = _context(args)
    from .learning.journal import journal_summary
    from .ops.metrics import trade_metrics
    from .ops.monitor import calibration_report

    print("== journal ==")
    print(json.dumps(journal_summary(db), indent=2))

    trades = db.trades_df()
    print("\n== trades ==")
    print(json.dumps(trade_metrics(trades, cfg.broker.paper_balance), indent=2, default=str))

    from .learning.journal import session_performance, setup_trade_performance

    by_setup = setup_trade_performance(db)
    if not by_setup.empty:
        print("\n== realized PnL by setup ==")
        print(by_setup.to_string())
        print("\nA setup with a losing profit factor over a decent sample is a")
        print("candidate to disable in config.yaml under strategy.setups.")

    by_session = session_performance(db)
    if not by_session.empty:
        print("\n== realized PnL by session / mode ==")
        print(by_session.to_string())
        print("\nThin sessions pay wider spreads. If a session's profit factor")
        print("stays below 1 over a decent sample, restrict it in modes.*.sessions.")

    calib = calibration_report(db)
    if not calib.empty:
        print("\n== confidence calibration ==")
        print(calib.to_string(index=False))
    return 0


def cmd_gate(args) -> int:
    cfg, db = _context(args)
    from .ops.gate import evaluate_gate

    report = evaluate_gate(db, cfg.gate, starting_equity=cfg.broker.paper_balance)
    print(report.render())
    return 0 if report.passed else 1


def cmd_resolve(args) -> int:
    cfg, db = _context(args)
    from .learning.journal import resolve_outcomes

    n = resolve_outcomes(db, cfg.data.base_timeframe)
    print(f"resolved {n} predictions")
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantbot", description=__doc__)
    p.add_argument("--config", help="path to config.yaml")
    p.add_argument("--log-level", help="DEBUG/INFO/WARNING")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="environment, data and model status")
    d.set_defaults(func=cmd_doctor)

    i = sub.add_parser("ingest", help="pull calendar + market data")
    i.add_argument("--bars", type=int, default=None)
    i.add_argument("--force", action="store_true", help="bypass the calendar cache")
    i.add_argument("--no-calendar", action="store_true")
    i.add_argument("--no-market", action="store_true")
    i.set_defaults(func=cmd_ingest)

    imp = sub.add_parser("import-csv", help="load OHLCV history from MT5/generic CSV")
    imp.add_argument("paths", nargs="+", help="csv files (globs allowed)")
    imp.add_argument("--symbol", required=True)
    imp.add_argument("--timeframe", help="M1/M5/M15/M30/H1/H4/D1 (inferred if omitted)")
    imp.add_argument(
        "--tz-offset",
        type=float,
        default=0.0,
        help="broker server UTC offset in hours (e.g. 3 for GMT+3)",
    )
    imp.set_defaults(func=cmd_import)

    ic = sub.add_parser("import-calendar", help="backfill historical calendar from CSV")
    ic.add_argument("paths", nargs="+", help="csv files (globs allowed)")
    ic.add_argument("--tz-offset", type=float, default=0.0, help="source UTC offset in hours")
    ic.set_defaults(func=cmd_import_calendar)

    pr = sub.add_parser("predict", help="signal for the latest bar")
    pr.add_argument("symbols", nargs="*")
    pr.add_argument("--journal", action="store_true", help="record the prediction")
    pr.set_defaults(func=cmd_predict)

    t = sub.add_parser("train", help="walk-forward retrain, promote if better")
    t.add_argument("symbols", nargs="*")
    t.add_argument("--no-activate", action="store_true")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("search", help="evolutionary structure/strategy search")
    s.add_argument("symbols", nargs="*")
    s.add_argument("--population", type=int, default=6)
    s.add_argument("--generations", type=int, default=3)
    s.add_argument("--apply", action="store_true", help="retrain with the winning structure")
    s.set_defaults(func=cmd_search)

    b = sub.add_parser("backtest", help="replay the full loop over stored bars")
    b.add_argument("symbols", nargs="*")
    b.add_argument("--warmup", type=int, default=300)
    b.add_argument("--retrain-every", type=int, default=500, help="walk-forward refit interval")
    b.add_argument(
        "--in-sample",
        action="store_true",
        help="replay with the active model (leaks; diagnostic only)",
    )
    b.add_argument("--out", help="directory for trades/equity CSVs")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("run", help="run the scheduled trading loop")
    r.add_argument("--once", action="store_true", help="single cycle then exit")
    r.add_argument("--max-cycles", type=int, default=None)
    r.add_argument("--broker", choices=["paper", "mt5"])
    r.add_argument("--force", action="store_true", help="start even if preflight fails")
    r.add_argument(
        "--live",
        action="store_true",
        help="actually send orders (still demo-only unless broker.allow_live)",
    )
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="journal + trade + calibration summary")
    rep.set_defaults(func=cmd_report)

    g = sub.add_parser("gate", help="evaluate the promotion gate")
    g.set_defaults(func=cmd_gate)

    rs = sub.add_parser("resolve", help="score elapsed predictions")
    rs.set_defaults(func=cmd_resolve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
