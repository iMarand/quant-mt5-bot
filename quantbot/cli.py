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

    # Per mode: a mode only ever finds a model registered on ITS OWN base
    # timeframe, so reporting one global timeframe hides the case where every
    # model exists but none is reachable by the running bot.
    print("\n-- models (per mode) --")
    from .decision.modes import build_modes

    for mode in build_modes(cfg):
        have, missing = [], []
        for symbol in cfg.data.symbols:
            row = db.active_model(symbol, mode.base_timeframe)
            if row:
                m = json.loads(row["metrics"])
                have.append(f"{symbol}({m.get('directional_accuracy')})")
            else:
                missing.append(symbol)
        print(f"  {mode.name} tf={mode.base_timeframe}:")
        if have:
            print(f"      trained: {', '.join(have)}")
        if missing:
            print(f"      NO MODEL (rules only): {', '.join(missing)}")
            print(f"      fix: python -m quantbot train --mode {mode.name}")

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


def cmd_study(args) -> int:
    cfg, db = _context(args)
    from .learning.counterfactual import (
        best_hours,
        holdout_check,
        profitable_combinations,
        run_study,
        summarize,
    )

    df = run_study(cfg, db, args.symbols or None)
    if df.empty:
        print("no setup triggers simulated — check that candles are ingested")
        return 1

    span = f"{df['ts'].min():%Y-%m-%d} -> {df['ts'].max():%Y-%m-%d}"
    print(f"\nsimulated {len(df):,} setup triggers ({span})")

    for title, keys in (
        ("by setup", ["setup"]),
        ("by setup x session", ["setup", "session"]),
        ("by session", ["session"]),
        ("by regime", ["regime"]),
    ):
        table = summarize(df, keys, min_n=args.min_n)
        print(f"\n== {title} ==")
        print(table.to_string() if not table.empty else "  (nothing with enough samples)")

    news = df[df["in_news_window"] > 0]
    if len(news) >= args.min_n:
        print("\n== inside a news window ==")
        print(summarize(news, ["setup"], min_n=args.min_n).to_string())

    if args.setup:
        print(f"\n== {args.setup}: by hour (UTC) ==")
        hours = best_hours(df, args.setup, min_n=args.min_n)
        print(hours.to_string() if not hours.empty else "  (nothing with enough samples)")

    winners = profitable_combinations(df, min_n=args.min_n)
    print("\n== combinations clearing the bar (IN-SAMPLE — selected on this data) ==")
    print(winners.to_string() if not winners.empty else "  none")

    check = holdout_check(df, min_n=args.min_n)
    if not check.empty:
        print("\n== do they hold up out-of-sample? ==")
        print(check.head(15).to_string())
        survivors = check[check["held_up"]]
        print(
            f"\n{len(survivors)}/{len(check)} combinations stayed positive after "
            f"{check['split_at'].iloc[0]}."
        )
        if survivors.empty:
            print(
                "None survived. Every combination that looked profitable in the\n"
                "earlier period lost money in the later one — which is what\n"
                "selecting the best of dozens on one dataset produces when there\n"
                "is no real edge."
            )
        else:
            print("Survivors: " + ", ".join(str(i) for i in survivors.index))

    print(
        "\nCounterfactual simulations of every trigger, including ones the live"
        "\nsystem would have skipped. Costs: spread at entry only, no slippage; a"
        "\nbar touching both barriers is scored as a loss. Anything that looks good"
        "\nhere was selected on the same data it is scored on — treat it as a"
        "\nhypothesis, not a result."
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nwrote {len(df):,} rows to {out}")

    if args.save:
        from .learning.reliability import study_stats_rows
        from .learning.selector import SetupSelector, selector_path

        rows = study_stats_rows(
            df,
            prior_strength=cfg.strategy.reliability_prior_strength * 5,
            min_samples=cfg.strategy.reliability_min_samples * 5,
            min_weight=cfg.strategy.reliability_min_weight,
            max_weight=cfg.strategy.reliability_max_weight,
        )
        n = db.save_setup_study(rows)
        print(f"\nsaved {n} setup/session weights — the runtime loads these on start")

        try:
            selector = SetupSelector()
            metrics = selector.evaluate(df)
            selector.fit(df)
            selector.metrics = metrics
            path = selector.save(selector_path(cfg))
            print(f"trained setup selector -> {path}")
            print("  validation (fit on the past, scored on the future):")
            for k, v in metrics.items():
                print(f"    {k:<22} {v}")
            print(
                "\n  decile_spread is the number that matters: win rate of the setups"
                "\n  it rated best minus those it rated worst. Near zero means it"
                "\n  cannot tell them apart, and it should stay switched off."
            )
        except Exception as exc:
            print(f"selector training failed: {exc}")
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
    from .decision.modes import build_modes
    from .learning.retrain import retrain_symbol

    # Train one model per (symbol, mode) — a mode looks its model up by ITS OWN
    # base timeframe, so a model registered under data.base_timeframe is never
    # found by a swing (H1) or scalp (M5) runtime and the bot silently runs on
    # rules alone.
    modes = build_modes(cfg)
    if args.mode:
        modes = [m for m in modes if m.name == args.mode]
        if not modes:
            print(f"unknown mode {args.mode!r}")
            return 1

    symbols = args.symbols or cfg.data.symbols
    total = len(modes) * len(symbols)
    print(f"training {total} model(s): {len(symbols)} symbol(s) x {len(modes)} mode(s)")
    print("modes: " + ", ".join(f"{m.name}(tf={m.base_timeframe})" for m in modes))

    done = 0
    for mode in modes:
        for symbol in symbols:
            done += 1
            label = f"[{done}/{total}] {mode.name}/{symbol} tf={mode.base_timeframe}"
            try:
                result = retrain_symbol(
                    mode.cfg, db, symbol, activate=not args.no_activate
                )
            except Exception as exc:
                print(f"{label}: FAILED — {exc}")
                continue
            print(
                f"{label}: dir_acc={result.get('directional_accuracy')} "
                f"rows={result.get('rows')} promoted={result.get('promoted')}"
            )
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

    from .learning.reliability import SetupReliability

    rel = SetupReliability.from_journal(
        db,
        prior_strength=cfg.strategy.reliability_prior_strength,
        min_samples=cfg.strategy.reliability_min_samples,
        min_weight=cfg.strategy.reliability_min_weight,
        max_weight=cfg.strategy.reliability_max_weight,
    )
    if rel.stats:
        print("\n== learned setup reliability ==")
        for line in rel.describe():
            print("  " + line)
        print(
            f"\nAccuracy is shrunk toward 0.5 by "
            f"{cfg.strategy.reliability_prior_strength:.0f} pseudo-observations, so a"
        )
        print("young setup reads as unproven rather than brilliant. The weight")
        print("multiplies that setup's quality when it fires.")

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

    st = sub.add_parser(
        "study", help="counterfactual: which setup would have won, when, for how many pips"
    )
    st.add_argument("symbols", nargs="*")
    st.add_argument("--min-n", type=int, default=30, help="ignore groups smaller than this")
    st.add_argument("--setup", help="also break this setup down by hour")
    st.add_argument("--out", help="write the full trigger table to CSV")
    st.add_argument(
        "--save",
        action="store_true",
        help="persist weights and train the selector, so the runtime uses them",
    )
    st.set_defaults(func=cmd_study)

    pr = sub.add_parser("predict", help="signal for the latest bar")
    pr.add_argument("symbols", nargs="*")
    pr.add_argument("--journal", action="store_true", help="record the prediction")
    pr.set_defaults(func=cmd_predict)

    t = sub.add_parser("train", help="walk-forward retrain, promote if better")
    t.add_argument("symbols", nargs="*")
    t.add_argument("--no-activate", action="store_true")
    t.add_argument("--mode", help="train only this mode (swing/scalp)")
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
