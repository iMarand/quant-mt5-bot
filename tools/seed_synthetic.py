"""Seed the database with synthetic candles + calendar events.

Lets the full pipeline (features -> train -> backtest -> gate) be exercised
without a broker connection. The generator embeds a *weak, real* structure —
momentum plus a news-reaction effect — so a working model should score slightly
above chance. It is a plumbing test, not a strategy validation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantbot.config import load_config  # noqa: E402
from quantbot.contracts import Candle, CalendarEvent, Impact, tf_minutes  # noqa: E402
from quantbot.storage import Database  # noqa: E402


def generate(symbol: str, timeframe: str, n: int, seed: int, end: datetime) -> list[Candle]:
    rng = np.random.default_rng(seed + len(timeframe) + hash(symbol) % 1000)
    minutes = tf_minutes(timeframe)
    price = 1.1000
    vol = 0.00035 * np.sqrt(minutes / 15)
    momentum = 0.0
    candles: list[Candle] = []

    for i in range(n):
        ts = end - timedelta(minutes=minutes * (n - i))
        # AR(1) momentum term -> a faint, learnable trend signal.
        momentum = 0.75 * momentum + rng.normal(0, vol * 0.45)
        step = momentum + rng.normal(0, vol)
        # Session volatility: London/NY are livelier than Asia.
        hour = ts.hour
        step *= 1.5 if 7 <= hour < 20 else 0.6

        open_ = price
        close = max(0.5, open_ + step)
        wick = abs(rng.normal(0, vol * 0.6))
        high = max(open_, close) + wick
        low = min(open_, close) - abs(rng.normal(0, vol * 0.6))
        candles.append(
            Candle(
                symbol=symbol,
                timeframe=timeframe,
                ts=ts.replace(second=0, microsecond=0, tzinfo=timezone.utc),
                open=round(open_, 5),
                high=round(high, 5),
                low=round(low, 5),
                close=round(close, 5),
                volume=float(rng.integers(200, 3000)),
                spread=12.0,
            )
        )
        price = close
    return candles


def generate_events(end: datetime, weeks: int, seed: int) -> list[CalendarEvent]:
    rng = np.random.default_rng(seed)
    names = [
        ("USD", "Non-Farm Employment Change", Impact.HIGH),
        ("USD", "CPI m/m", Impact.HIGH),
        ("USD", "Unemployment Claims", Impact.MEDIUM),
        ("EUR", "ECB Press Conference", Impact.HIGH),
        ("EUR", "German Flash Manufacturing PMI", Impact.MEDIUM),
        ("EUR", "Retail Sales m/m", Impact.LOW),
    ]
    out: list[CalendarEvent] = []
    for week in range(weeks):
        base = end - timedelta(weeks=weeks - week)
        for i, (ccy, name, impact) in enumerate(names):
            ts = (base + timedelta(days=i % 5, hours=12, minutes=30)).replace(
                second=0, microsecond=0, tzinfo=timezone.utc
            )
            forecast = float(rng.normal(0.2, 0.1))
            actual = forecast + float(rng.normal(0, 0.12))
            out.append(
                CalendarEvent(
                    event_id=f"synth-{week}-{i}",
                    source="synthetic",
                    currency=ccy,
                    name=name,
                    ts_utc=ts,
                    impact=impact,
                    forecast=round(forecast, 4),
                    previous=round(forecast - 0.05, 4),
                    actual=round(actual, 4) if ts < end else None,
                )
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--bars", type=int, default=6000, help="bars on the base timeframe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--append",
        action="store_true",
        help="keep existing candles (only do this if you know the grids align)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    db = Database(cfg.db_file)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    base_min = tf_minutes(cfg.data.base_timeframe)

    # Each run generates an independent random walk anchored to "now", so its
    # bar timestamps land on a different grid than a previous run's. Appending
    # interleaves two unrelated series into a sawtooth that looks wildly
    # predictable to a model and produces a nonsense ATR. Wipe by default.
    if not args.append:
        with db.connect() as conn:
            for symbol in cfg.data.symbols:
                cur = conn.execute("DELETE FROM candles WHERE symbol=?", (symbol,))
                if cur.rowcount > 0:
                    print(f"cleared {cur.rowcount} existing {symbol} candles")
            conn.execute("DELETE FROM calendar_events WHERE source='synthetic'")

    total = 0
    for symbol in cfg.data.symbols:
        for tf in cfg.data.timeframes:
            n = max(300, int(args.bars * base_min / tf_minutes(tf)))
            total += db.upsert_candles(generate(symbol, tf, n, args.seed, end))
    weeks = max(4, int(args.bars * base_min / (60 * 24 * 7)) + 1)
    n_events = db.upsert_events(generate_events(end, weeks, args.seed))

    print(f"seeded {total} synthetic candles and {n_events} events into {cfg.db_file}")
    print("NOTE: synthetic data. Any metrics from it describe the code, not the market.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
