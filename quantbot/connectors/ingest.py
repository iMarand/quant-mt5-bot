"""Normalizer / deduplicator: the one place all feeds converge (§3.4)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from ..config import ROOT, Config
from ..contracts import Candle, tf_minutes, utcnow
from ..storage import Database
from .forexfactory import ForexFactoryCalendar
from .investing import InvestingCalendar, cross_check
from .policy import ComplianceError, FetchPolicy

log = logging.getLogger(__name__)


class Ingestor:
    #: Floor per timeframe, so a 200-period indicator still has warm-up room.
    min_bars_per_timeframe = 1500

    def __init__(self, cfg: Config, db: Database, market) -> None:
        self.cfg = cfg
        self.db = db
        self.market = market
        self.policy = FetchPolicy(
            user_agent=cfg.calendar.user_agent,
            min_interval_s=cfg.calendar.request_min_interval_s,
            cache_dir=ROOT / "artifacts" / "cache",
            cache_ttl_s=cfg.calendar.cache_minutes * 60,
        )
        self.ff = ForexFactoryCalendar(
            url=cfg.calendar.forexfactory_url,
            policy=self.policy,
            currencies=cfg.calendar.currencies,
            cache_minutes=cfg.calendar.cache_minutes,
        )
        self.investing = InvestingCalendar(
            policy=self.policy,
            api_key=cfg.calendar.investing_api_key,
            currencies=cfg.calendar.currencies,
            enabled=cfg.calendar.investing_enabled,
        )

    # -- market data -------------------------------------------------------
    def ingest_market(self, count: int | None = None, symbols: list[str] | None = None) -> int:
        """Parallel multi-timeframe pull (§3.3), deduplicated by the DB's PK."""
        symbols = symbols or (self.cfg.data.symbols + self.cfg.data.correlated_symbols)
        count = count or self.cfg.data.history_bars
        jobs = [(s, tf) for s in symbols for tf in self.cfg.data.timeframes]
        total = 0

        # `count` is expressed in base-timeframe bars. Requesting the same
        # number on every timeframe asks for absurd spans (20k D1 bars is 55
        # years), which makes the terminal download history nobody needs and
        # stalls the whole ingest. Scale by timeframe, with a floor so higher
        # timeframes still have enough bars for a 200-period indicator.
        base_min = tf_minutes(self.cfg.data.base_timeframe)
        counts = {
            tf: max(self.min_bars_per_timeframe, int(count * base_min / tf_minutes(tf)))
            for tf in self.cfg.data.timeframes
        }

        # Establish the connection once. Without this, a dead terminal produces
        # one full failure (and one full remedy hint) per symbol/timeframe.
        try:
            self.market.connect()
        except Exception as exc:
            log.error("market data unavailable, skipping %d series: %s", len(jobs), exc)
            self.db.alert("error", "ingest", f"market data unavailable: {exc}")
            self.db.log_run("ingest_market", "error", str(exc))
            return 0

        def pull(job: tuple[str, str]) -> list[Candle]:
            symbol, tf = job
            try:
                return self.market.fetch_candles(symbol, tf, count=counts[tf])
            except Exception as exc:
                log.error("fetch failed %s %s: %s", symbol, tf, exc)
                self.db.alert("error", "ingest", f"{symbol} {tf}: {exc}")
                return []

        # MT5's Python bridge serializes calls internally; a small pool still
        # overlaps the request/response round-trips without hammering it.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for candles in pool.map(pull, jobs):
                total += self.db.upsert_candles(candles)
        self.db.log_run("ingest_market", "ok", f"{total} candles across {len(jobs)} series")
        log.info("ingested %d candles (%d series)", total, len(jobs))
        return total

    # -- calendar ----------------------------------------------------------
    def ingest_calendar(self, force: bool = False) -> int:
        if not self.cfg.calendar.enabled:
            return 0
        try:
            events = self.ff.fetch_events(force=force)
        except ComplianceError as exc:
            self.db.alert("error", "calendar", str(exc))
            log.error("calendar blocked by policy: %s", exc)
            return 0
        except Exception as exc:
            self.db.alert("error", "calendar", f"forexfactory fetch failed: {exc}")
            log.error("calendar fetch failed: %s", exc)
            return 0

        n = self.db.upsert_events(events)

        try:
            secondary = self.investing.fetch_events(force=force)
        except ComplianceError as exc:
            log.info("secondary calendar unavailable: %s", exc)
            secondary = []
        except Exception as exc:
            log.warning("secondary calendar failed: %s", exc)
            secondary = []

        if secondary:
            self.db.upsert_events(secondary)
            for issue in cross_check(events, secondary):
                self.db.alert("warn", "calendar_crosscheck", str(issue))

        self.db.log_run("ingest_calendar", "ok", f"{n} events")
        log.info("ingested %d calendar events", n)
        return n

    # -- health ------------------------------------------------------------
    def check_feed_gaps(self) -> list[str]:
        """A silently broken connector is worse than a loudly broken one (§8.2)."""
        problems: list[str] = []
        now = utcnow()
        for symbol in self.cfg.data.symbols:
            for tf in self.cfg.data.timeframes:
                last = self.db.last_candle_ts(symbol, tf)
                if last is None:
                    problems.append(f"{symbol} {tf}: no data at all")
                    continue
                tolerance = timedelta(
                    minutes=tf_minutes(tf) * (self.cfg.ops.feed_gap_tolerance_bars + 1)
                )
                gap = now - last
                # Weekends are expected gaps in FX, not failures.
                if gap > tolerance and now.weekday() < 5:
                    problems.append(f"{symbol} {tf}: stale by {gap}")
        for p in problems:
            self.db.alert("warn", "feed_gap", p)
        return problems
