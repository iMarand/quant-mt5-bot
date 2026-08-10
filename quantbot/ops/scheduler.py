"""Scheduler (architecture §8.1).

Base cadence plus event-triggered passes a few minutes either side of every
high-impact release — the moments the multi-timeframe analysis is most likely to
matter. Deliberately a single-process loop: Airflow/Prefect are the right answer
at scale, but adding an orchestrator before the pipeline is proven is cost with
no benefit.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from ..config import Config
from ..contracts import utcnow
from ..features.events import next_high_impact_events
from ..storage import Database
from .runner import CycleResult, Runner

log = logging.getLogger(__name__)


@dataclass
class ScheduledPass:
    at: datetime
    reason: str


class Scheduler:
    def __init__(self, cfg: Config, db: Database, runner: Runner) -> None:
        self.cfg = cfg
        self.db = db
        self.runner = runner
        self._stop = False
        self._fired: set[str] = set()

    def request_stop(self, *_: object) -> None:
        log.info("stop requested; finishing current cycle")
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):  # not on the main thread
                pass

    # -- event-triggered passes -------------------------------------------
    def event_passes(self, horizon_min: int = 90) -> list[ScheduledPass]:
        events = self.db.events_df()
        if events.empty:
            return []
        now = pd.Timestamp(utcnow())
        upcoming = next_high_impact_events(events, now, within_min=horizon_min)
        out: list[ScheduledPass] = []
        for _, ev in upcoming.iterrows():
            key_pre = f"pre:{ev['event_id']}"
            key_post = f"post:{ev['event_id']}"
            pre_at = ev["ts_utc"] - pd.Timedelta(minutes=self.cfg.ops.event_lookahead_minutes)
            post_at = ev["ts_utc"] + pd.Timedelta(minutes=self.cfg.ops.event_followup_minutes)
            if key_pre not in self._fired and pre_at > now:
                out.append(ScheduledPass(pre_at.to_pydatetime(), key_pre))
            if key_post not in self._fired and post_at > now:
                out.append(ScheduledPass(post_at.to_pydatetime(), key_post))
        return sorted(out, key=lambda p: p.at)

    def _next_wake(self, last_run: datetime) -> tuple[datetime, str]:
        base_next = last_run + timedelta(minutes=self.cfg.ops.interval_minutes)
        candidates = [ScheduledPass(base_next, "interval")] + self.event_passes()
        chosen = min(candidates, key=lambda p: p.at)
        return chosen.at, chosen.reason

    # -- main loop ---------------------------------------------------------
    def run_forever(self, max_cycles: int | None = None) -> None:
        self.install_signal_handlers()
        cycles = 0
        last_run = utcnow() - timedelta(minutes=self.cfg.ops.interval_minutes)
        log.info(
            "scheduler started: every %d min, event passes at -%d/+%d min around high-impact news",
            self.cfg.ops.interval_minutes,
            self.cfg.ops.event_lookahead_minutes,
            self.cfg.ops.event_followup_minutes,
        )
        while not self._stop and (max_cycles is None or cycles < max_cycles):
            wake_at, reason = self._next_wake(last_run)
            now = utcnow()
            if wake_at > now:
                delay = (wake_at - now).total_seconds()
                log.info("next pass in %.0fs (%s)", delay, reason)
                self._sleep(delay)
                if self._stop:
                    break
            if reason != "interval":
                self._fired.add(reason)

            try:
                result = self.run_once(reason)
                print(result.render(), flush=True)
            except Exception as exc:
                # A crashed cycle must not end the loop — a bot that dies at
                # 3am with positions open is worse than one that skips a pass.
                log.exception("cycle failed")
                self.db.alert("error", "scheduler", f"cycle failed: {exc}")
                print(f"cycle failed: {exc}", flush=True)
            last_run = utcnow()
            cycles += 1
        log.info("scheduler stopped after %d cycles", cycles)

    def run_once(self, reason: str = "manual") -> CycleResult:
        log.info("cycle start (%s)", reason)
        result = self.runner.run_cycle()
        self.runner.sync_closed_trades()
        return result

    def _sleep(self, seconds: float) -> None:
        """Sleep in slices so Ctrl-C is responsive."""
        end = time.monotonic() + seconds
        while not self._stop and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))
