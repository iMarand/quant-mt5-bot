"""Preflight decides whether the scheduler should commit to a loop at all."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantbot.config import Config
from quantbot.contracts import Candle
from quantbot.decision.execution.paper import PaperBroker
from quantbot.ops.runner import Runner
from quantbot.storage import Database


class _DeadMarket:
    """A market connector whose terminal is unreachable."""

    name = "dead"

    def connect(self):
        raise RuntimeError("mt5.initialize failed: (-6) Terminal: Authorization failed")

    def disconnect(self):
        pass


class _FakeIngestor:
    def __init__(self, market):
        self.market = market


@pytest.fixture()
def cfg(tmp_path):
    c = Config()
    c.db_path = str(tmp_path / "pf.db")
    c.data.symbols = ["EURUSD"]
    return c


def _seed(db, n):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.upsert_candles(
        [
            Candle("EURUSD", "M15", start + timedelta(minutes=15 * i), 1.1, 1.11, 1.09, 1.1, 1)
            for i in range(n)
        ]
    )


def _runner(cfg, db, ingestor=None):
    broker = PaperBroker(balance=10_000)
    broker.connect()
    return Runner(cfg, db, broker, ingestor)


def test_no_candles_blocks_startup(cfg):
    db = Database(cfg.db_file)
    blocking, warnings = _runner(cfg, db).preflight()
    assert any("no M15 candles" in b for b in blocking)


def test_too_few_candles_blocks_startup(cfg):
    db = Database(cfg.db_file)
    _seed(db, 50)
    blocking, _ = _runner(cfg, db).preflight()
    assert any("only 50" in b for b in blocking)


def test_dead_feed_with_stored_data_is_only_a_warning(cfg):
    """Stored candles still produce signals; a dead feed is degraded, not fatal."""
    db = Database(cfg.db_file)
    _seed(db, 500)
    blocking, warnings = _runner(cfg, db, _FakeIngestor(_DeadMarket())).preflight()
    assert blocking == []
    assert any("stored candles only" in w for w in warnings)


def test_dead_feed_with_no_data_blocks(cfg):
    db = Database(cfg.db_file)
    blocking, _ = _runner(cfg, db, _FakeIngestor(_DeadMarket())).preflight()
    assert any("feed unavailable" in b for b in blocking)


def test_healthy_setup_passes_clean(cfg):
    db = Database(cfg.db_file)
    _seed(db, 500)
    blocking, warnings = _runner(cfg, db).preflight()
    assert blocking == [] and warnings == []
