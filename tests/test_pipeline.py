"""Storage, labeling, calendar parsing, journal and gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantbot.config import GateConfig
from quantbot.connectors.forexfactory import ForexFactoryCalendar, parse_number
from quantbot.connectors.policy import FetchPolicy
from quantbot.contracts import Candle, CalendarEvent, Direction, Impact, Regime, Signal
from quantbot.engine.labeling import purge_and_embargo, triple_barrier_labels, walk_forward_splits
from quantbot.engine.model import RuleModel, directional_confidence
from quantbot.learning.journal import resolve_outcomes
from quantbot.ops.gate import evaluate_gate
from quantbot.ops.metrics import expected_calibration_error, max_drawdown, profit_factor
from quantbot.storage import Database


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "test.db")


# -- storage ---------------------------------------------------------------


def test_candle_upsert_is_idempotent(db):
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    c = Candle("EURUSD", "M15", ts, 1.1, 1.2, 1.0, 1.15, 100)
    db.upsert_candles([c, c])
    db.upsert_candles([Candle("EURUSD", "M15", ts, 1.1, 1.3, 1.0, 1.18, 200)])
    df = db.load_candles("EURUSD", "M15")
    assert len(df) == 1
    assert df["close"].iloc[0] == 1.18, "later data must overwrite, not duplicate"


def test_calendar_upsert_fills_in_the_actual_later(db):
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc)
    scheduled = CalendarEvent("e1", "ff", "USD", "CPI", ts, Impact.HIGH, forecast=0.3, previous=0.2)
    db.upsert_events([scheduled])
    assert db.load_events()[0].actual is None

    released = CalendarEvent("e1", "ff", "USD", "CPI", ts, Impact.HIGH, forecast=0.3, previous=0.2, actual=0.5)
    db.upsert_events([released])
    events = db.load_events()
    assert len(events) == 1, "same event must upsert, not insert a second row"
    assert events[0].actual == 0.5
    assert events[0].surprise == pytest.approx(0.2)
    assert events[0].revision == pytest.approx(0.3)


def test_upsert_never_erases_a_known_actual(db):
    ts = datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc)
    db.upsert_events([CalendarEvent("e1", "ff", "USD", "CPI", ts, Impact.HIGH, actual=0.5)])
    # A later fetch that has lost the actual must not wipe it.
    db.upsert_events([CalendarEvent("e1", "ff", "USD", "CPI", ts, Impact.HIGH, actual=None)])
    assert db.load_events()[0].actual == 0.5


# -- calendar parsing ------------------------------------------------------


def test_parse_number_handles_display_formats():
    assert parse_number("3.2%") == pytest.approx(3.2)
    assert parse_number("215K") == pytest.approx(215_000)
    assert parse_number("-1.5M") == pytest.approx(-1_500_000)
    assert parse_number("1,234") == pytest.approx(1234)
    assert parse_number("") is None
    assert parse_number("-") is None, "empty must be None, not 0.0"


def test_forexfactory_parse_and_stable_ids():
    payload = [
        {
            "title": "Non-Farm Employment Change",
            "country": "USD",
            "date": "2026-08-07T13:30:00+00:00",
            "impact": "High",
            "forecast": "185K",
            "previous": "206K",
            "actual": "",
        },
        {
            "title": "Retail Sales m/m",
            "country": "GBP",
            "date": "2026-08-07T07:00:00+00:00",
            "impact": "Medium",
            "forecast": "0.3%",
            "previous": "0.1%",
            "actual": "0.5%",
        },
    ]
    policy = FetchPolicy(user_agent="test", cache_dir="artifacts/cache-test")
    cal = ForexFactoryCalendar("http://example.invalid", policy, currencies=["USD"])
    events = cal.parse(payload)
    assert len(events) == 1, "currency filter must apply"
    ev = events[0]
    assert ev.impact is Impact.HIGH
    assert ev.forecast == pytest.approx(185_000)
    assert ev.actual is None

    # The same event re-fetched with an actual must land on the same id.
    payload[0]["actual"] = "190K"
    assert cal.parse(payload)[0].event_id == ev.event_id


# -- labeling --------------------------------------------------------------


def _bars(closes, highs=None, lows=None):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="15min", tz="UTC")
    closes = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes if highs is None else np.array(highs, dtype=float),
            "low": closes if lows is None else np.array(lows, dtype=float),
            "close": closes,
            "volume": np.ones(len(closes)),
        },
        index=idx,
    )


def test_triple_barrier_labels_up_move():
    rng = np.random.default_rng(0)
    base = 1.10 + np.cumsum(rng.normal(0, 0.0002, 60))
    df = _bars(base, base + 0.0003, base - 0.0003)
    labels = triple_barrier_labels(df, horizon_bars=5, atr_mult=1.0)
    valid = labels.dropna()
    assert set(valid["label"].unique()) <= {-1.0, 0.0, 1.0}
    assert (valid["bars_held"] <= 5).all()
    assert (valid["bars_held"] >= 1).all()


def test_walk_forward_splits_always_train_on_the_past():
    splits = walk_forward_splits(2000, n_splits=4, min_train=500)
    assert len(splits) == 4
    for train, test in splits:
        assert train.max() < test.min(), "test data must be strictly in the future"


def test_purge_removes_train_rows_overlapping_the_test_window():
    train = np.arange(0, 100)
    test = np.arange(100, 120)
    purged = purge_and_embargo(None, train, test, horizon_bars=8, embargo_bars=5)
    assert purged.max() < 92, "rows whose label window reaches into test must go"


# -- model contracts -------------------------------------------------------


def test_rule_model_probabilities_sum_to_one():
    row = pd.Series(
        {
            "M15_ema_fast_slow": 0.001,
            "M15_rsi_14": 60.0,
            "M15_bb_pct": 0.7,
            "M15_plus_di": 30.0,
            "M15_minus_di": 15.0,
            "M15_macd_hist_norm": 0.0002,
            "M15_pat_bullish_engulfing": 1.0,
            "last_surprise_signed": 0.5,
        }
    )
    proba = RuleModel().predict_proba(pd.DataFrame([row]))
    assert proba.shape == (1, 3)
    assert proba.sum(axis=1)[0] == pytest.approx(1.0)
    assert (proba >= 0).all()
    assert proba[0][2] > proba[0][0], "a bullish row should favour up"


def test_directional_confidence_ignores_the_flat_class():
    # 60/30 up/down with a big flat mass -> 2/3 confidence in "up".
    d, c = directional_confidence(np.array([0.10, 0.60, 0.30]))
    assert d == 1
    assert c == pytest.approx(0.75)


# -- metrics ---------------------------------------------------------------


def test_profit_factor_and_drawdown():
    profits = pd.Series([100.0, -50.0, 200.0, -50.0])
    assert profit_factor(profits) == pytest.approx(3.0)
    equity = pd.Series([100, 120, 90, 130])
    _, pct = max_drawdown(equity)
    assert pct == pytest.approx(-25.0, abs=0.1)


def test_calibration_error_is_zero_for_a_perfectly_calibrated_model():
    conf = pd.Series([0.9] * 100 + [0.6] * 100)
    correct = pd.Series([1] * 90 + [0] * 10 + [1] * 60 + [0] * 40)
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=0.02)


# -- journal ---------------------------------------------------------------


def test_resolve_outcomes_scores_a_correct_call(db):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    closes = [1.1000 + 0.0010 * i for i in range(30)]  # steady uptrend
    candles = [
        Candle("EURUSD", "M15", start + timedelta(minutes=15 * i), c, c + 0.0002, c - 0.0002, c, 1)
        for i, c in enumerate(closes)
    ]
    db.upsert_candles(candles)

    signal = Signal(
        instrument="EURUSD",
        timeframe="M15",
        ts=start,
        direction=Direction.LONG,
        confidence=0.8,
        horizon_min=120,
        regime=Regime.TRENDING,
    )
    pid = db.record_prediction(signal)
    assert resolve_outcomes(db, "M15") == 1

    row = db.query("SELECT * FROM outcomes WHERE prediction_id=?", (pid,))[0]
    assert row["label"] == 1
    assert row["correct"] == 1
    assert row["realized_return"] > 0


def test_unresolvable_prediction_is_left_alone(db):
    signal = Signal(
        instrument="EURUSD",
        timeframe="M15",
        ts=datetime.now(timezone.utc),
        direction=Direction.LONG,
        confidence=0.8,
        horizon_min=120,
        regime=Regime.TRENDING,
    )
    db.record_prediction(signal)
    assert resolve_outcomes(db, "M15") == 0, "horizon has not elapsed yet"


# -- gate ------------------------------------------------------------------


def test_gate_fails_on_an_empty_journal(db):
    report = evaluate_gate(db, GateConfig())
    assert not report.passed
    assert any(c.name == "sample size" and not c.passed for c in report.checks)


def test_gate_fails_on_too_few_trades_even_when_profitable(db):
    for i in range(20):
        db.record_trade(
            ticket=i,
            symbol="EURUSD",
            side="long",
            volume=0.1,
            entry_price=1.1,
            exit_price=1.11,
            opened_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            closed_at=(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat(),
            profit=100.0,
            broker="paper",
        )
    report = evaluate_gate(db, GateConfig(min_trades=200))
    assert not report.passed, "a short winning streak must not pass the gate"


def test_feature_columns_exclude_label_derived_fields():
    """Label columns describe the future; training on them is direct leakage."""
    from quantbot.features.mtf import LABEL_COLUMNS, feature_columns

    df = pd.DataFrame(
        {
            "M15_rsi_14": [50.0, 55.0],
            "M15_close": [1.1, 1.1],
            "close": [1.1, 1.1],
            "symbol": ["EURUSD", "EURUSD"],
            "label": [1.0, -1.0],
            "fwd_return": [0.001, -0.001],
            "bars_held": [3.0, 5.0],
            "barrier_width": [0.0005, 0.0005],
        }
    )
    cols = feature_columns(df)
    assert "M15_rsi_14" in cols
    for leaky in LABEL_COLUMNS:
        assert leaky not in cols, f"{leaky} leaks the future into training"
    assert "close" not in cols and "M15_close" not in cols


# -- csv import ------------------------------------------------------------


def _write_mt5_export(path):
    rows = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
    for i in range(10):
        rows.append(
            f"2026.03.02\t{i:02d}:00:00\t1.10000\t1.10050\t1.09950\t1.10020\t500\t0\t12"
        )
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_read_mt5_export_parses_tab_format(tmp_path):
    from quantbot.connectors.csv_import import infer_timeframe, read_ohlcv_csv

    df = read_ohlcv_csv(_write_mt5_export(tmp_path / "EURUSD_H1.csv"))
    assert len(df) == 10
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "spread"]
    assert df.index.tz is not None, "timestamps must end up tz-aware UTC"
    assert df["close"].iloc[0] == pytest.approx(1.10020)
    assert infer_timeframe(df.index) == "H1"


def test_csv_import_applies_the_server_utc_offset(tmp_path):
    from quantbot.connectors.csv_import import read_ohlcv_csv

    path = _write_mt5_export(tmp_path / "EURUSD_H1.csv")
    utc = read_ohlcv_csv(path, tz_offset_hours=0)
    gmt3 = read_ohlcv_csv(path, tz_offset_hours=3)
    # A GMT+3 server stamping 00:00 means 21:00 UTC the previous day.
    assert (utc.index[0] - gmt3.index[0]) == pd.Timedelta(hours=3)
    assert gmt3.index[0] == pd.Timestamp("2026-03-01 21:00", tz="UTC")


def test_csv_import_upserts_into_the_candle_store(db, tmp_path):
    from quantbot.connectors.csv_import import import_csv

    path = _write_mt5_export(tmp_path / "EURUSD_H1.csv")
    n, tf = import_csv(db, path, symbol="EURUSD")
    assert (n, tf) == (10, "H1")
    # Re-importing the same file must not duplicate rows.
    import_csv(db, path, symbol="EURUSD")
    assert len(db.load_candles("EURUSD", "H1")) == 10


def test_csv_import_rejects_a_file_with_no_timestamp(tmp_path):
    from quantbot.connectors.csv_import import read_ohlcv_csv

    bad = tmp_path / "bad.csv"
    bad.write_text("open,high,low,close\n1,2,0,1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no date/time column"):
        read_ohlcv_csv(bad)


def test_alerts_are_stored_as_a_single_scannable_line(db):
    """A multi-line remedy in an alert row makes `doctor` unreadable."""
    from quantbot.storage.db import summarize

    db.alert("error", "ingest", "connect failed: (-6) auth\n  1. do this\n  2. then that")
    stored = db.recent_alerts(1)[0]["message"]
    assert "\n" not in stored
    assert stored == "connect failed: (-6) auth"

    long = "x" * 500
    db.alert("warn", "test", long)
    assert len(db.recent_alerts(1)[0]["message"]) <= db.ALERT_MAX_LEN
    assert summarize("\n\n  real line\nsecond") == "real line"


def test_calendar_gap_is_excluded_but_both_sides_are_kept(db):
    """An archive ending before the live feed starts leaves a permanent hole.

    Truncating at a single boundary kept only the older block and silently threw
    away everything the bot collects from now on.
    """
    import pandas as pd

    from quantbot.connectors.calendar_import import calendar_covered_mask, calendar_gaps

    def ev(when, i):
        return CalendarEvent(
            f"e{i}", "test", "USD", "CPI",
            pd.Timestamp(when, tz="UTC").to_pydatetime(), Impact.HIGH,
        )

    # Archive through March, nothing until August, then a live feed.
    events = [ev(f"2026-01-{d:02d} 12:00", d) for d in range(1, 29)]
    events += [ev(f"2026-02-{d:02d} 12:00", 100 + d) for d in range(1, 29)]
    events += [ev(f"2026-03-{d:02d} 12:00", 200 + d) for d in range(1, 29)]
    events += [ev(f"2026-08-{d:02d} 12:00", 300 + d) for d in range(1, 15)]
    db.upsert_events(events)

    gaps = calendar_gaps(db, max_gap_days=14)
    assert len(gaps) == 1
    assert gaps[0][0].month == 3 and gaps[0][1].month == 8

    idx = pd.DatetimeIndex(
        [pd.Timestamp(t, tz="UTC") for t in
         ["2026-02-10 12:00", "2026-05-15 12:00", "2026-08-05 12:00"]]
    )
    mask = calendar_covered_mask(idx, db, max_gap_days=14)
    assert bool(mask.iloc[0]) is True, "archive side must be kept"
    assert bool(mask.iloc[1]) is False, "the hole must be dropped"
    assert bool(mask.iloc[2]) is True, "live-feed side must be kept too"
