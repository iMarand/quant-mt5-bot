"""SQLite-backed store: time-series, calendar, feature store and journal.

The architecture (§9) suggests TimescaleDB + PostgreSQL + Redis. SQLite is used
here so the system runs on a single machine with zero infrastructure; the access
layer below is the only place that knows about SQL, so swapping the backend is a
rewrite of this file and nothing else.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..contracts import Candle, CalendarEvent, Impact, Signal, as_utc, utcnow

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    ts          TEXT NOT NULL,          -- ISO-8601 UTC, bar open
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      REAL NOT NULL DEFAULT 0,
    spread      REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(symbol, timeframe, ts DESC);

CREATE TABLE IF NOT EXISTS calendar_events (
    event_id    TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    currency    TEXT NOT NULL,
    name        TEXT NOT NULL,
    ts_utc      TEXT NOT NULL,
    impact      TEXT NOT NULL,
    forecast    REAL,
    previous    REAL,
    actual      REAL,
    surprise    REAL,
    revision    REAL,
    raw         TEXT,
    fetched_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cal_time ON calendar_events(ts_utc);
CREATE INDEX IF NOT EXISTS idx_cal_ccy ON calendar_events(currency, ts_utc);

CREATE TABLE IF NOT EXISTS features (
    symbol          TEXT NOT NULL,
    timeframe       TEXT NOT NULL,
    ts              TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    payload         TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts, feature_version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    symbol            TEXT NOT NULL,
    timeframe         TEXT NOT NULL,
    direction         TEXT NOT NULL,
    confidence        REAL NOT NULL,
    horizon_min       INTEGER NOT NULL,
    regime            TEXT NOT NULL,
    driving_features  TEXT NOT NULL,
    features          TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    acted_on          INTEGER NOT NULL DEFAULT 0,
    veto_reason       TEXT,
    setup             TEXT,
    rationale         TEXT,
    session           TEXT,
    mode              TEXT
);
-- idx_pred_setup is created in _migrate(): on a pre-existing database the
-- column doesn't exist yet at this point, and indexing it here would fail.
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(symbol, ts);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id   INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    resolved_at     TEXT NOT NULL,
    realized_return REAL NOT NULL,     -- signed, in price units
    realized_r      REAL,              -- in units of the intended risk
    label           INTEGER NOT NULL,  -- -1 down, 0 flat, +1 up
    correct         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    ts            TEXT NOT NULL,
    broker        TEXT NOT NULL,
    ticket        INTEGER,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    volume        REAL NOT NULL,
    price         REAL,
    sl            REAL,
    tp            REAL,
    status        TEXT NOT NULL,
    reason        TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    ticket        INTEGER,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    volume        REAL NOT NULL,
    entry_price   REAL NOT NULL,
    exit_price    REAL NOT NULL,
    opened_at     TEXT NOT NULL,
    closed_at     TEXT NOT NULL,
    profit        REAL NOT NULL,
    r_multiple    REAL,
    regime        TEXT,
    exit_reason   TEXT,
    broker        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_closed ON trades(closed_at);

CREATE TABLE IF NOT EXISTS model_registry (
    version    TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    path       TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    metrics    TEXT NOT NULL,
    params     TEXT NOT NULL,
    active     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    kind      TEXT NOT NULL,
    status    TEXT NOT NULL,
    detail    TEXT
);

CREATE TABLE IF NOT EXISTS symbol_specs (
    symbol        TEXT PRIMARY KEY,
    digits        INTEGER NOT NULL,
    point         REAL NOT NULL,
    contract_size REAL NOT NULL,
    volume_min    REAL NOT NULL,
    volume_max    REAL NOT NULL,
    volume_step   REAL NOT NULL,
    tick_value    REAL NOT NULL,
    tick_size     REAL NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS setup_study (
    setup      TEXT NOT NULL,
    session    TEXT NOT NULL,
    n          INTEGER NOT NULL,
    wins       INTEGER NOT NULL,
    win_rate   REAL NOT NULL,
    avg_r      REAL,
    total_pips REAL,
    weight     REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (setup, session)
);

CREATE TABLE IF NOT EXISTS alerts (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    level   TEXT NOT NULL,
    source  TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


def summarize(message: str, max_len: int = 300) -> str:
    """First meaningful line of a message, truncated — safe for a log table."""
    first = next((ln.strip() for ln in str(message).splitlines() if ln.strip()), "")
    return first if len(first) <= max_len else first[: max_len - 1] + "…"


class Database:
    """Thin, thread-safe SQLite wrapper. All timestamps stored as ISO-8601 UTC."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Additive migrations so an existing database keeps its history."""
        existing = {r["name"] for r in self.query("PRAGMA table_info(predictions)")}
        for column, ddl in (
            ("setup", "TEXT"),
            ("rationale", "TEXT"),
            ("session", "TEXT"),
            ("mode", "TEXT"),
        ):
            if column not in existing:
                with self.connect() as conn:
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {ddl}")
        with self.connect() as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_setup ON predictions(setup)")

    # -- plumbing ----------------------------------------------------------
    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self):
        conn = self._conn
        try:
            conn.execute("BEGIN")
            yield conn
            # executescript() and DDL commit implicitly, so the transaction we
            # opened may already be gone by now — committing then would raise.
            if conn.in_transaction:
                conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params))

    def query_df(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._conn, params=list(params))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # -- candles -----------------------------------------------------------
    def upsert_candles(self, candles: Iterable[Candle]) -> int:
        rows = [
            (
                c.symbol,
                c.timeframe,
                as_utc(c.ts).isoformat(),
                c.open,
                c.high,
                c.low,
                c.close,
                c.volume,
                c.spread,
            )
            for c in candles
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO candles(symbol,timeframe,ts,open,high,low,close,volume,spread)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol,timeframe,ts) DO UPDATE SET
                     open=excluded.open, high=excluded.high, low=excluded.low,
                     close=excluded.close, volume=excluded.volume, spread=excluded.spread""",
                rows,
            )
        return len(rows)

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        sql = "SELECT ts, open, high, low, close, volume, spread FROM candles WHERE symbol=? AND timeframe=?"
        params: list[Any] = [symbol, timeframe]
        if start is not None:
            sql += " AND ts >= ?"
            params.append(as_utc(start).isoformat())
        if end is not None:
            sql += " AND ts <= ?"
            params.append(as_utc(end).isoformat())
        sql += " ORDER BY ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = self.query_df(sql, params)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df.sort_values("ts").set_index("ts")

    def last_candle_ts(self, symbol: str, timeframe: str) -> datetime | None:
        row = self.query(
            "SELECT MAX(ts) AS m FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        )
        if not row or row[0]["m"] is None:
            return None
        return as_utc(datetime.fromisoformat(row[0]["m"]))

    # -- calendar ----------------------------------------------------------
    def upsert_events(self, events: Iterable[CalendarEvent]) -> int:
        now = utcnow().isoformat()
        rows = [
            (
                e.event_id,
                e.source,
                e.currency,
                e.name,
                as_utc(e.ts_utc).isoformat(),
                e.impact.value,
                e.forecast,
                e.previous,
                e.actual,
                e.surprise,
                e.revision,
                json.dumps(e.raw),
                now,
            )
            for e in events
        ]
        if not rows:
            return 0
        with self.connect() as conn:
            # Upsert, not insert: a row's forecast becomes an actual later (§3.1).
            conn.executemany(
                """INSERT INTO calendar_events
                   (event_id,source,currency,name,ts_utc,impact,forecast,previous,actual,
                    surprise,revision,raw,fetched_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET
                     ts_utc=excluded.ts_utc, impact=excluded.impact,
                     forecast=COALESCE(excluded.forecast, calendar_events.forecast),
                     previous=COALESCE(excluded.previous, calendar_events.previous),
                     actual=COALESCE(excluded.actual, calendar_events.actual),
                     surprise=COALESCE(excluded.surprise, calendar_events.surprise),
                     revision=COALESCE(excluded.revision, calendar_events.revision),
                     raw=excluded.raw, fetched_at=excluded.fetched_at""",
                rows,
            )
        return len(rows)

    def load_events(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        currencies: Sequence[str] | None = None,
    ) -> list[CalendarEvent]:
        sql = "SELECT * FROM calendar_events WHERE 1=1"
        params: list[Any] = []
        if start is not None:
            sql += " AND ts_utc >= ?"
            params.append(as_utc(start).isoformat())
        if end is not None:
            sql += " AND ts_utc <= ?"
            params.append(as_utc(end).isoformat())
        if currencies:
            sql += f" AND currency IN ({','.join('?' * len(currencies))})"
            params.extend(currencies)
        sql += " ORDER BY ts_utc"
        out = []
        for r in self.query(sql, params):
            out.append(
                CalendarEvent(
                    event_id=r["event_id"],
                    source=r["source"],
                    currency=r["currency"],
                    name=r["name"],
                    ts_utc=as_utc(datetime.fromisoformat(r["ts_utc"])),
                    impact=Impact(r["impact"]),
                    forecast=r["forecast"],
                    previous=r["previous"],
                    actual=r["actual"],
                    raw=json.loads(r["raw"] or "{}"),
                )
            )
        return out

    def events_df(self) -> pd.DataFrame:
        df = self.query_df("SELECT * FROM calendar_events ORDER BY ts_utc")
        if not df.empty:
            df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, format="ISO8601")
        return df

    # -- feature store -----------------------------------------------------
    def save_features(
        self, symbol: str, timeframe: str, ts: datetime, version: str, payload: dict[str, Any]
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO features(symbol,timeframe,ts,feature_version,payload)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(symbol,timeframe,ts,feature_version)
                   DO UPDATE SET payload=excluded.payload""",
                (symbol, timeframe, as_utc(ts).isoformat(), version, json.dumps(payload)),
            )

    # -- journal -----------------------------------------------------------
    def record_prediction(self, signal: Signal, veto_reason: str | None = None) -> int:
        row = signal.to_row()
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (ts,symbol,timeframe,direction,confidence,horizon_min,regime,
                    driving_features,features,model_version,acted_on,veto_reason,
                    setup,rationale,session,mode)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["ts"],
                    row["symbol"],
                    row["timeframe"],
                    row["direction"],
                    row["confidence"],
                    row["horizon_min"],
                    row["regime"],
                    row["driving_features"],
                    row["features"],
                    row["model_version"],
                    0,
                    veto_reason,
                    row["setup"],
                    row["rationale"],
                    row["session"],
                    row["mode"],
                ),
            )
            return int(cur.lastrowid)

    def mark_acted(self, prediction_id: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE predictions SET acted_on=1 WHERE id=?", (prediction_id,))

    def record_outcome(
        self,
        prediction_id: int,
        realized_return: float,
        label: int,
        correct: bool,
        realized_r: float | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO outcomes
                   (prediction_id,resolved_at,realized_return,realized_r,label,correct)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(prediction_id) DO UPDATE SET
                     resolved_at=excluded.resolved_at,
                     realized_return=excluded.realized_return,
                     realized_r=excluded.realized_r,
                     label=excluded.label, correct=excluded.correct""",
                (
                    prediction_id,
                    utcnow().isoformat(),
                    realized_return,
                    realized_r,
                    label,
                    int(correct),
                ),
            )

    def record_order(self, **kw: Any) -> int:
        cols = (
            "prediction_id ts broker ticket symbol side volume price sl tp status reason".split()
        )
        values = [kw.get(c) for c in cols]
        with self.connect() as conn:
            cur = conn.execute(
                f"INSERT INTO orders({','.join(cols)}) VALUES({','.join('?' * len(cols))})", values
            )
            return int(cur.lastrowid)

    def record_trade(self, **kw: Any) -> int:
        cols = (
            "prediction_id ticket symbol side volume entry_price exit_price opened_at "
            "closed_at profit r_multiple regime exit_reason broker".split()
        )
        values = [kw.get(c) for c in cols]
        with self.connect() as conn:
            cur = conn.execute(
                f"INSERT INTO trades({','.join(cols)}) VALUES({','.join('?' * len(cols))})", values
            )
            return int(cur.lastrowid)

    def trades_df(self, broker: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM trades"
        params: list[Any] = []
        if broker:
            sql += " WHERE broker=?"
            params.append(broker)
        sql += " ORDER BY closed_at"
        df = self.query_df(sql, params)
        if not df.empty:
            for col in ("opened_at", "closed_at"):
                df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
        return df

    def predictions_df(self, with_outcomes: bool = True) -> pd.DataFrame:
        sql = "SELECT p.*, o.realized_return, o.realized_r, o.label, o.correct FROM predictions p"
        sql += " LEFT JOIN outcomes o ON o.prediction_id = p.id" if with_outcomes else ""
        sql += " ORDER BY p.ts"
        df = self.query_df(sql)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    def open_predictions(self) -> list[sqlite3.Row]:
        return self.query(
            """SELECT p.* FROM predictions p
               LEFT JOIN outcomes o ON o.prediction_id=p.id
               WHERE o.prediction_id IS NULL ORDER BY p.ts"""
        )

    # -- model registry ----------------------------------------------------
    def register_model(
        self,
        version: str,
        path: str,
        symbol: str,
        timeframe: str,
        metrics: dict[str, Any],
        params: dict[str, Any],
        activate: bool = False,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO model_registry(version,created_at,path,symbol,timeframe,metrics,params,active)
                   VALUES(?,?,?,?,?,?,?,0)
                   ON CONFLICT(version) DO UPDATE SET
                     path=excluded.path, metrics=excluded.metrics, params=excluded.params""",
                (
                    version,
                    utcnow().isoformat(),
                    path,
                    symbol,
                    timeframe,
                    json.dumps(metrics),
                    json.dumps(params),
                ),
            )
            if activate:
                conn.execute(
                    "UPDATE model_registry SET active=0 WHERE symbol=? AND timeframe=?",
                    (symbol, timeframe),
                )
                conn.execute("UPDATE model_registry SET active=1 WHERE version=?", (version,))

    def active_model(self, symbol: str, timeframe: str) -> sqlite3.Row | None:
        rows = self.query(
            "SELECT * FROM model_registry WHERE symbol=? AND timeframe=? AND active=1"
            " ORDER BY created_at DESC LIMIT 1",
            (symbol, timeframe),
        )
        return rows[0] if rows else None

    def model_history(self, symbol: str, timeframe: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM model_registry WHERE symbol=? AND timeframe=? ORDER BY created_at DESC",
            (symbol, timeframe),
        )

    # -- ops ---------------------------------------------------------------
    def log_run(self, kind: str, status: str, detail: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runs(ts,kind,status,detail) VALUES(?,?,?,?)",
                (utcnow().isoformat(), kind, status, detail),
            )

    #: Alerts are a scannable log, not a place for multi-line remedies. Long
    #: guidance belongs in the runtime logger; storing it here made `doctor`
    #: unreadable.
    ALERT_MAX_LEN = 300

    def alert(self, level: str, source: str, message: str) -> None:
        summary = summarize(message, self.ALERT_MAX_LEN)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO alerts(ts,level,source,message) VALUES(?,?,?,?)",
                (utcnow().isoformat(), level, source, summary),
            )

    def save_symbol_spec(self, spec) -> None:
        """Cache the broker's real contract details.

        Offline tools (study, backtest) otherwise fall back to FX defaults,
        which are wrong by 100x for JPY pairs and 1000x for gold — making both
        stop distances and pip counts meaningless for those symbols.
        """
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO symbol_specs
                   (symbol,digits,point,contract_size,volume_min,volume_max,
                    volume_step,tick_value,tick_size,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol) DO UPDATE SET
                     digits=excluded.digits, point=excluded.point,
                     contract_size=excluded.contract_size,
                     volume_min=excluded.volume_min, volume_max=excluded.volume_max,
                     volume_step=excluded.volume_step, tick_value=excluded.tick_value,
                     tick_size=excluded.tick_size, updated_at=excluded.updated_at""",
                (
                    spec.symbol, spec.digits, spec.point, spec.contract_size,
                    spec.volume_min, spec.volume_max, spec.volume_step,
                    spec.tick_value, spec.tick_size, utcnow().isoformat(),
                ),
            )

    def load_symbol_spec(self, symbol: str):
        from ..contracts import SymbolSpec

        rows = self.query("SELECT * FROM symbol_specs WHERE symbol=?", (symbol,))
        if not rows:
            return None
        r = rows[0]
        return SymbolSpec(
            symbol=r["symbol"], digits=int(r["digits"]), point=float(r["point"]),
            contract_size=float(r["contract_size"]), volume_min=float(r["volume_min"]),
            volume_max=float(r["volume_max"]), volume_step=float(r["volume_step"]),
            tick_value=float(r["tick_value"]), tick_size=float(r["tick_size"]),
        )

    def save_setup_study(self, rows: list[dict]) -> int:
        """Persist a counterfactual study so the runtime can use it cheaply."""
        if not rows:
            return 0
        now = utcnow().isoformat()
        payload = [
            (
                r["setup"], r["session"], int(r["n"]), int(r["wins"]),
                float(r["win_rate"]), r.get("avg_r"), r.get("total_pips"),
                float(r["weight"]), now,
            )
            for r in rows
        ]
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO setup_study
                   (setup,session,n,wins,win_rate,avg_r,total_pips,weight,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(setup,session) DO UPDATE SET
                     n=excluded.n, wins=excluded.wins, win_rate=excluded.win_rate,
                     avg_r=excluded.avg_r, total_pips=excluded.total_pips,
                     weight=excluded.weight, created_at=excluded.created_at""",
                payload,
            )
        return len(payload)

    def setup_study(self) -> list:
        return self.query("SELECT * FROM setup_study")

    def recent_alerts(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,))
