"""Counterfactual study: what would each strategy have earned, and when?

The direction model asks one question of every bar: "up or down?" That is the
wrong question for a strategy-first system, and it is why a generic model sits
at the majority-class baseline — most bars have no answer worth having.

This asks a different question, of every bar where a setup *actually fired*:

    had this setup been traded here, with this mode's real stop and target,
    what would have happened — win or lose, how many R, how many pips, how
    long held?

Every trigger is simulated, including the ones the live system would have
skipped (already in a position, outside session, model vetoed). That is the
counterfactual part: outcomes from paths not taken (architecture §7.4), which is
what makes the sample large enough to learn from. 23 journalled trades cannot
tell you when `breakout` works; 20,000 simulated triggers can.

Each outcome is tagged with the context it happened in — hour, session, regime,
news window, surprise, volatility percentile — so the result answers *which
strategy wins, in which session, at which hour, for how many pips*.

Caveats, stated plainly: this uses bar highs/lows, so intrabar path is unknown
and a bar touching both barriers is scored as a loss. Spread is charged at entry
from the recorded per-bar spread. There is no slippage model. These are
optimistic-leaning estimates of a rule's behaviour, not a P&L forecast.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..contracts import Direction, SymbolSpec, tf_minutes
from ..decision.risk import RiskManager
from ..decision.sessions import SESSIONS
from ..features import feature_columns
from ..strategy.base import StrategyContext

log = logging.getLogger(__name__)

#: Context columns recorded alongside every simulated outcome.
CONTEXT_COLUMNS = [
    "hour",
    "dow",
    "session",
    "regime",
    "in_news_window",
    "minutes_since_last_high",
    "last_surprise_signed",
    "atr_percentile",
    "adx",
]


def _session_of(ts: pd.Timestamp) -> str:
    live = [name for name, s in SESSIONS.items() if s.contains(ts.to_pydatetime())]
    # The overlap is the interesting label when it applies.
    if "london_ny_overlap" in live:
        return "london_ny_overlap"
    return live[0] if live else "off_session"


def simulate_setup_outcomes(
    cfg,
    db,
    symbol: str,
    mode_name: str,
    book,
    risk: RiskManager,
    max_hold_bars: int | None = None,
    spec: SymbolSpec | None = None,
    predictor=None,
) -> pd.DataFrame:
    """One row per setup trigger, with the trade it would have produced."""
    from ..engine.predictor import Predictor
    from ..engine.regime import classify_regime

    predictor = predictor or Predictor(cfg, db)
    base_tf = cfg.data.base_timeframe
    frame = predictor.build_features(symbol, bars=cfg.data.history_bars)
    feats = feature_columns(frame)
    frame = frame.dropna(subset=feats, how="all")

    candles = db.load_candles(symbol, base_tf, limit=cfg.data.history_bars)
    frame = frame.join(candles[["high", "low", "close", "spread"]], how="inner", rsuffix="_bar")
    if frame.empty:
        return pd.DataFrame()

    # Real contract details when we have them; FX defaults are wrong by 100x
    # for JPY pairs and 1000x for gold.
    spec = spec or db.load_symbol_spec(symbol) or SymbolSpec(symbol=symbol)
    horizon = max_hold_bars or max(cfg.model.horizon_bars * 3, 24)
    atr_col = f"{base_tf}_atr"

    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    spread_pts = frame["spread"].to_numpy(float)
    atr = frame[atr_col].to_numpy(float) if atr_col in frame else np.full(len(frame), np.nan)
    index = frame.index

    rows: list[dict] = []
    prev = None
    for i, (ts, row) in enumerate(frame.iterrows()):
        ctx = StrategyContext(row, prev, base_tf)
        prev = row
        if i + 2 >= len(frame) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        triggered = book.triggered_setups(ctx)
        if not triggered:
            continue

        spread = spread_pts[i] * spec.point
        context = _context_of(row, ts, classify_regime, cfg)

        for setup in triggered:
            entry = close[i] + (spread / 2 if setup.direction is Direction.LONG else -spread / 2)
            sl, tp, risk_distance = risk.stop_and_target(
                entry, setup.direction, atr[i], spec, spread=spread
            )
            outcome = _walk_to_barrier(
                high, low, close, i, min(i + horizon, len(frame) - 1), setup.direction, sl, tp
            )
            if outcome is None:
                continue
            exit_price, exit_reason, bars_held = outcome
            move = (exit_price - entry) * setup.direction.sign
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "mode": mode_name,
                    "setup": setup.name,
                    "direction": setup.direction.value,
                    "quality": round(setup.quality, 4),
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "minutes_held": bars_held * tf_minutes(base_tf),
                    "pips": move / (spec.point * 10),
                    "r_multiple": move / risk_distance if risk_distance else np.nan,
                    "won": int(move > 0),
                    **context,
                }
            )

    df = pd.DataFrame(rows)
    log.info("counterfactual %s/%s: %d simulated triggers", symbol, mode_name, len(df))
    return df


def _context_of(row, ts, classify_regime, cfg) -> dict:
    base = cfg.data.base_timeframe
    return {
        "hour": int(ts.hour),
        "dow": int(ts.dayofweek),
        "session": _session_of(ts),
        "regime": classify_regime(row, cfg.risk.news_veto_minutes * 2).value,
        "in_news_window": float(row.get("in_news_window", 0.0) or 0.0),
        "minutes_since_last_high": float(row.get("minutes_since_last_high", 1440.0) or 1440.0),
        "last_surprise_signed": float(row.get("last_surprise_signed", 0.0) or 0.0),
        "atr_percentile": float(row.get(f"{base}_atr_percentile", np.nan) or np.nan),
        "adx": float(row.get(f"{base}_adx", np.nan) or np.nan),
    }


def _walk_to_barrier(high, low, close, start, end, direction, sl, tp):
    """First barrier touched after `start`. Stop wins ties (see module note)."""
    for k in range(start + 1, end + 1):
        if direction is Direction.LONG:
            hit_sl, hit_tp = low[k] <= sl, high[k] >= tp
        else:
            hit_sl, hit_tp = high[k] >= sl, low[k] <= tp
        if hit_sl:
            return sl, "stop_loss", k - start
        if hit_tp:
            return tp, "take_profit", k - start
    if end > start:
        return close[end], "timeout", end - start
    return None


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def summarize(df: pd.DataFrame, by: list[str], min_n: int = 20) -> pd.DataFrame:
    """Win rate and expectancy grouped however you ask.

    Pips are only reported when the group is a single instrument. A pip is not
    the same size on EURUSD, USDJPY and XAUUSD, so summing them across symbols
    produces impressive-looking nonsense; R is the unit that compares.
    """
    if df.empty:
        return df
    aggs = {
        "n": ("won", "size"),
        "win_rate": ("won", "mean"),
        "avg_r": ("r_multiple", "mean"),
        "total_r": ("r_multiple", "sum"),
        "median_minutes": ("minutes_held", "median"),
    }
    single_symbol = "symbol" in by or df["symbol"].nunique() <= 1
    if single_symbol and "pips" in df.columns:
        aggs["total_pips"] = ("pips", "sum")
        aggs["avg_pips"] = ("pips", "mean")

    grouped = df.groupby(by).agg(**aggs)
    grouped = grouped[grouped["n"] >= min_n]
    return grouped.sort_values("avg_r", ascending=False).round(4)


def best_hours(df: pd.DataFrame, setup: str | None = None, min_n: int = 20) -> pd.DataFrame:
    """Which hours of the day a setup actually pays for itself."""
    sub = df if setup is None else df[df["setup"] == setup]
    return summarize(sub, ["hour"], min_n=min_n)


def profitable_combinations(
    df: pd.DataFrame, min_n: int = 30, min_avg_r: float = 0.05
) -> pd.DataFrame:
    """setup x session combinations that cleared a bar, best first.

    This is the table the whole module exists for: *which strategy won, in
    which session*. Treat it as a hypothesis generator — anything here was
    selected by looking at the same data it is scored on, so it needs
    out-of-sample confirmation before it means anything.
    """
    table = summarize(df, ["setup", "session"], min_n=min_n)
    if table.empty:
        return table
    return table[table["avg_r"] >= min_avg_r]


def run_study(cfg, db, symbols: list[str] | None = None) -> pd.DataFrame:
    """Simulate every mode over every symbol and concatenate the outcomes."""
    from ..decision.modes import build_modes
    from ..engine.predictor import Predictor
    from ..strategy import StrategyBook

    frames = []
    for mode in build_modes(cfg):
        book = StrategyBook(mode.cfg.strategy)
        risk = RiskManager(mode.cfg.risk)
        predictor = Predictor(mode.cfg, db)
        for symbol in symbols or cfg.data.symbols:
            try:
                frames.append(
                    simulate_setup_outcomes(
                        mode.cfg, db, symbol, mode.name, book, risk, predictor=predictor
                    )
                )
            except Exception as exc:
                log.error("study %s/%s failed: %s", mode.name, symbol, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def holdout_check(
    df: pd.DataFrame,
    by: list[str] | None = None,
    min_n: int = 30,
    split: float = 0.6,
) -> pd.DataFrame:
    """Did the combinations that looked good early still work later?

    With ~11 setups across ~6 sessions there are dozens of combinations, and
    ranking them on one dataset will always surface something impressive. This
    splits the study period in two: picks the winners on the earlier portion,
    then scores those same picks on the later portion, which they had no part in
    selecting.

    A combination that is positive in both halves is a hypothesis worth testing
    live. One that flips sign was noise, and the table is doing its job by
    saying so.
    """
    if df.empty:
        return df
    by = by or ["setup", "session"]
    ordered = df.sort_values("ts")
    cut = ordered["ts"].quantile(split)
    early, late = ordered[ordered["ts"] <= cut], ordered[ordered["ts"] > cut]
    if early.empty or late.empty:
        return pd.DataFrame()

    cols = ["n", "win_rate", "avg_r", "total_r"]
    a = summarize(early, by, min_n=min_n)[cols]
    b = summarize(late, by, min_n=max(10, min_n // 3))[cols]
    joined = a.join(b, how="inner", lsuffix="_early", rsuffix="_late")
    if joined.empty:
        return joined

    joined["held_up"] = (joined["avg_r_early"] > 0) & (joined["avg_r_late"] > 0)
    joined["split_at"] = cut.date().isoformat()
    return joined.sort_values("avg_r_early", ascending=False).round(4)
