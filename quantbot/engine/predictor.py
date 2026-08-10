"""Prediction engine (architecture §5) — features in, `Signal` out.

Never places an order. It emits a signal; `decision.risk` decides independently
whether that signal is tradeable (§1.3).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..contracts import Direction, Regime, Signal, as_utc, tf_minutes
from ..features import build_feature_frame, feature_columns
from ..storage import Database
from ..strategy import StrategyBook
from .model import Ensemble, GBMModel, RuleModel, directional_confidence
from .regime import classify_regime

log = logging.getLogger(__name__)


class Predictor:
    def __init__(self, cfg: Config, db: Database) -> None:
        self.cfg = cfg
        self.db = db
        self._cache: dict[tuple[str, str], tuple[str, Ensemble]] = {}
        self.book = StrategyBook(cfg.strategy)

    # -- model loading -----------------------------------------------------
    def ensemble_for(self, symbol: str, timeframe: str) -> Ensemble:
        """Loads the registry-active model, hot-reloading when retrain swaps it."""
        row = self.db.active_model(symbol, timeframe)
        key = (symbol, timeframe)
        version = row["version"] if row else "none"
        cached = self._cache.get(key)
        if cached and cached[0] == version:
            return cached[1]

        gbm = None
        if row is not None:
            path = Path(row["path"])
            if path.exists():
                try:
                    gbm = GBMModel.load(path)
                except Exception as exc:
                    log.error("failed to load model %s: %s", row["version"], exc)
                    self.db.alert("error", "predictor", f"model load failed: {exc}")
            else:
                log.warning("registry points at missing file %s", path)

        ens = Ensemble(RuleModel(), gbm)
        self._cache[key] = (version, ens)
        return ens

    # -- feature assembly --------------------------------------------------
    def build_features(self, symbol: str, bars: int | None = None) -> pd.DataFrame:
        bars = bars or self.cfg.data.history_bars
        frames = {}
        base_min = tf_minutes(self.cfg.data.base_timeframe)
        for tf in self.cfg.data.timeframes:
            # Higher timeframes need proportionally fewer bars for the same span.
            n = max(300, int(bars * base_min / tf_minutes(tf)))
            df = self.db.load_candles(symbol, tf, limit=n)
            if not df.empty:
                frames[tf] = df
        if self.cfg.data.base_timeframe not in frames:
            raise ValueError(
                f"no {self.cfg.data.base_timeframe} candles for {symbol}; run `ingest` first"
            )
        correlated = {}
        for sym in self.cfg.data.correlated_symbols:
            df = self.db.load_candles(sym, self.cfg.data.base_timeframe, limit=bars)
            if not df.empty:
                correlated[sym] = df

        events = self.db.events_df()
        return build_feature_frame(
            frames,
            events,
            symbol=symbol,
            base_timeframe=self.cfg.data.base_timeframe,
            atr_period=self.cfg.risk.atr_period,
            news_window_min=self.cfg.risk.news_veto_minutes * 2,
            correlated=correlated,
        )

    # -- inference ---------------------------------------------------------
    def predict_row(
        self, symbol: str, row: pd.Series, feats: list[str], prev: pd.Series | None = None
    ) -> Signal:
        """Strategy-first: a named setup must fire before anything else happens."""
        tf = self.cfg.data.base_timeframe
        ens = self.ensemble_for(symbol, tf)

        # The model is consulted only to *adjust* a decision the setups made.
        proba = None
        if self.cfg.strategy.effective_model_role() != "off":
            X = row.reindex(feats).to_frame().T.apply(pd.to_numeric, errors="coerce")
            proba = ens.predict_proba(X)[0]

        decision = self.book.decide(row, prev, tf, proba)
        regime = classify_regime(row, news_window_min=self.cfg.risk.news_veto_minutes * 2)

        driving: dict[str, float] = {
            f"setup_{s.name}": round(s.quality, 4) for s in decision.setups
        }
        if proba is not None:
            driving["p_down"], driving["p_flat"], driving["p_up"] = (
                round(float(p), 4) for p in proba
            )

        return Signal(
            instrument=symbol,
            timeframe=tf,
            ts=as_utc(row.name.to_pydatetime()),
            direction=decision.direction,
            confidence=float(min(max(decision.confidence, 0.0), 1.0)),
            horizon_min=self.cfg.model.horizon_bars * tf_minutes(tf),
            regime=regime,
            driving_features=driving,
            model_version=ens.version,
            features={k: _clean(row.get(k)) for k in feats},
            setup=decision.setup_names,
            rationale="; ".join(
                decision.reasons[:4] + ([decision.model_note] if decision.model_note else [])
            ),
        )

    def predict_latest(self, symbol: str) -> Signal:
        df = self.build_features(symbol)
        feats = feature_columns(df)
        df = df.dropna(subset=feats, how="all")
        if df.empty:
            raise ValueError(f"no usable feature rows for {symbol}")
        prev = df.iloc[-2] if len(df) > 1 else None
        return self.predict_row(symbol, df.iloc[-1], feats, prev=prev)

    def predict_frame(
        self, symbol: str, df: pd.DataFrame, proba: "np.ndarray | None" = None
    ) -> list[Signal]:
        """Batch decisioning — used by the backtester.

        `proba` lets the caller supply out-of-sample model probabilities; when
        omitted the registry model is used, which is in-sample over history.
        """
        feats = feature_columns(df)
        tf = self.cfg.data.base_timeframe
        ens = self.ensemble_for(symbol, tf)
        if proba is None and self.cfg.strategy.effective_model_role() != "off":
            proba = ens.predict_proba(df[feats])

        out: list[Signal] = []
        prev = None
        for i, (ts, row) in enumerate(df.iterrows()):
            decision = self.book.decide(
                row, prev, tf, None if proba is None else proba[i]
            )
            prev = row
            out.append(
                Signal(
                    instrument=symbol,
                    timeframe=tf,
                    ts=as_utc(ts.to_pydatetime()),
                    direction=decision.direction,
                    confidence=float(min(max(decision.confidence, 0.0), 1.0)),
                    horizon_min=self.cfg.model.horizon_bars * tf_minutes(tf),
                    regime=classify_regime(row, self.cfg.risk.news_veto_minutes * 2),
                    driving_features={
                        f"setup_{s.name}": round(s.quality, 4) for s in decision.setups
                    },
                    model_version=ens.version,
                    setup=decision.setup_names,
                    rationale="; ".join(decision.reasons[:3]),
                )
            )
        return out


def _clean(value: object) -> float:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(f) else round(f, 8)
