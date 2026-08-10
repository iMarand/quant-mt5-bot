"""One trading cycle, end to end (architecture phase 3).

    ingest -> features -> predict -> journal -> risk -> execute -> manage -> resolve

Every prediction is journaled *before* the risk layer sees it, including vetoed
ones, so the journal records what the model thought and separately what the
system did about it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from ..config import Config
from ..connectors.ingest import Ingestor
from ..contracts import Direction, Signal, tf_minutes, utcnow
from ..decision.execution.base import Broker
from ..decision.manager import TradeManager
from ..decision.modes import build_modes
from ..decision.pretrade import GateResult, PreTradeGate
from ..decision.risk import DailyLossTracker, RiskManager, TradePlan, Veto
from ..engine.predictor import Predictor
from ..features import feature_columns
from ..strategy import ConfirmationPolicy
from ..learning.journal import resolve_outcomes
from ..storage import Database

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    ts: pd.Timestamp
    signals: list[Signal] = field(default_factory=list)
    plans: list[TradePlan] = field(default_factory=list)
    vetoes: list[tuple[str, Veto]] = field(default_factory=list)
    fills: list[str] = field(default_factory=list)
    managed: list[str] = field(default_factory=list)
    gates: list[tuple[str, "GateResult"]] = field(default_factory=list)
    resolved: int = 0
    errors: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"cycle @ {self.ts:%Y-%m-%d %H:%M:%S} UTC"]
        for label, gate in self.gates:
            if not gate.allowed:
                lines.append(f"  {label}: stand down [{gate.stage}] {gate.reason}")
        for s in self.signals:
            tag = f"{s.mode}/" if s.mode else ""
            if s.direction.value == "flat":
                lines.append(f"  {tag}{s.instrument}: no setup triggered")
            else:
                lines.append(
                    f"  {tag}{s.instrument} {s.direction.value:<5} conf={s.confidence:.3f} "
                    f"[{s.setup}] regime={s.regime.value}"
                )
                if s.rationale:
                    lines.append(f"      why: {s.rationale}")
        for symbol, veto in self.vetoes:
            lines.append(f"  veto {symbol}: {veto}")
        for plan in self.plans:
            lines.append(
                f"  plan {plan.symbol} {plan.side.value} vol={plan.volume} "
                f"entry={plan.entry} sl={plan.sl} tp={plan.tp} rr={plan.rr:.2f} "
                f"risk={plan.risk_amount:.2f}"
            )
        lines += [f"  fill {f}" for f in self.fills]
        lines += [f"  manage {m}" for m in self.managed]
        if self.resolved:
            lines.append(f"  resolved {self.resolved} past predictions")
        lines += [f"  ERROR {e}" for e in self.errors]
        return "\n".join(lines)


class Runner:
    def __init__(
        self,
        cfg: Config,
        db: Database,
        broker: Broker,
        ingestor: Ingestor | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.broker = broker
        self.ingestor = ingestor
        self.daily = DailyLossTracker()

        # One predictor / risk manager per mode: scalp and swing genuinely need
        # different timeframes, stops and thresholds (decision/modes.py).
        self.modes = build_modes(cfg)
        self.runtimes: dict[str, dict] = {}
        for mode in self.modes:
            self.runtimes[mode.name] = {
                "mode": mode,
                "predictor": Predictor(mode.cfg, db),
                "risk": RiskManager(mode.cfg.risk),
                "manager": TradeManager(
                    mode.cfg.risk,
                    broker,
                    max_bars_in_trade=mode.max_bars_in_trade,
                    bar_minutes=tf_minutes(mode.base_timeframe),
                ),
                "gate": PreTradeGate(cfg.pretrade, mode.session_policy),
                "confirm": ConfirmationPolicy(
                    mode=cfg.strategy.confirmation,
                    confirm_atr_mult=cfg.strategy.confirm_atr_mult,
                    max_wait_bars=cfg.strategy.confirm_max_wait_bars,
                ),
            }
        log.info("modes: %s", ", ".join(str(m) for m in self.modes))
        self._guard_live()

    def preflight(self) -> tuple[list[str], list[str]]:
        """Check the setup before the scheduler commits to a loop.

        Returns (blocking, warnings). Looping every 15 minutes on a setup that
        cannot work is worse than refusing to start — it buries the real cause
        under repeated identical failures.
        """
        blocking: list[str] = []
        warnings: list[str] = []
        base_tf = self.cfg.data.base_timeframe

        have_data = True
        for symbol in self.cfg.data.symbols:
            n = self.db.query(
                "SELECT COUNT(*) c FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, base_tf),
            )[0]["c"]
            if n == 0:
                blocking.append(f"no {base_tf} candles for {symbol}")
                have_data = False
            elif n < 300:
                blocking.append(f"only {n} {base_tf} candles for {symbol} (need ~300+)")
                have_data = False

        if self.ingestor is not None:
            try:
                self.ingestor.market.connect()
            except Exception as exc:
                detail = str(exc).splitlines()[0]
                # With stored data the loop still produces signals; it just
                # cannot refresh candles. That is degraded, not unusable.
                if have_data:
                    warnings.append(
                        f"market data feed unavailable ({detail}) — running on stored "
                        "candles only, no fresh bars"
                    )
                else:
                    blocking.append(f"market data feed unavailable: {detail}")

        try:
            self.broker.equity()
        except Exception as exc:
            blocking.append(f"broker not usable: {exc}")

        return blocking, warnings

    def _guard_live(self) -> None:
        """§1.1 enforced at the boundary: nothing live without explicit opt-in."""
        if not getattr(self.broker, "is_demo", True) and not self.cfg.broker.allow_live:
            raise RuntimeError(
                "refusing to run against a live account with broker.allow_live=false"
            )

    # -- the cycle ---------------------------------------------------------
    def run_cycle(self, ingest: bool = True) -> CycleResult:
        result = CycleResult(ts=pd.Timestamp(utcnow()))

        # A dropped terminal must not silently skip a cycle: reconnect first,
        # then re-adopt anything open so trailing resumes.
        if not self.ensure_connected():
            result.errors.append("broker unreachable — cycle skipped")
            return result
        try:
            result.managed.extend(self.adopt_open_positions())
        except Exception as exc:
            result.errors.append(f"adopt: {exc}")

        if ingest and self.ingestor is not None:
            try:
                self.ingestor.ingest_calendar()
                self.ingestor.ingest_market(count=max(400, self.cfg.model.horizon_bars * 40))
            except Exception as exc:
                result.errors.append(f"ingest: {exc}")
                log.error("ingest failed: %s", exc)

        # Manage what's already open before opening anything new.
        try:
            result.managed.extend(self._manage_open())
        except Exception as exc:
            result.errors.append(f"manage: {exc}")

        events = self.db.events_df()
        equity = self.broker.equity()

        for name, rt in self.runtimes.items():
            for symbol in self.cfg.data.symbols:
                try:
                    self._process_symbol(name, rt, symbol, equity, events, result)
                except Exception as exc:
                    log.exception("%s/%s failed", name, symbol)
                    result.errors.append(f"{name}/{symbol}: {exc}")
                    self.db.alert("error", "runner", f"{name}/{symbol}: {exc}")

        try:
            result.resolved = resolve_outcomes(self.db, self.cfg.data.base_timeframe)
        except Exception as exc:
            result.errors.append(f"resolve: {exc}")

        self.db.log_run(
            "cycle",
            "error" if result.errors else "ok",
            f"{len(result.signals)} signals, {len(result.plans)} plans, {len(result.fills)} fills",
        )
        return result

    def _process_symbol(
        self,
        mode_name: str,
        rt: dict,
        symbol: str,
        equity: float,
        events: pd.DataFrame,
        result: CycleResult,
    ) -> None:
        mode = rt["mode"]
        predictor: Predictor = rt["predictor"]
        base_tf = mode.base_timeframe

        # --- ordered pre-trade checks, before any indicator work ----------
        gate_result = rt["gate"].check(
            symbol,
            now=utcnow(),
            events=events,
            last_bar_ts=self.db.last_candle_ts(symbol, base_tf),
            base_timeframe=base_tf,
        )
        result.gates.append((f"{mode_name}/{symbol}", gate_result))
        if not gate_result.allowed:
            return

        df = predictor.build_features(symbol)
        feats = feature_columns(df)
        df = df.dropna(subset=feats, how="all")
        if df.empty:
            result.errors.append(f"{mode_name}/{symbol}: no feature rows")
            return

        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        signal = predictor.predict_row(symbol, row, feats, prev=prev)
        signal.mode = mode_name
        signal.session = "+".join(gate_result.sessions)
        result.signals.append(signal)
        if signal.direction is Direction.FLAT:
            return  # no setup fired; nothing to risk-check

        atr_value = _latest_atr(row, base_tf)
        spec = self.broker.symbol_spec(symbol)

        # Entry timing: a setup says "conditions are right", not "the move has
        # started". When enabled, hold the signal until price confirms it.
        confirm: ConfirmationPolicy = rt["confirm"]
        key = f"{mode_name}/{symbol}"
        if confirm.enabled:
            price_now = float(row.get(f"{base_tf}_close", 0.0) or 0.0)
            if confirm.pending_for(key) is None:
                confirm.register(
                    key, symbol, signal.direction, price_now, atr_value,
                    signal.setup, signal.confidence, utcnow(), signal.reasons_list(),
                )
                result.managed.append(f"{key}: waiting for entry confirmation")
                self.db.record_prediction(signal, veto_reason="awaiting_confirmation")
                return
            ok, note = confirm.check(key, price_now)
            if note:
                result.managed.append(f"{key}: {note}")
            if not ok:
                self.db.record_prediction(signal, veto_reason="awaiting_confirmation")
                return
        # The paper broker has no feed of its own; drive it from the last bar so
        # the same cycle code works with or without a real broker connection.
        if hasattr(self.broker, "set_price"):
            close_col = f"{base_tf}_close"
            last_close = row.get(close_col, row.get("close"))
            if last_close is not None and pd.notna(last_close):
                self.broker.set_price(symbol, float(last_close), ts=row.name.to_pydatetime())
        try:
            tick = self.broker.tick(symbol)
            entry_price = tick.ask if signal.direction is Direction.LONG else tick.bid
            spread = max(tick.ask - tick.bid, 0.0)
        except Exception as exc:
            result.errors.append(f"{mode_name}/{symbol}: no tick ({exc})")
            return

        decision = rt["risk"].evaluate(
            signal,
            equity=equity,
            entry_price=entry_price,
            atr_value=atr_value,
            spec=spec,
            open_positions=self.broker.get_positions(),
            realized_pnl_today=self.daily.value(),
            spread=spread,
            upcoming_events=events,
            now=pd.Timestamp(utcnow()),
        )

        if isinstance(decision, Veto):
            self.db.record_prediction(signal, veto_reason=str(decision))
            result.vetoes.append((symbol, decision))
            return

        prediction_id = self.db.record_prediction(signal)
        result.plans.append(decision)

        if self.cfg.dry_run:
            result.fills.append(f"{mode_name}/{symbol}: DRY RUN, no order sent")
            return

        fill = self.broker.place_order(
            symbol=decision.symbol,
            side=decision.side,
            volume=decision.volume,
            sl=decision.sl,
            tp=decision.tp,
            comment=f"qb{prediction_id}-{mode_name[:5]}",
        )
        self.db.record_order(
            prediction_id=prediction_id,
            ts=fill.ts.isoformat(),
            broker=self.broker.name,
            ticket=fill.ticket,
            symbol=symbol,
            side=decision.side.value,
            volume=decision.volume,
            price=fill.price,
            sl=decision.sl,
            tp=decision.tp,
            status=fill.status.value,
            reason=fill.reason,
        )
        if fill.status.value == "open":
            self.db.mark_acted(prediction_id)
            rt["manager"].register(fill.ticket, decision.risk_distance)
            result.fills.append(
                f"{mode_name}/{symbol} {decision.side.value} {decision.volume} @ {fill.price} (#{fill.ticket})"
            )
        else:
            result.errors.append(f"{mode_name}/{symbol}: order rejected — {fill.reason}")
            self.db.alert("error", "execution", f"{symbol}: {fill.reason}")

    def _manage_open(self) -> list[str]:
        positions = self.broker.get_positions()
        if not positions:
            return []
        atr_by_symbol = {}
        for pos in positions:
            candles = self.db.load_candles(pos.symbol, self.modes[0].base_timeframe, limit=200)
            if candles.empty:
                continue
            from ..features.indicators import atr as atr_fn

            series = atr_fn(candles, self.cfg.risk.atr_period)
            if not series.dropna().empty:
                atr_by_symbol[pos.symbol] = float(series.dropna().iloc[-1])
        actions: list[str] = []
        for rt in self.runtimes.values():
            actions.extend(rt["manager"].manage(positions, atr_by_symbol))
        return actions

    # -- recovery ----------------------------------------------------------
    def adopt_open_positions(self) -> list[str]:
        """Re-attach to positions this process did not open, and repair them.

        Runs on startup and every cycle. Three things happen here:
          1. every position is assigned to a mode's manager, so a restart never
             leaves a trade un-trailed;
          2. a position with no stop loss gets one immediately — an unprotected
             position is the single worst state this system can be in;
          3. the assignment follows the order comment ('qb123-scalp'), so a
             scalp keeps scalp rules after a restart.
        """
        notes: list[str] = []
        try:
            positions = self.broker.get_positions()
        except Exception as exc:
            log.error("could not list positions: %s", exc)
            return [f"position list failed: {exc}"]

        for pos in positions:
            if any(rt["manager"].owns(pos.ticket) for rt in self.runtimes.values()):
                continue

            rt = self._runtime_for(pos)
            if pos.sl is None:
                repaired = self._repair_missing_stop(pos, rt)
                notes.append(repaired)
            rt["manager"].adopt(pos)
            notes.append(
                f"adopted #{pos.ticket} {pos.symbol} {pos.side.value} into '{rt['mode'].name}'"
            )
            self.db.alert(
                "warn", "adopt", f"re-adopted #{pos.ticket} {pos.symbol} after restart"
            )
        return notes

    def _runtime_for(self, pos) -> dict:
        """Which mode should own this position — from its order comment."""
        comment = (pos.comment or "").lower()
        for name, rt in self.runtimes.items():
            if name[:5].lower() in comment:
                return rt
        # Unknown origin: give it to the slowest mode, whose wider stops and
        # later breakeven are the more conservative choice for a trade we
        # cannot attribute.
        return self.runtimes[self.modes[-1].name if len(self.modes) == 1 else self.modes[0].name]

    def _repair_missing_stop(self, pos, rt) -> str:
        """Attach an ATR stop to an unprotected position."""
        mode = rt["mode"]
        candles = self.db.load_candles(pos.symbol, mode.base_timeframe, limit=200)
        spec = self.broker.symbol_spec(pos.symbol)
        atr_value = 0.0
        if not candles.empty:
            from ..features.indicators import atr as atr_fn

            series = atr_fn(candles, mode.cfg.risk.atr_period).dropna()
            if not series.empty:
                atr_value = float(series.iloc[-1])
        if atr_value <= 0:
            atr_value = spec.point * 200  # last resort, better than nothing

        sl, tp, _ = rt["risk"].stop_and_target(pos.entry_price, pos.side, atr_value, spec)
        ok = self.broker.modify_position(pos.ticket, sl=sl, tp=pos.tp or tp)
        msg = f"#{pos.ticket} {pos.symbol} had NO stop loss — {'set to ' + str(sl) if ok else 'REPAIR FAILED'}"
        log.warning(msg)
        self.db.alert("error" if not ok else "warn", "unprotected_position", msg)
        return msg

    def ensure_connected(self) -> bool:
        """Reconnect the broker/feed if the terminal dropped out."""
        try:
            self.broker.equity()
            return True
        except Exception as exc:
            log.warning("broker connection lost (%s); reconnecting", exc)
            self.db.alert("warn", "connection", f"reconnecting: {exc}")
        for attempt in range(1, 4):
            try:
                self.broker.disconnect()
            except Exception:
                pass
            try:
                self.broker.connect()
                self.broker.equity()
                log.info("broker reconnected on attempt %d", attempt)
                self.db.alert("info", "connection", "broker reconnected")
                # A reconnect means new Position objects: re-adopt them so the
                # in-memory trailing state is rebuilt rather than lost.
                for rt in self.runtimes.values():
                    rt["manager"]._owned.clear()
                self.adopt_open_positions()
                return True
            except Exception as exc:
                log.error("reconnect attempt %d failed: %s", attempt, exc)
                time.sleep(5 * attempt)
        self.db.alert("error", "connection", "broker reconnection failed")
        return False

    def _forget(self, ticket: int) -> None:
        for rt in self.runtimes.values():
            rt["manager"].forget(ticket)

    # -- reconciliation ----------------------------------------------------
    def sync_closed_trades(self, lookback_hours: int = 48) -> int:
        """Pull closed deals from the broker into the journal (§6, last bullet)."""
        if not hasattr(self.broker, "deals_since"):
            # Paper broker keeps its own closed-trade list.
            closed = getattr(self.broker, "closed_trades", [])
            n = 0
            for t in closed:
                self.db.record_trade(broker=self.broker.name, **t)
                self.daily.add(float(t["profit"]))
                self._forget(int(t["ticket"]))
                n += 1
            if hasattr(self.broker, "closed_trades"):
                self.broker.closed_trades.clear()
            return n

        since = utcnow() - timedelta(hours=lookback_hours)
        existing = {
            r["ticket"] for r in self.db.query("SELECT ticket FROM trades WHERE ticket IS NOT NULL")
        }
        n = 0
        for deal in self.broker.deals_since(since):
            ticket = int(deal.get("position_id") or deal.get("ticket") or 0)
            if not ticket or ticket in existing or float(deal.get("volume", 0)) == 0:
                continue
            if int(deal.get("entry", 0)) != 1:  # DEAL_ENTRY_OUT only
                continue
            profit = float(deal.get("profit", 0)) + float(deal.get("commission", 0)) + float(
                deal.get("swap", 0)
            )
            self.db.record_trade(
                ticket=ticket,
                symbol=deal.get("symbol", ""),
                side="long" if int(deal.get("type", 0)) == 1 else "short",
                volume=float(deal.get("volume", 0)),
                entry_price=float(deal.get("price", 0)),
                exit_price=float(deal.get("price", 0)),
                opened_at=pd.Timestamp(int(deal.get("time", 0)), unit="s", tz="UTC").isoformat(),
                closed_at=pd.Timestamp(int(deal.get("time", 0)), unit="s", tz="UTC").isoformat(),
                profit=profit,
                exit_reason=deal.get("comment", ""),
                broker=self.broker.name,
            )
            self.daily.add(profit)
            self._forget(ticket)
            n += 1
        return n


def _latest_atr(row: pd.Series, base_tf: str) -> float:
    for name in (f"{base_tf}_atr", "atr"):
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    for name in row.index:
        if name.endswith("_atr") and pd.notna(row[name]):
            return float(row[name])
    return 0.0
