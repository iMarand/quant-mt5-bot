"""Risk manager (architecture §6) — the independent gate between signal and order.

The predictor is optimistic by construction; this layer is the pessimist. It can
veto for reasons the model knows nothing about (daily loss already hit, too much
open exposure, a high-impact release two minutes out) and it owns sizing, SL and
TP placement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from ..config import RiskConfig
from ..contracts import Direction, Position, Regime, Signal, SymbolSpec, as_utc, utcnow

log = logging.getLogger(__name__)


@dataclass
class TradePlan:
    """A signal that survived risk checks, priced and sized."""

    signal: Signal
    symbol: str
    side: Direction
    volume: float
    entry: float
    sl: float
    tp: float
    risk_amount: float
    risk_distance: float
    rr: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_long(self) -> bool:
        return self.side is Direction.LONG


def loss_per_lot(risk_distance: float, spec: SymbolSpec) -> float:
    """Account-currency loss if a 1.0-lot position runs to its stop."""
    if risk_distance <= 0 or spec.tick_size <= 0 or spec.tick_value <= 0:
        return 0.0
    return risk_distance / spec.tick_size * spec.tick_value


@dataclass
class Veto:
    reason: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    # -- sizing ------------------------------------------------------------
    def position_size(
        self,
        equity: float,
        risk_distance: float,
        spec: SymbolSpec,
        risk_pct: float | None = None,
    ) -> tuple[float, float]:
        """Volume such that hitting the stop costs ~`risk_pct` of equity.

        Returns (lots, risk_amount_in_account_currency).
        """
        risk_pct = self.cfg.risk_per_trade_pct if risk_pct is None else risk_pct
        risk_amount = equity * risk_pct / 100.0
        per_lot = loss_per_lot(risk_distance, spec)
        if per_lot <= 0:
            return spec.volume_min, risk_amount
        volume = spec.round_volume(risk_amount / per_lot)
        # Report the risk of the *rounded* volume, not the ideal one — clamping
        # to volume_min on a small account can mean risking more than intended.
        return volume, volume * per_lot

    # -- stops / targets ---------------------------------------------------
    def stop_and_target(
        self,
        entry: float,
        side: Direction,
        atr_value: float,
        spec: SymbolSpec,
        spread: float = 0.0,
    ) -> tuple[float, float, float]:
        """ATR-based SL/TP (addendum §B). Returns (sl, tp, risk_distance).

        The stop is floored so it clears the spread and an absolute minimum.
        On a quiet M5 bar an ATR stop can be narrower than the spread itself,
        which is both un-survivable and — since size scales inversely with stop
        distance — a route to an enormous position.
        """
        distance = max(
            atr_value * self.cfg.sl_atr_mult,
            spread * self.cfg.min_stop_spread_mult,
            spec.point * self.cfg.min_stop_points,
        )
        # The target keeps the intended reward:risk against the widened stop.
        target = max(
            atr_value * self.cfg.tp_atr_mult,
            distance * (self.cfg.tp_atr_mult / max(self.cfg.sl_atr_mult, 1e-9)),
        )
        if side is Direction.LONG:
            sl, tp = entry - distance, entry + target
        else:
            sl, tp = entry + distance, entry - target
        return spec.round_price(sl), spec.round_price(tp), distance

    # -- the gate ----------------------------------------------------------
    def evaluate(
        self,
        signal: Signal,
        *,
        equity: float,
        entry_price: float,
        atr_value: float,
        spec: SymbolSpec,
        open_positions: list[Position],
        realized_pnl_today: float,
        spread: float = 0.0,
        upcoming_events: pd.DataFrame | None = None,
        now: pd.Timestamp | None = None,
    ) -> TradePlan | Veto:
        reasons: list[str] = []

        if signal.direction is Direction.FLAT:
            return Veto("flat_signal")

        if signal.confidence < self.cfg.min_confidence:
            return Veto(
                "low_confidence", f"{signal.confidence:.3f} < {self.cfg.min_confidence:.3f}"
            )

        # Daily loss cap — measured against equity at the *start* of the day is
        # ideal; using current equity is close enough and always available.
        max_daily_loss = -abs(equity * self.cfg.max_daily_loss_pct / 100.0)
        if realized_pnl_today <= max_daily_loss:
            return Veto(
                "daily_loss_cap", f"realized {realized_pnl_today:.2f} <= {max_daily_loss:.2f}"
            )

        if len(open_positions) >= self.cfg.max_open_positions:
            return Veto("max_positions", f"{len(open_positions)} open")

        same_symbol = [p for p in open_positions if p.symbol == signal.instrument]
        if len(same_symbol) >= self.cfg.max_positions_per_symbol:
            return Veto("symbol_exposure", f"{len(same_symbol)} on {signal.instrument}")

        if atr_value <= 0 or not pd.notna(atr_value):
            return Veto("no_volatility_estimate", "ATR unavailable")

        # Hard veto around extreme-impact news (§6).
        veto = self._news_veto(signal, upcoming_events, now)
        if veto is not None:
            return veto

        side = signal.direction
        sl, tp, risk_distance = self.stop_and_target(
            entry_price, side, atr_value, spec, spread=spread
        )

        # A stop this wide means the volatility estimate is broken (bad data,
        # a mixed feed, a stale ATR). Sizing off it would be sizing off garbage.
        stop_pct = risk_distance / entry_price * 100 if entry_price else float("inf")
        if stop_pct > self.cfg.max_sl_distance_pct:
            return Veto(
                "implausible_stop",
                f"stop is {stop_pct:.1f}% of price (max {self.cfg.max_sl_distance_pct}%) "
                f"— check the ATR/candle data",
            )

        reward = abs(tp - entry_price)
        rr = reward / risk_distance if risk_distance > 0 else 0.0
        if rr < self.cfg.min_rr:
            return Veto("poor_rr", f"{rr:.2f} < {self.cfg.min_rr}")

        volume, risk_amount = self.position_size(equity, risk_distance, spec)
        if volume < spec.volume_min:
            return Veto("size_below_min", f"{volume} < {spec.volume_min}")

        # position_size clamps up to volume_min, which can risk more than the
        # budget allows. The right response is to skip the trade, not to take a
        # position the risk config never authorized. Checked before the leverage
        # cap so the message names the more fundamental problem.
        budget = equity * self.cfg.risk_per_trade_pct / 100.0
        if risk_amount > budget * self.cfg.max_risk_overshoot:
            return Veto(
                "risk_exceeds_budget",
                f"min lot risks {risk_amount:.2f} vs budget {budget:.2f} "
                f"(cap {self.cfg.max_risk_overshoot:.2f}x)",
            )

        # Leverage backstop. Risk-based sizing says nothing about position
        # *size*: a very tight stop meets a small risk budget with a huge
        # notional, and slippage on that notional is what actually hurts.
        notional = volume * spec.contract_size * entry_price
        leverage = notional / equity if equity > 0 else float("inf")
        if leverage > self.cfg.max_position_leverage:
            ideal = self.cfg.max_position_leverage * equity / (spec.contract_size * entry_price)
            # Round *down* to the lot step: rounding to nearest would sometimes
            # land just above the very cap being enforced.
            steps = math.floor(ideal / spec.volume_step + 1e-9)
            capped = round(max(steps, 0) * spec.volume_step, 8)
            if capped < spec.volume_min:
                return Veto(
                    "leverage_cap",
                    f"{leverage:.1f}x exceeds {self.cfg.max_position_leverage}x and the "
                    f"capped size is below the minimum lot",
                )
            reasons.append(
                f"size capped {volume}->{capped} lots ({leverage:.1f}x -> "
                f"{self.cfg.max_position_leverage}x leverage)"
            )
            volume = capped
            # Capping size cuts the money at risk proportionally.
            risk_amount = capped * loss_per_lot(risk_distance, spec)

        # Portfolio-level risk budget.
        open_risk_pct = self.cfg.risk_per_trade_pct * len(open_positions)
        if open_risk_pct + self.cfg.risk_per_trade_pct > self.cfg.max_total_risk_pct:
            return Veto("total_risk_cap", f"{open_risk_pct:.2f}% already at risk")

        if signal.regime is Regime.HIGH_VOL:
            reasons.append("high_vol_regime: size unchanged, ATR already widened the stop")

        return TradePlan(
            signal=signal,
            symbol=signal.instrument,
            side=side,
            volume=volume,
            entry=spec.round_price(entry_price),
            sl=sl,
            tp=tp,
            risk_amount=risk_amount,
            risk_distance=risk_distance,
            rr=rr,
            reasons=reasons,
        )

    def _news_veto(
        self,
        signal: Signal,
        upcoming_events: pd.DataFrame | None,
        now: pd.Timestamp | None,
    ) -> Veto | None:
        if self.cfg.news_veto_minutes <= 0 or upcoming_events is None or upcoming_events.empty:
            return None
        now = now or pd.Timestamp(utcnow())
        window = timedelta(minutes=self.cfg.news_veto_minutes)
        soon = upcoming_events[
            (upcoming_events["ts_utc"] >= now - window)
            & (upcoming_events["ts_utc"] <= now + window)
            & (upcoming_events["impact"] == self.cfg.news_veto_impact)
        ]
        if not soon.empty:
            names = ", ".join(soon["name"].head(2).astype(str))
            return Veto("news_window", f"{len(soon)} high-impact within +/-{window}: {names}")
        return None


class DailyLossTracker:
    """Realized PnL since the current UTC day started."""

    def __init__(self) -> None:
        self._day = None
        self._pnl = 0.0

    def add(self, profit: float) -> None:
        self._roll()
        self._pnl += profit

    def value(self) -> float:
        self._roll()
        return self._pnl

    def _roll(self) -> None:
        today = as_utc(utcnow()).date()
        if self._day != today:
            self._day = today
            self._pnl = 0.0
