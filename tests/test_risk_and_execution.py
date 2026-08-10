"""Risk-layer and execution invariants — the parts that cost real money."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quantbot.config import RiskConfig
from quantbot.contracts import Direction, Position, Regime, Signal, SymbolSpec
from quantbot.decision.execution.paper import PaperBroker
from quantbot.decision.manager import TradeManager
from quantbot.decision.risk import RiskManager, TradePlan, Veto

SPEC = SymbolSpec(symbol="EURUSD", digits=5, point=1e-5, tick_size=1e-5, tick_value=1.0)


def make_signal(direction=Direction.LONG, confidence=0.7, regime=Regime.TRENDING) -> Signal:
    return Signal(
        instrument="EURUSD",
        timeframe="M15",
        ts=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        direction=direction,
        confidence=confidence,
        horizon_min=120,
        regime=regime,
    )


def evaluate(rm: RiskManager, signal: Signal, **kw):
    defaults = dict(
        equity=10_000.0,
        entry_price=1.10000,
        atr_value=0.0005,
        spec=SPEC,
        open_positions=[],
        realized_pnl_today=0.0,
        upcoming_events=None,
        now=pd.Timestamp("2026-01-01 12:00", tz="UTC"),
    )
    defaults.update(kw)
    return rm.evaluate(signal, **defaults)


# -- sizing ----------------------------------------------------------------


def test_position_size_risks_the_configured_fraction():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=1.0))
    volume, risk = rm.position_size(equity=10_000, risk_distance=0.0010, spec=SPEC)
    # 0.0010 = 100 ticks; at $1/tick/lot, $100 of risk needs 1.0 lot.
    assert volume == pytest.approx(1.0, abs=0.01)
    assert risk == pytest.approx(100.0, rel=0.02)


def test_position_size_respects_volume_step_and_minimum():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.01))
    volume, _ = rm.position_size(equity=100, risk_distance=0.0100, spec=SPEC)
    assert volume >= SPEC.volume_min
    assert round(volume / SPEC.volume_step) * SPEC.volume_step == pytest.approx(volume)


def test_stop_is_below_entry_for_long_and_above_for_short():
    rm = RiskManager(RiskConfig())
    sl, tp, dist = rm.stop_and_target(1.1, Direction.LONG, 0.0005, SPEC)
    assert sl < 1.1 < tp and dist > 0
    sl, tp, dist = rm.stop_and_target(1.1, Direction.SHORT, 0.0005, SPEC)
    assert tp < 1.1 < sl and dist > 0


# -- vetoes ----------------------------------------------------------------


def test_low_confidence_is_vetoed():
    rm = RiskManager(RiskConfig(min_confidence=0.6))
    result = evaluate(rm, make_signal(confidence=0.55))
    assert isinstance(result, Veto) and result.reason == "low_confidence"


def test_flat_signal_is_vetoed():
    rm = RiskManager(RiskConfig())
    assert isinstance(evaluate(rm, make_signal(direction=Direction.FLAT)), Veto)


def test_daily_loss_cap_blocks_further_trading():
    rm = RiskManager(RiskConfig(max_daily_loss_pct=2.0))
    result = evaluate(rm, make_signal(), realized_pnl_today=-250.0)
    assert isinstance(result, Veto) and result.reason == "daily_loss_cap"


def test_max_open_positions_blocks_new_entries():
    rm = RiskManager(RiskConfig(max_open_positions=1))
    pos = Position(1, "GBPUSD", Direction.LONG, 0.1, 1.25, None, None, datetime.now(timezone.utc))
    result = evaluate(rm, make_signal(), open_positions=[pos])
    assert isinstance(result, Veto) and result.reason == "max_positions"


def test_high_impact_news_window_is_a_hard_veto():
    rm = RiskManager(RiskConfig(news_veto_minutes=15))
    events = pd.DataFrame(
        {
            "name": ["NFP"],
            "impact": ["high"],
            "ts_utc": [pd.Timestamp("2026-01-01 12:05", tz="UTC")],
        }
    )
    result = evaluate(rm, make_signal(), upcoming_events=events)
    assert isinstance(result, Veto) and result.reason == "news_window"


def test_low_impact_event_does_not_veto():
    rm = RiskManager(RiskConfig(news_veto_minutes=15))
    events = pd.DataFrame(
        {
            "name": ["Retail Sales"],
            "impact": ["low"],
            "ts_utc": [pd.Timestamp("2026-01-01 12:05", tz="UTC")],
        }
    )
    assert isinstance(evaluate(rm, make_signal(), upcoming_events=events), TradePlan)


def test_poor_reward_risk_is_vetoed():
    rm = RiskManager(RiskConfig(sl_atr_mult=3.0, tp_atr_mult=1.0, min_rr=1.2))
    result = evaluate(rm, make_signal())
    assert isinstance(result, Veto) and result.reason == "poor_rr"


def test_valid_signal_produces_a_complete_plan():
    plan = evaluate(RiskManager(RiskConfig()), make_signal())
    assert isinstance(plan, TradePlan)
    assert plan.sl < plan.entry < plan.tp
    assert plan.volume > 0 and plan.rr >= 1.2
    # Risk never exceeds the budget; the leverage cap may put it under.
    assert 0 < plan.risk_amount <= 50.0 * 1.25


# -- stop-distance floors and the leverage cap ------------------------------


def test_stop_must_clear_the_spread():
    """An ATR stop narrower than the spread is taken out by the spread alone."""
    rm = RiskManager(RiskConfig(sl_atr_mult=1.2, min_stop_spread_mult=4.0, min_stop_points=0))
    tiny_atr = 0.00002  # 0.2 pips
    spread = 0.00015  # 1.5 pips
    sl, _tp, distance = rm.stop_and_target(1.10, Direction.LONG, tiny_atr, SPEC, spread=spread)
    assert distance >= spread * 4.0
    assert 1.10 - sl == pytest.approx(distance, abs=1e-9)


def test_absolute_minimum_stop_distance_is_enforced():
    rm = RiskManager(RiskConfig(sl_atr_mult=1.2, min_stop_points=80, min_stop_spread_mult=0))
    _sl, _tp, distance = rm.stop_and_target(1.10, Direction.LONG, 0.00002, SPEC)
    assert distance >= SPEC.point * 80


def test_widened_stop_keeps_the_intended_reward_risk():
    rm = RiskManager(RiskConfig(sl_atr_mult=1.2, tp_atr_mult=1.8, min_stop_points=200))
    sl, tp, distance = rm.stop_and_target(1.10, Direction.LONG, 0.00002, SPEC)
    rr = (tp - 1.10) / (1.10 - sl)
    assert rr == pytest.approx(1.8 / 1.2, rel=0.02)


def test_leverage_cap_shrinks_an_oversized_position():
    """A 3-pip stop met a 0.25% budget with 8 lots at 11x. Never again."""
    rm = RiskManager(
        RiskConfig(risk_per_trade_pct=0.25, max_position_leverage=3.0, min_stop_points=0,
                   min_stop_spread_mult=0)
    )
    plan = evaluate(rm, make_signal(), equity=100_000.0, atr_value=0.000025)
    assert isinstance(plan, TradePlan)
    notional = plan.volume * SPEC.contract_size * plan.entry
    assert notional / 100_000.0 <= 3.0 + 1e-6
    assert any("capped" in r for r in plan.reasons)


def test_normal_position_is_not_capped():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5, max_position_leverage=3.0))
    plan = evaluate(rm, make_signal(), equity=100_000.0, atr_value=0.0012)
    assert isinstance(plan, TradePlan)
    assert not any("capped" in r for r in plan.reasons)


# -- paper broker ----------------------------------------------------------


def test_paper_broker_fills_and_computes_pnl():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    broker.set_price("EURUSD", 1.10000, ts=datetime.now(timezone.utc))
    fill = broker.place_order("EURUSD", Direction.LONG, 1.0)
    assert fill.status.value == "open"

    broker.set_price("EURUSD", 1.10100)  # +100 ticks
    assert broker.equity() == pytest.approx(10_100.0, abs=1.0)

    broker.close_position(fill.ticket, reason="test")
    assert broker.balance() == pytest.approx(10_100.0, abs=1.0)
    assert broker.get_positions() == []


def test_stop_loss_triggers_on_bar_low_not_only_on_close():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    now = datetime.now(timezone.utc)
    broker.set_price("EURUSD", 1.10000, ts=now)
    fill = broker.place_order("EURUSD", Direction.LONG, 1.0, sl=1.09900, tp=1.10300)

    # A bar that dips through the stop but closes above it must still stop out.
    fills = broker.feed_bar("EURUSD", high=1.10050, low=1.09850, close=1.10020, ts=now)
    assert len(fills) == 1
    assert fills[0].reason == "stop_loss"
    assert broker.get_positions() == []
    assert broker.closed_trades[0]["exit_price"] == pytest.approx(1.09900)


def test_bar_spanning_both_barriers_assumes_the_stop_hit_first():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    now = datetime.now(timezone.utc)
    broker.set_price("EURUSD", 1.10000, ts=now)
    broker.place_order("EURUSD", Direction.LONG, 1.0, sl=1.09900, tp=1.10100)
    fills = broker.feed_bar("EURUSD", high=1.10200, low=1.09800, close=1.10000, ts=now)
    assert fills[0].reason == "stop_loss", "optimistic fills are how backtests lie"


def test_partial_close_leaves_the_remainder_open():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    broker.set_price("EURUSD", 1.10000, ts=datetime.now(timezone.utc))
    fill = broker.place_order("EURUSD", Direction.LONG, 1.0)
    broker.close_position(fill.ticket, volume=0.4, reason="partial_tp")
    positions = broker.get_positions()
    assert len(positions) == 1
    assert positions[0].volume == pytest.approx(0.6)


# -- trade manager ---------------------------------------------------------


def _managed_position(broker: PaperBroker, entry=1.10000, sl=1.09900):
    broker.register_spec(SPEC)
    broker.set_price("EURUSD", entry, ts=datetime.now(timezone.utc))
    fill = broker.place_order("EURUSD", Direction.LONG, 1.0, sl=sl, tp=1.10500)
    return fill.ticket


def test_stop_moves_past_breakeven_into_profit_after_one_r():
    """A stop at exact entry still loses the spread; the buffer locks a real gain."""
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    ticket = _managed_position(broker)
    cfg = RiskConfig(breakeven_at_r=1.0, breakeven_buffer_r=0.1, partial_tp_fraction=0.0)
    manager = TradeManager(cfg, broker)
    manager.register(ticket, 0.00100)

    broker.set_price("EURUSD", 1.10120)  # +1.2R
    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    sl = broker.get_positions()[0].sl
    assert sl == pytest.approx(1.10010)  # entry + 0.1R
    assert sl > 1.10000, "must be in profit, not merely scratch"


def test_breakeven_buffer_can_be_disabled():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    ticket = _managed_position(broker)
    cfg = RiskConfig(breakeven_at_r=1.0, breakeven_buffer_r=0.0, partial_tp_fraction=0.0)
    manager = TradeManager(cfg, broker)
    manager.register(ticket, 0.00100)
    broker.set_price("EURUSD", 1.10120)
    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    assert broker.get_positions()[0].sl == pytest.approx(1.10000)


def test_time_stop_closes_a_trade_that_went_nowhere():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    opened = datetime.now(timezone.utc) - timedelta(minutes=200)
    broker.set_price("EURUSD", 1.10000, ts=opened)
    fill = broker.place_order("EURUSD", Direction.LONG, 1.0, sl=1.09900, tp=1.10500)

    manager = TradeManager(
        RiskConfig(breakeven_at_r=1.0), broker, max_bars_in_trade=24, bar_minutes=5
    )
    manager.register(fill.ticket, 0.00100)
    broker.set_price("EURUSD", 1.10005)  # 200 min later, gone nowhere

    actions = manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    assert any("time stop" in a for a in actions)
    assert broker.get_positions() == []


def test_time_stop_spares_a_trade_that_is_working():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    opened = datetime.now(timezone.utc) - timedelta(minutes=200)
    broker.set_price("EURUSD", 1.10000, ts=opened)
    fill = broker.place_order("EURUSD", Direction.LONG, 1.0, sl=1.09900, tp=1.10500)

    manager = TradeManager(
        RiskConfig(breakeven_at_r=1.0, partial_tp_fraction=0.0), broker,
        max_bars_in_trade=24, bar_minutes=5,
    )
    manager.register(fill.ticket, 0.00100)
    broker.set_price("EURUSD", 1.10150)  # +1.5R — let the winner run

    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    assert len(broker.get_positions()) == 1


def test_a_mode_only_manages_positions_it_opened():
    """Scalp and swing run side by side; neither may touch the other's trades."""
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    broker.register_spec(SPEC)
    broker.set_price("EURUSD", 1.10000, ts=datetime.now(timezone.utc))
    mine = broker.place_order("EURUSD", Direction.LONG, 1.0, sl=1.09900, tp=1.10500)
    theirs = broker.place_order("EURUSD", Direction.LONG, 1.0, sl=1.09900, tp=1.10500)

    manager = TradeManager(RiskConfig(breakeven_at_r=1.0, partial_tp_fraction=0.0), broker)
    manager.register(mine.ticket, 0.00100)

    broker.set_price("EURUSD", 1.10120)
    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})

    by_ticket = {p.ticket: p for p in broker.get_positions()}
    assert by_ticket[mine.ticket].sl > 1.10000
    assert by_ticket[theirs.ticket].sl == pytest.approx(1.09900), "untouched"


def test_stop_is_never_loosened():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    ticket = _managed_position(broker)
    manager = TradeManager(RiskConfig(breakeven_at_r=1.0, partial_tp_fraction=0.0), broker)
    manager.register(ticket, 0.00100)

    broker.set_price("EURUSD", 1.10200)
    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    tightened = broker.get_positions()[0].sl

    broker.set_price("EURUSD", 1.10050)  # price falls back
    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    assert broker.get_positions()[0].sl >= tightened


def test_partial_take_profit_scales_out_once():
    broker = PaperBroker(balance=10_000, spread_points=0, commission_per_lot=0)
    ticket = _managed_position(broker)
    cfg = RiskConfig(partial_tp_at_r=1.5, partial_tp_fraction=0.5, trail_start_r=99, breakeven_at_r=99)
    manager = TradeManager(cfg, broker)
    manager.register(ticket, 0.00100)

    broker.set_price("EURUSD", 1.10200)  # +2R
    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    assert broker.get_positions()[0].volume == pytest.approx(0.5)

    manager.manage(broker.get_positions(), {"EURUSD": 0.0005})
    assert broker.get_positions()[0].volume == pytest.approx(0.5), "must not scale out twice"


# -- risk-budget and data-sanity guards ------------------------------------


def test_min_lot_risking_more_than_budget_is_vetoed():
    """Clamping volume up to volume_min must not silently exceed the budget."""
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5, max_risk_overshoot=1.25))
    # $100 account: budget is $0.50, but a 0.01 lot with a 90-tick stop risks
    # $0.90 — nearly 2x. The trade must be refused, not silently oversized.
    result = evaluate(rm, make_signal(), equity=100.0, atr_value=0.0005)
    assert isinstance(result, Veto) and result.reason == "risk_exceeds_budget"


def test_normal_account_is_not_blocked_by_the_budget_guard():
    rm = RiskManager(RiskConfig(risk_per_trade_pct=0.5, max_risk_overshoot=1.25))
    plan = evaluate(rm, make_signal(), equity=10_000.0, atr_value=0.0005)
    assert isinstance(plan, TradePlan)
    budget = 10_000 * 0.005
    assert plan.risk_amount <= budget * 1.25


def test_implausible_stop_distance_is_vetoed():
    """A broken ATR (mixed feeds, bad data) must not become a 25,000-pip stop."""
    rm = RiskManager(RiskConfig(max_sl_distance_pct=5.0))
    result = evaluate(rm, make_signal(), atr_value=0.14)  # ATR larger than price moves
    assert isinstance(result, Veto) and result.reason == "implausible_stop"


def test_realistic_atr_passes_the_sanity_bound():
    rm = RiskManager(RiskConfig(max_sl_distance_pct=5.0))
    assert isinstance(evaluate(rm, make_signal(), atr_value=0.0008), TradePlan)
