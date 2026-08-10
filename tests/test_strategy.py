"""Strategy layer: setups decide, the model only assists."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbot.config import StrategyConfig
from quantbot.contracts import Direction
from quantbot.strategy import (
    Breakout,
    MeanReversion,
    NewsReaction,
    Setup,
    StrategyBook,
    StrategyContext,
    SupportResistanceRejection,
    TrendPullback,
    build_strategies,
)

BASE = "M15"


def row(**kw) -> pd.Series:
    """A neutral bar; keyword args override individual features."""
    base = {
        "M15_rsi_14": 50.0,
        "M15_macd_hist_norm": 0.0,
        "M15_adx": 15.0,
        "M15_atr_percentile": 0.5,
        "M15_bb_pct": 0.5,
        "M15_bb_width": 0.01,
        "M15_range_position": 0.5,
        "M15_dist_to_high": 0.01,
        "M15_dist_to_low": 0.01,
        "M15_upper_wick_ratio": 0.1,
        "M15_lower_wick_ratio": 0.1,
        "M15_pat_pin_bull": 0.0,
        "M15_pat_pin_bear": 0.0,
        "M15_pat_bullish_engulfing": 0.0,
        "M15_pat_bearish_engulfing": 0.0,
        "M15_ret_5": 0.0,
        "H1_ema_fast_slow": 0.0,
        "H1_adx": 10.0,
        "minutes_since_last_high": 1440.0,
        "minutes_to_next_high": 1440.0,
        "last_surprise_signed": 0.0,
    }
    base.update(kw)
    return pd.Series(base)


def ctx(now: pd.Series, prev: pd.Series | None = None) -> StrategyContext:
    return StrategyContext(now, prev, BASE)


# -- individual setups -----------------------------------------------------


def test_neutral_bar_triggers_nothing():
    book = StrategyBook(StrategyConfig())
    assert book.triggered_setups(ctx(row(), row())) == []


def test_trend_pullback_fires_on_uptrend_dip_with_momentum_turning():
    s = TrendPullback()
    now = row(H1_ema_fast_slow=0.0015, H1_adx=28.0, M15_rsi_14=45.0, M15_macd_hist_norm=0.0001)
    prev = row(M15_macd_hist_norm=-0.0002)
    setup = s.evaluate(ctx(now, prev))
    assert setup is not None
    assert setup.direction is Direction.LONG
    assert setup.quality > 0


def test_trend_pullback_requires_momentum_to_be_turning():
    """A pullback that is still deepening is not an entry."""
    s = TrendPullback()
    now = row(H1_ema_fast_slow=0.0015, H1_adx=28.0, M15_rsi_14=45.0, M15_macd_hist_norm=-0.0004)
    prev = row(M15_macd_hist_norm=-0.0001)  # momentum still falling
    assert s.evaluate(ctx(now, prev)) is None


def test_trend_pullback_ignored_without_a_trend():
    s = TrendPullback(min_adx=20.0)
    now = row(H1_ema_fast_slow=0.0015, H1_adx=12.0, M15_rsi_14=45.0, M15_macd_hist_norm=0.0001)
    assert s.evaluate(ctx(now, row(M15_macd_hist_norm=-0.0002))) is None


def test_breakout_must_be_fresh():
    s = Breakout()
    fresh = s.evaluate(
        ctx(row(M15_range_position=0.97, M15_adx=25.0, M15_atr_percentile=0.7),
            row(M15_range_position=0.80))
    )
    assert fresh is not None and fresh.direction is Direction.LONG

    # Already broken out last bar — chasing, not breaking.
    stale = s.evaluate(
        ctx(row(M15_range_position=0.99, M15_adx=25.0, M15_atr_percentile=0.7),
            row(M15_range_position=0.97))
    )
    assert stale is None


def test_mean_reversion_needs_a_range_and_a_rejection_candle():
    s = MeanReversion()
    # Extended but trending -> no trade (don't fade a trend).
    assert s.evaluate(ctx(row(M15_bb_pct=0.97, M15_adx=35.0, M15_pat_pin_bear=1.0))) is None
    # Extended, ranging, but no rejection candle -> no trade.
    assert s.evaluate(ctx(row(M15_bb_pct=0.97, M15_adx=12.0))) is None
    # All three conditions -> short.
    setup = s.evaluate(ctx(row(M15_bb_pct=0.97, M15_adx=12.0, M15_pat_pin_bear=1.0, M15_rsi_14=72.0)))
    assert setup is not None and setup.direction is Direction.SHORT


def test_sr_rejection_needs_proximity_and_a_wick():
    s = SupportResistanceRejection(proximity=0.002)
    setup = s.evaluate(
        ctx(row(M15_dist_to_low=0.0005, M15_pat_pin_bull=1.0, M15_lower_wick_ratio=0.6))
    )
    assert setup is not None and setup.direction is Direction.LONG
    # Far from any level -> nothing.
    assert s.evaluate(ctx(row(M15_dist_to_low=0.02, M15_pat_pin_bull=1.0))) is None


# -- news setups -----------------------------------------------------------


def test_news_reaction_fires_after_a_confirmed_surprise():
    s = NewsReaction(min_surprise_z=0.5)
    setup = s.evaluate(
        ctx(row(minutes_since_last_high=10.0, last_surprise_signed=1.4,
                M15_macd_hist_norm=0.0002, M15_ret_5=0.001))
    )
    assert setup is not None
    assert setup.direction is Direction.LONG
    assert setup.is_news


def test_news_reaction_refuses_before_the_release():
    """Holding into a print is a volatility bet, not a directional edge."""
    s = NewsReaction()
    assert s.evaluate(
        ctx(row(minutes_to_next_high=5.0, minutes_since_last_high=1440.0,
                last_surprise_signed=2.0))
    ) is None


def test_news_reaction_requires_price_confirmation():
    s = NewsReaction(min_surprise_z=0.5, require_confirmation=True)
    # Surprise says long, price says down -> stand aside.
    assert s.evaluate(
        ctx(row(minutes_since_last_high=10.0, last_surprise_signed=1.4,
                M15_macd_hist_norm=-0.0002, M15_ret_5=-0.001))
    ) is None


def test_news_reaction_ignores_a_small_surprise():
    s = NewsReaction(min_surprise_z=1.0)
    assert s.evaluate(
        ctx(row(minutes_since_last_high=10.0, last_surprise_signed=0.2,
                M15_macd_hist_norm=0.0002, M15_ret_5=0.001))
    ) is None


# -- the book --------------------------------------------------------------


def test_no_trigger_means_no_trade():
    book = StrategyBook(StrategyConfig())
    decision = book.combine([])
    assert not decision.triggered
    assert decision.direction is Direction.FLAT


def test_conflicting_setups_cancel_rather_than_net_out():
    book = StrategyBook(StrategyConfig())
    decision = book.combine(
        [
            Setup("a", Direction.LONG, 0.9),
            Setup("b", Direction.SHORT, 0.4),
        ]
    )
    assert not decision.triggered
    assert decision.vetoed_by == "conflict"


def test_confluence_raises_confidence():
    cfg = StrategyConfig(min_confluence=1)
    book = StrategyBook(cfg)
    one = book.combine([Setup("a", Direction.LONG, 0.5)])
    two = book.combine([Setup("a", Direction.LONG, 0.5), Setup("b", Direction.LONG, 0.5)])
    assert two.confidence > one.confidence


def test_min_confluence_blocks_a_lone_setup():
    book = StrategyBook(StrategyConfig(min_confluence=2))
    decision = book.combine([Setup("a", Direction.LONG, 0.9)])
    assert not decision.triggered
    assert decision.vetoed_by == "insufficient_confluence"


def test_require_news_rejects_a_purely_technical_trigger():
    book = StrategyBook(StrategyConfig(require_news=True))
    technical = book.combine([Setup("breakout", Direction.LONG, 0.8, tags={"technical"})])
    assert not technical.triggered
    assert technical.vetoed_by == "no_news_trigger"

    event = book.combine([Setup("news_reaction", Direction.LONG, 0.8, tags={"news"})])
    assert event.triggered


# -- the model is an assistant, not a decider -------------------------------


def test_model_cannot_create_a_trade_when_nothing_triggered():
    """The whole point of strategy-first: no setup, no trade, however sure the model is."""
    book = StrategyBook(StrategyConfig())
    flat = book.combine([])
    # Model screams "up" with 99% confidence.
    result = book.apply_model(flat, np.array([0.005, 0.005, 0.99]))
    assert not result.triggered


def test_model_cannot_flip_the_direction():
    book = StrategyBook(StrategyConfig(model_veto_below=0.0))
    long_decision = book.combine([Setup("a", Direction.LONG, 0.8)])
    result = book.apply_model(long_decision, np.array([0.9, 0.05, 0.05]))
    assert result.direction is Direction.LONG


def test_model_agreement_boosts_and_disagreement_damps():
    book = StrategyBook(StrategyConfig(model_assist_weight=0.4, model_veto_below=0.0))
    baseline = book.combine([Setup("a", Direction.LONG, 0.6)]).confidence

    agree = book.apply_model(
        book.combine([Setup("a", Direction.LONG, 0.6)]), np.array([0.2, 0.1, 0.8])
    )
    disagree = book.apply_model(
        book.combine([Setup("a", Direction.LONG, 0.6)]), np.array([0.45, 0.1, 0.45])
    )
    assert agree.confidence > baseline
    assert disagree.confidence <= baseline


def test_model_vetoes_a_setup_it_strongly_disagrees_with():
    book = StrategyBook(StrategyConfig(model_veto_below=0.4))
    decision = book.apply_model(
        book.combine([Setup("a", Direction.LONG, 0.8)]), np.array([0.8, 0.05, 0.15])
    )
    assert not decision.triggered
    assert decision.vetoed_by == "model_disagrees"


def test_model_role_off_ignores_the_model_entirely():
    book = StrategyBook(StrategyConfig(model_role="off"))
    decision = book.combine([Setup("a", Direction.LONG, 0.8)])
    before = decision.confidence
    after = book.apply_model(decision, np.array([0.9, 0.05, 0.05]))
    assert after.triggered and after.confidence == before


# -- config ----------------------------------------------------------------


def test_disabled_setups_are_not_built():
    cfg = StrategyConfig(setups={"breakout": {"enabled": False}, "sr_rejection": {"enabled": True}})
    names = [s.name for s in build_strategies(cfg)]
    assert names == ["sr_rejection"]


def test_unknown_setup_name_is_rejected_loudly():
    with pytest.raises(ValueError, match="unknown setup"):
        build_strategies(StrategyConfig(setups={"does_not_exist": {}}))


def test_bad_setup_parameter_is_rejected_loudly():
    with pytest.raises(ValueError, match="bad parameters"):
        build_strategies(StrategyConfig(setups={"breakout": {"nonsense_param": 1}}))


def test_a_broken_setup_does_not_kill_the_cycle():
    class Exploding(TrendPullback):
        name = "exploding"

        def evaluate(self, ctx):
            raise RuntimeError("boom")

    book = StrategyBook(StrategyConfig(), strategies=[Exploding(), SupportResistanceRejection()])
    setups = book.triggered_setups(
        ctx(row(M15_dist_to_low=0.0005, M15_pat_pin_bull=1.0, M15_lower_wick_ratio=0.6))
    )
    assert [s.name for s in setups] == ["sr_rejection"]


# -- new setup families -----------------------------------------------------


def test_ema_cross_requires_a_fresh_cross():
    from quantbot.strategy import EmaCross

    s = EmaCross(require_htf_agreement=False, min_adx=0)
    fresh = s.evaluate(ctx(row(M15_ema_fast_slow=0.0004), row(M15_ema_fast_slow=-0.0002)))
    assert fresh is not None and fresh.direction is Direction.LONG
    # Already crossed last bar -> not a trigger.
    stale = s.evaluate(ctx(row(M15_ema_fast_slow=0.0008), row(M15_ema_fast_slow=0.0004)))
    assert stale is None


def test_ema_cross_respects_the_higher_timeframe():
    from quantbot.strategy import EmaCross

    s = EmaCross(require_htf_agreement=True, min_adx=0)
    against = s.evaluate(
        ctx(row(M15_ema_fast_slow=0.0004, H1_ema_fast_slow=-0.002),
            row(M15_ema_fast_slow=-0.0002))
    )
    assert against is None


def test_ema_ribbon_fires_on_a_fresh_pullback_to_the_ribbon():
    from quantbot.strategy import EmaRibbon

    s = EmaRibbon(min_adx=20, max_stretch=0.004, touch_distance=0.0008)
    touched = s.evaluate(
        ctx(row(M15_ema_10_dist=0.0004, M15_ema_20_dist=0.002, M15_ema_50_dist=0.003, M15_adx=30),
            row(M15_ema_10_dist=0.0020))  # was away from the ribbon last bar
    )
    assert touched is not None and touched.direction is Direction.LONG


def test_ema_ribbon_does_not_fire_every_bar_of_a_trend():
    """Stacked EMAs are a state, not an event — firing on it would trigger
    on nearly every bar of a trend."""
    from quantbot.strategy import EmaRibbon

    s = EmaRibbon(min_adx=20, max_stretch=0.004, touch_distance=0.0008)
    # Already at the ribbon last bar too -> not a fresh touch.
    assert s.evaluate(
        ctx(row(M15_ema_10_dist=0.0004, M15_ema_20_dist=0.002, M15_ema_50_dist=0.003, M15_adx=30),
            row(M15_ema_10_dist=0.0005))
    ) is None
    # Stacked but stretched far above -> chasing, no trade.
    assert s.evaluate(
        ctx(row(M15_ema_10_dist=0.02, M15_ema_20_dist=0.03, M15_ema_50_dist=0.04, M15_adx=30),
            row(M15_ema_10_dist=0.03))
    ) is None


def test_volume_surge_needs_flow_and_price_to_agree():
    from quantbot.strategy import VolumeSurge

    s = VolumeSurge(min_obv_slope=0.1, min_body_ratio=0.5, min_atr_percentile=0.4)
    agree = s.evaluate(ctx(row(M15_obv_slope=0.5, M15_body_ratio=0.7, M15_ret_1=0.001,
                               M15_atr_percentile=0.6)))
    assert agree is not None and agree.direction is Direction.LONG
    # Flow up, price down -> churn, no trade.
    assert s.evaluate(ctx(row(M15_obv_slope=0.5, M15_body_ratio=0.7, M15_ret_1=-0.001,
                              M15_atr_percentile=0.6))) is None


def test_divergence_reversal_will_not_fade_a_strong_trend():
    from quantbot.strategy import DivergenceReversal

    s = DivergenceReversal(max_adx=28)
    trending = s.evaluate(
        ctx(row(M15_range_position=0.95, M15_rsi_14=60, M15_macd_hist_norm=0.0001,
                M15_adx=40, M15_pat_pin_bear=1.0),
            row(M15_macd_hist_norm=0.0003))
    )
    assert trending is None


def test_price_action_is_indicator_free():
    from quantbot.strategy import PriceAction

    s = PriceAction(min_body_ratio=0.4)
    # Only candle features present — no RSI/ADX/EMA at all.
    bar = pd.Series({
        "M15_pat_bullish_engulfing": 1.0, "M15_pat_bearish_engulfing": 0.0,
        "M15_pat_pin_bull": 0.0, "M15_pat_pin_bear": 0.0,
        "M15_body_ratio": 0.8, "M15_lower_wick_ratio": 0.1, "M15_upper_wick_ratio": 0.1,
    })
    setup = s.evaluate(StrategyContext(bar, None, BASE))
    assert setup is not None and setup.direction is Direction.LONG


def test_session_open_range_only_fires_near_a_session_open():
    from quantbot.strategy import SessionOpenRange

    s = SessionOpenRange(open_hours=(7.0,), window_hours=3.0, threshold=0.9)
    near = s.evaluate(ctx(row(M15_hour=8.0, M15_range_position=0.95, M15_atr_percentile=0.6)))
    assert near is not None and near.direction is Direction.LONG
    far = s.evaluate(ctx(row(M15_hour=18.0, M15_range_position=0.95, M15_atr_percentile=0.6)))
    assert far is None


# -- the AI master switch ---------------------------------------------------


def test_use_ai_false_disables_the_model_entirely():
    cfg = StrategyConfig(use_ai=False, model_role="assistant")
    assert cfg.effective_model_role() == "off"
    book = StrategyBook(cfg)
    decision = book.combine([Setup("a", Direction.LONG, 0.8)])
    before = decision.confidence
    after = book.apply_model(decision, np.array([0.9, 0.05, 0.05]))
    assert after.triggered and after.confidence == before


# -- entry confirmation -----------------------------------------------------


def test_momentum_confirmation_waits_for_price_to_move_in_favour():
    from datetime import datetime, timezone

    from quantbot.strategy import ConfirmationPolicy

    policy = ConfirmationPolicy(mode="momentum", confirm_atr_mult=0.5, max_wait_bars=3)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    policy.register("k", "EURUSD", Direction.LONG, 1.1000, 0.0010, "breakout", 0.7, now)

    ok, _ = policy.check("k", 1.10010)  # +1 pip, needs +5
    assert not ok
    ok, note = policy.check("k", 1.10060)  # +6 pips
    assert ok and "confirmed" in note


def test_pending_entry_expires_rather_than_waiting_forever():
    from datetime import datetime, timezone

    from quantbot.strategy import ConfirmationPolicy

    policy = ConfirmationPolicy(mode="momentum", confirm_atr_mult=0.5, max_wait_bars=2)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    policy.register("k", "EURUSD", Direction.LONG, 1.1000, 0.0010, "breakout", 0.7, now)
    policy.check("k", 1.1000)
    policy.check("k", 1.1000)
    ok, note = policy.check("k", 1.1000)
    assert not ok and "expired" in note
    assert policy.pending_for("k") is None


def test_pullback_confirmation_abandons_a_runaway_move():
    from datetime import datetime, timezone

    from quantbot.strategy import ConfirmationPolicy

    policy = ConfirmationPolicy(mode="pullback", confirm_atr_mult=0.5, max_wait_bars=5)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    policy.register("k", "EURUSD", Direction.LONG, 1.1000, 0.0010, "breakout", 0.7, now)
    ok, note = policy.check("k", 1.1030)  # ran 30 pips without us
    assert not ok and "ran away" in note


def test_confirmation_off_means_no_pending_state():
    from quantbot.strategy import ConfirmationPolicy

    assert not ConfirmationPolicy(mode="off").enabled
