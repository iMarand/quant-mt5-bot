"""Sessions, the ordered pre-trade gate, and trading modes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from quantbot.config import Config, ModeConfig, PreTradeConfig
from quantbot.decision.modes import build_mode, build_modes
from quantbot.decision.pretrade import PreTradeGate
from quantbot.decision.sessions import (
    SESSIONS,
    SessionPolicy,
    active_sessions,
    sessions_for_currencies,
)


def utc(y=2026, m=6, d=10, hh=14, mm=0):
    """Default: Wednesday 14:00 UTC — inside the London/NY overlap."""
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def events(rows):
    return pd.DataFrame(rows)


def high_impact(when, currency="USD", name="NFP"):
    return events([{"currency": currency, "impact": "high", "name": name, "ts_utc": pd.Timestamp(when)}])


# -- sessions --------------------------------------------------------------


def test_london_is_open_midday_and_shut_overnight():
    london = SESSIONS["london"]
    assert london.contains(utc(hh=12))
    assert not london.contains(utc(hh=3))


def test_tokyo_session_wraps_midnight_correctly():
    tokyo = SESSIONS["tokyo"]
    assert tokyo.contains(utc(hh=2))
    assert not tokyo.contains(utc(hh=15))


def test_sydney_wraps_across_midnight():
    sydney = SESSIONS["sydney"]
    assert sydney.contains(utc(hh=23))
    assert sydney.contains(utc(hh=2))
    assert not sydney.contains(utc(hh=12))


def test_sessions_derived_from_the_pair():
    names = sessions_for_currencies(["EUR", "USD"])
    assert "london" in names and "newyork" in names
    assert "tokyo" not in names
    assert "tokyo" in sessions_for_currencies(["USD", "JPY"])


def test_dst_shifts_the_london_window_by_an_hour():
    london = SESSIONS["london"]
    summer_start, _ = london.window(utc(m=7))
    winter_start, _ = london.window(utc(m=1))
    assert summer_start == winter_start - 1


def test_overlap_is_the_busiest_window():
    assert "london_ny_overlap" in active_sessions(utc(hh=14))
    assert "london_ny_overlap" not in active_sessions(utc(hh=9))


# -- session policy --------------------------------------------------------


def test_weekend_is_closed():
    policy = SessionPolicy()
    saturday = utc(d=13)  # 2026-06-13 is a Saturday
    ok, reason = policy.evaluate(saturday, ["EUR", "USD"])
    assert not ok and "closed" in reason


def test_sunday_before_sydney_open_is_closed_but_after_is_open():
    policy = SessionPolicy(allowed=["sydney"], avoid_rollover=False)
    sunday = utc(d=14)  # 2026-06-14 is a Sunday
    assert not policy.evaluate(sunday.replace(hour=12), ["AUD"])[0]
    assert policy.evaluate(sunday.replace(hour=22), ["AUD"])[0]


def test_friday_cutoff_avoids_weekend_gap_risk():
    policy = SessionPolicy(friday_cutoff_hour=19.0, avoid_rollover=False)
    friday = utc(d=12)  # 2026-06-12 is a Friday
    assert policy.evaluate(friday.replace(hour=14), ["EUR", "USD"])[0]
    ok, reason = policy.evaluate(friday.replace(hour=20), ["EUR", "USD"])
    assert not ok and "gap" in reason


def test_rollover_window_is_avoided():
    policy = SessionPolicy(allowed=["sydney"], rollover_hour=21.0, rollover_buffer_min=30)
    ok, reason = policy.evaluate(utc(hh=21, mm=10), ["AUD"])
    assert not ok and "rollover" in reason


def test_pair_outside_its_session_is_refused():
    policy = SessionPolicy(avoid_rollover=False)
    ok, reason = policy.evaluate(utc(hh=3), ["EUR", "USD"])
    assert not ok and "outside" in reason


# -- the ordered gate ------------------------------------------------------


def gate(**overrides):
    cfg = PreTradeConfig(**overrides)
    return PreTradeGate(cfg, SessionPolicy(avoid_rollover=False))


def fresh_calendar(now):
    return high_impact(now + timedelta(days=2))


def test_gate_passes_when_everything_is_clear():
    now = utc()
    result = gate().check("EURUSD", now=now, events=fresh_calendar(now), last_bar_ts=now)
    assert result.allowed
    assert result.stage == "clear"


def test_calendar_is_checked_first():
    now = utc()
    result = gate().check("EURUSD", now=now, events=None, last_bar_ts=now)
    assert not result.allowed
    assert result.stage == "calendar"


def test_stale_calendar_blocks_trading():
    """A calendar that stopped updating would let us trade blind into a release."""
    now = utc()
    old = high_impact(now - timedelta(days=30))
    result = gate(max_calendar_age_days=10).check("EURUSD", now=now, events=old, last_bar_ts=now)
    assert not result.allowed and result.stage == "calendar"


def test_calendar_can_be_made_optional():
    now = utc()
    result = gate(require_calendar=False).check("EURUSD", now=now, events=None, last_bar_ts=now)
    assert result.allowed


def test_clock_blocks_the_weekend():
    saturday = utc(d=13)
    result = gate().check(
        "EURUSD", now=saturday, events=fresh_calendar(saturday), last_bar_ts=saturday
    )
    assert not result.allowed and result.stage == "clock"


def test_stale_price_data_blocks_trading():
    now = utc()
    result = gate(max_bar_age_multiple=4).check(
        "EURUSD",
        now=now,
        events=fresh_calendar(now),
        last_bar_ts=now - timedelta(hours=6),
        base_timeframe="M15",
    )
    assert not result.allowed and result.stage == "clock"


def test_blackout_date_blocks_trading():
    now = utc()
    result = gate(blackout_dates=[now.date().isoformat()]).check(
        "EURUSD", now=now, events=fresh_calendar(now), last_bar_ts=now
    )
    assert not result.allowed and result.stage == "clock"


def test_imminent_high_impact_release_blocks_new_positions():
    now = utc()
    soon = high_impact(now + timedelta(minutes=10))
    result = gate(block_before_news_min=20).check(
        "EURUSD", now=now, events=soon, last_bar_ts=now
    )
    assert not result.allowed and result.stage == "news"
    assert "NFP" in result.reason


def test_release_for_another_currency_does_not_block():
    now = utc()
    other = high_impact(now + timedelta(minutes=10), currency="JPY")
    result = gate(block_before_news_min=20).check(
        "EURUSD", now=now, events=other, last_bar_ts=now
    )
    assert result.allowed


def test_post_release_window_is_flagged_as_news_active():
    now = utc()
    just_out = high_impact(now - timedelta(minutes=5))
    result = gate(news_window_after_min=45).check(
        "EURUSD", now=now, events=just_out, last_bar_ts=now
    )
    assert result.allowed and result.news_active


def test_news_window_can_override_a_closed_session():
    """A high-impact print moves the pair whoever's session it is."""
    now = utc(hh=3)  # outside London/NY
    just_out = high_impact(now - timedelta(minutes=5))
    result = PreTradeGate(
        PreTradeConfig(news_overrides_session=True), SessionPolicy(avoid_rollover=False)
    ).check("EURUSD", now=now, events=just_out, last_bar_ts=now)
    assert result.allowed

    strict = PreTradeGate(
        PreTradeConfig(news_overrides_session=False), SessionPolicy(avoid_rollover=False)
    ).check("EURUSD", now=now, events=just_out, last_bar_ts=now)
    assert not strict.allowed and strict.stage == "session"


def test_session_is_the_last_gate():
    now = utc(hh=3)
    result = gate().check("EURUSD", now=now, events=fresh_calendar(now), last_bar_ts=now)
    assert not result.allowed and result.stage == "session"


# -- modes -----------------------------------------------------------------


def test_scalp_and_swing_get_genuinely_different_settings():
    cfg = Config()
    modes = {m.name: m for m in build_modes(cfg)}
    assert set(modes) == {"swing", "scalp"}

    scalp, swing = modes["scalp"], modes["swing"]
    assert scalp.base_timeframe == "M5" and swing.base_timeframe == "H1"
    assert scalp.cfg.risk.sl_atr_mult < swing.cfg.risk.sl_atr_mult
    assert scalp.cfg.risk.risk_per_trade_pct < swing.cfg.risk.risk_per_trade_pct
    assert scalp.cfg.risk.min_confidence > swing.cfg.risk.min_confidence
    assert scalp.cfg.risk.breakeven_at_r < swing.cfg.risk.breakeven_at_r


def test_mode_restricts_the_strategy_book():
    cfg = Config()
    scalp = build_mode(cfg, "scalp", cfg.modes["scalp"])
    assert "trend_pullback" not in scalp.cfg.strategy.setups
    assert "sr_rejection" in scalp.cfg.strategy.setups


def test_mode_base_timeframe_is_always_present_in_its_timeframes():
    cfg = Config()
    mode = build_mode(cfg, "odd", ModeConfig(base_timeframe="M30", timeframes=["H1", "H4"]))
    assert "M30" in mode.cfg.data.timeframes


def test_mode_with_unknown_setup_is_rejected():
    cfg = Config()
    with pytest.raises(ValueError, match="unknown setups"):
        build_mode(cfg, "bad", ModeConfig(setups=["not_a_setup"]))


def test_disabled_mode_is_skipped():
    cfg = Config()
    cfg.modes["scalp"].enabled = False
    assert [m.name for m in build_modes(cfg)] == ["swing"]


def test_mode_sessions_become_its_session_policy():
    cfg = Config()
    scalp = build_mode(cfg, "scalp", cfg.modes["scalp"])
    assert scalp.session_policy.allowed == ["london_ny_overlap"]
    assert not scalp.session_policy.evaluate(utc(hh=9), ["EUR", "USD"])[0]
    assert scalp.session_policy.evaluate(utc(hh=14), ["EUR", "USD"])[0]
