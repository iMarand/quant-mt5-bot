"""Per-setup reliability learned from the journal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantbot.config import StrategyConfig
from quantbot.contracts import Direction, Regime, Signal
from quantbot.learning.reliability import SetupReliability
from quantbot.storage import Database
from quantbot.strategy import Setup, StrategyBook


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "rel.db")


def journal(db, setup: str, n: int, wins: int, start=None):
    """Record n scored predictions for `setup`, `wins` of them correct."""
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        sig = Signal(
            instrument="EURUSD",
            timeframe="M15",
            ts=start + timedelta(minutes=15 * i),
            direction=Direction.LONG,
            confidence=0.7,
            horizon_min=120,
            regime=Regime.TRENDING,
            setup=setup,
        )
        pid = db.record_prediction(sig)
        db.record_outcome(pid, realized_return=0.001, label=1, correct=(i < wins))


def test_no_journal_means_no_opinions(db):
    rel = SetupReliability.from_journal(db)
    assert rel.stats == {}
    assert rel.weight("breakout") == 1.0


def test_small_sample_is_not_treated_as_proof(db):
    """3-for-4 is not a 75% setup."""
    journal(db, "breakout", n=4, wins=3)
    rel = SetupReliability.from_journal(db, min_samples=20)
    stat = rel.stats["breakout"]
    assert stat.raw_accuracy == 0.75
    assert stat.shrunk_accuracy < 0.6, "must be pulled toward the prior"
    assert rel.weight("breakout") == 1.0, "below min_samples -> no opinion"


def test_a_reliably_winning_setup_is_boosted(db):
    journal(db, "breakout", n=200, wins=140)  # 70%
    rel = SetupReliability.from_journal(db, min_samples=20)
    assert rel.weight("breakout") > 1.0


def test_a_reliably_losing_setup_is_damped(db):
    journal(db, "mean_reversion", n=200, wins=60)  # 30%
    rel = SetupReliability.from_journal(db, min_samples=20)
    assert rel.weight("mean_reversion") < 1.0


def test_weights_are_bounded_in_both_directions(db):
    journal(db, "perfect", n=500, wins=500)
    journal(db, "hopeless", n=500, wins=0, start=datetime(2026, 6, 1, tzinfo=timezone.utc))
    rel = SetupReliability.from_journal(db, min_samples=20, min_weight=0.6, max_weight=1.4)
    assert rel.weight("perfect") == pytest.approx(1.4)
    assert rel.weight("hopeless") == pytest.approx(0.6)


def test_composite_triggers_credit_every_setup(db):
    journal(db, "breakout+volume_surge", n=100, wins=70)
    rel = SetupReliability.from_journal(db, min_samples=20)
    assert "breakout" in rel.stats and "volume_surge" in rel.stats
    assert rel.stats["breakout"].n == 100
    assert rel.weight("volume_surge") > 1.0


def test_reliability_scales_setup_quality_in_the_book(db):
    journal(db, "breakout", n=300, wins=210)
    rel = SetupReliability.from_journal(db, min_samples=20)

    class AlwaysFires:
        name = "breakout"
        enabled = True
        weight = 1.0

        def evaluate(self, ctx):
            return Setup("breakout", Direction.LONG, 0.5)

    cfg = StrategyConfig(min_setup_quality=0.0)
    plain = StrategyBook(cfg, strategies=[AlwaysFires()])
    weighted = StrategyBook(cfg, strategies=[AlwaysFires()], reliability=rel)

    import pandas as pd

    from quantbot.strategy.base import StrategyContext

    ctx = StrategyContext(pd.Series({"M15_close": 1.1}), None, "M15")
    assert weighted.triggered_setups(ctx)[0].quality > plain.triggered_setups(ctx)[0].quality


def test_reliability_can_be_switched_off(db):
    journal(db, "breakout", n=300, wins=210)
    rel = SetupReliability.from_journal(db, min_samples=20)

    class AlwaysFires:
        name = "breakout"
        enabled = True
        weight = 1.0

        def evaluate(self, ctx):
            return Setup("breakout", Direction.LONG, 0.5)

    import pandas as pd

    from quantbot.strategy.base import StrategyContext

    cfg = StrategyConfig(min_setup_quality=0.0, use_setup_reliability=False)
    book = StrategyBook(cfg, strategies=[AlwaysFires()], reliability=rel)
    ctx = StrategyContext(pd.Series({"M15_close": 1.1}), None, "M15")
    assert book.triggered_setups(ctx)[0].quality == pytest.approx(0.5)


def test_setup_quality_columns_are_signed_by_direction():
    import pandas as pd

    from quantbot.learning.reliability import setup_quality_columns

    class Longs:
        name = "up"
        enabled = True
        weight = 1.0

        def evaluate(self, ctx):
            return Setup("up", Direction.LONG, 0.8)

    class Shorts:
        name = "down"
        enabled = True
        weight = 1.0

        def evaluate(self, ctx):
            return Setup("down", Direction.SHORT, 0.6)

    class Book:
        strategies = [Longs(), Shorts()]

    df = pd.DataFrame(
        {"M15_close": [1.1, 1.2]},
        index=pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC"),
    )
    cols = setup_quality_columns(Book(), df, "M15")
    assert cols["sq_up"].iloc[0] == pytest.approx(0.8)
    assert cols["sq_down"].iloc[0] == pytest.approx(-0.6), "shorts must be negative"


# -- seeding weights from a counterfactual study ----------------------------


def _study_rows(setup, n, wins, session="london"):
    import pandas as pd

    return pd.DataFrame(
        {
            "setup": [setup] * n,
            "session": [session] * n,
            "won": [1] * wins + [0] * (n - wins),
            "r_multiple": [0.5] * wins + [-0.5] * (n - wins),
        }
    )


def test_study_keys_are_plain_setup_names():
    """A key of "('breakout',)" would never match a setup lookup."""
    from quantbot.learning.reliability import from_counterfactual

    rel = from_counterfactual(_study_rows("breakout", 500, 300))
    assert "breakout" in rel.stats
    assert rel.weight("breakout") > 1.0


def test_study_seeds_are_more_conservative_than_journal_seeds():
    """Simulations should nudge, not dictate — they use an optimistic cost model."""
    import pandas as pd

    from quantbot.learning.reliability import SetupStats, from_counterfactual

    rel = from_counterfactual(_study_rows("breakout", 1000, 600))  # 60% raw
    assert rel.stats["breakout"].shrunk_accuracy < 0.60


def test_study_below_min_samples_gives_no_opinion():
    from quantbot.learning.reliability import from_counterfactual

    rel = from_counterfactual(_study_rows("rare_setup", 20, 18), min_samples=100)
    assert rel.weight("rare_setup") == 1.0


def test_study_can_key_by_session():
    import pandas as pd

    from quantbot.learning.reliability import from_counterfactual

    df = pd.concat([
        _study_rows("breakout", 400, 260, session="london"),
        _study_rows("breakout", 400, 140, session="tokyo"),
    ])
    rel = from_counterfactual(df, by_session=True)
    assert rel.weight("breakout@london") > 1.0
    assert rel.weight("breakout@tokyo") < 1.0


def test_empty_study_yields_no_opinions():
    import pandas as pd

    from quantbot.learning.reliability import from_counterfactual

    assert from_counterfactual(pd.DataFrame()).stats == {}
