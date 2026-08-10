"""Configuration: one YAML file + env overrides, validated into dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class DataConfig:
    symbols: list[str] = field(default_factory=lambda: ["EURUSD"])
    timeframes: list[str] = field(default_factory=lambda: ["M5", "M15", "M30", "H1", "H4", "D1"])
    base_timeframe: str = "M15"
    history_bars: int = 5000
    correlated_symbols: list[str] = field(default_factory=list)


@dataclass
class CalendarConfig:
    enabled: bool = True
    currencies: list[str] = field(default_factory=lambda: ["USD", "EUR"])
    cache_minutes: int = 30
    #: Public weekly JSON feed published by Forex Factory / FairEconomy.
    forexfactory_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    user_agent: str = "quantbot/0.1 (research; contact: set-me-in-config)"
    #: Investing.com scraping violates their ToS; enable only with a licensed
    #: API key of your own (architecture §3.4, compliance-by-design).
    investing_enabled: bool = False
    investing_api_key: str | None = None
    request_min_interval_s: float = 2.0


@dataclass
class BrokerConfig:
    kind: str = "paper"  # "paper" | "mt5"
    #: Refuse to trade a live (non-demo) MT5 account unless explicitly allowed.
    allow_live: bool = False
    mt5_login: int | None = None
    mt5_password: str | None = None
    mt5_server: str | None = None
    mt5_path: str | None = None
    magic: int = 990101
    deviation: int = 20
    paper_balance: float = 10_000.0
    paper_leverage: float = 100.0
    paper_spread_points: float = 12.0
    paper_commission_per_lot: float = 3.5


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.5  # % of equity risked per trade
    max_daily_loss_pct: float = 3.0
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    max_total_risk_pct: float = 2.0
    min_confidence: float = 0.58
    atr_period: int = 14
    sl_atr_mult: float = 1.8
    tp_atr_mult: float = 2.7
    min_rr: float = 1.2
    #: Clamping volume up to the broker's minimum lot can risk more than
    #: risk_per_trade_pct. Reject rather than silently over-risk beyond this
    #: multiple of the budget.
    max_risk_overshoot: float = 1.25
    #: Sanity bound on stop distance as a % of price. A 5% stop is already
    #: enormous for FX; anything past it means the ATR estimate is broken.
    max_sl_distance_pct: float = 5.0
    #: A stop must clear the spread by this multiple. An ATR-derived stop can
    #: land 3 pips away on a quiet M5 bar, which the spread alone will take out
    #: — and because size is inverse to stop distance, it also produces an
    #: enormous position.
    min_stop_spread_mult: float = 4.0
    #: Absolute floor on stop distance, in points.
    min_stop_points: float = 80.0
    #: Hard cap on position notional as a multiple of equity. Backstop against
    #: a tiny stop turning 0.25% risk into a 10x-leveraged position.
    max_position_leverage: float = 3.0
    # Dynamic stop management (addendum §B)
    breakeven_at_r: float = 1.0
    #: Move the stop to entry + this much R, not to entry exactly. A stop at
    #: exact entry still loses the spread and commission; a small buffer means
    #: "stopped out" becomes a small win instead of a scratch.
    breakeven_buffer_r: float = 0.1
    trail_start_r: float = 1.2
    trail_atr_mult: float = 1.5
    partial_tp_at_r: float = 1.5
    partial_tp_fraction: float = 0.5
    # News veto
    news_veto_minutes: int = 15
    news_veto_impact: str = "high"


@dataclass
class ModelConfig:
    horizon_bars: int = 8  # label horizon on the base timeframe
    label_atr_mult: float = 1.0  # triple-barrier width in ATR units
    min_train_rows: int = 800
    n_splits: int = 5  # walk-forward folds
    embargo_bars: int = 24
    search_trials: int = 20
    registry_dir: str = "artifacts/models"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 40,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_estimators": 300,
            "verbose": -1,
        }
    )


@dataclass
class PreTradeConfig:
    """Ordered gates that run before any setup is evaluated."""

    require_calendar: bool = True
    #: Refuse to trade a pair with no calendar entries at all for its currencies.
    require_relevant_calendar: bool = False
    max_calendar_age_days: float = 10.0
    #: How many bar-lengths old the newest bar may be before data is "stale".
    max_bar_age_multiple: float = 4.0
    #: No new positions in this window *before* a high-impact release.
    block_before_news_min: float = 20.0
    #: How long after a release counts as the tradeable news window.
    news_window_after_min: float = 45.0
    #: A big release moves a pair whoever's session it is.
    news_overrides_session: bool = True
    year_end_blackout_from_day: int = 24
    blackout_dates: list[str] = field(default_factory=list)


@dataclass
class ModeConfig:
    """One trading style. Scalping and swing differ in far more than timeframe."""

    enabled: bool = True
    base_timeframe: str = "M15"
    #: Timeframes this mode needs for its multi-timeframe context.
    timeframes: list[str] = field(default_factory=lambda: ["M5", "M15", "M30", "H1", "H4", "D1"])
    horizon_bars: int = 8
    #: Session names to restrict to; empty derives them from the pair.
    sessions: list[str] = field(default_factory=list)
    #: Setup names active in this mode; empty means all enabled setups.
    setups: list[str] = field(default_factory=list)
    # Per-mode risk overrides (fall back to RiskConfig when None).
    min_confidence: float | None = None
    sl_atr_mult: float | None = None
    tp_atr_mult: float | None = None
    risk_per_trade_pct: float | None = None
    max_positions: int | None = None
    breakeven_at_r: float | None = None
    trail_start_r: float | None = None
    trail_atr_mult: float | None = None
    partial_tp_at_r: float | None = None
    #: Close scalps that have gone nowhere after this many bars (0 = never).
    max_bars_in_trade: int = 0


@dataclass
class StrategyConfig:
    """Strategy-first decisioning. Setups decide; the model only assists."""

    #: "assistant" — model can damp/boost/veto a triggered setup, never create one.
    #: "off"       — pure rules, no model involvement at all.
    model_role: str = "assistant"
    #: How much the model may scale confidence, e.g. 0.4 = up to +/-40%.
    model_assist_weight: float = 0.4
    #: Veto when the model's agreement with the setup falls below this.
    model_veto_below: float = 0.38

    #: Drop a setup whose rule match is weaker than this.
    min_setup_quality: float = 0.25
    #: How many agreeing setups are required to trade.
    min_confluence: int = 1
    #: Extra conviction per additional agreeing setup (applied sublinearly).
    confluence_bonus: float = 0.15
    #: Only trade when an event setup fired. Off by default: technical setups
    #: are legitimate triggers too.
    require_news: bool = False

    #: Confidence assigned to a minimum-conviction trigger, and to a perfect one.
    base_confidence: float = 0.55
    max_confidence: float = 0.90

    setups: dict[str, Any] = field(
        default_factory=lambda: {
            "trend_pullback": {"enabled": True, "htf": "H1", "min_adx": 20.0},
            "breakout": {"enabled": True, "threshold": 0.95, "min_adx": 18.0},
            "mean_reversion": {"enabled": True, "max_adx": 20.0, "band": 0.92},
            "sr_rejection": {"enabled": True, "proximity": 0.0015},
            "news_reaction": {"enabled": True, "min_surprise_z": 0.6},
            "news_breakout": {"enabled": True, "threshold": 0.9},
        }
    )


@dataclass
class GateConfig:
    """Pre-committed promotion bar (architecture §8.3). Set before you start."""

    min_trades: int = 200
    min_directional_accuracy: float = 0.53
    min_profit_factor: float = 1.2
    min_sharpe: float = 0.8
    max_drawdown_pct: float = 15.0
    max_calibration_error: float = 0.08
    min_regimes_passing: int = 2


@dataclass
class OpsConfig:
    interval_minutes: int = 15
    event_lookahead_minutes: int = 5
    event_followup_minutes: int = 3
    drift_psi_threshold: float = 0.25
    feed_gap_tolerance_bars: int = 3
    log_level: str = "INFO"


@dataclass
class Config:
    db_path: str = "artifacts/quantbot.db"
    dry_run: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    pretrade: PreTradeConfig = field(default_factory=PreTradeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    modes: dict[str, ModeConfig] = field(
        default_factory=lambda: {
            "swing": ModeConfig(
                base_timeframe="H1",
                timeframes=["M15", "M30", "H1", "H4", "D1"],
                horizon_bars=12,
                sessions=["london", "newyork"],
                setups=["trend_pullback", "breakout", "news_reaction", "news_breakout"],
                min_confidence=0.60,
                sl_atr_mult=2.0,
                tp_atr_mult=3.5,
                risk_per_trade_pct=0.5,
                max_positions=2,
                breakeven_at_r=1.0,
                trail_start_r=1.5,
            ),
            "scalp": ModeConfig(
                base_timeframe="M5",
                timeframes=["M5", "M15", "M30", "H1"],
                horizon_bars=6,
                sessions=["london_ny_overlap"],
                setups=["sr_rejection", "mean_reversion", "breakout", "news_breakout"],
                min_confidence=0.66,
                sl_atr_mult=1.2,
                tp_atr_mult=1.8,
                risk_per_trade_pct=0.25,
                max_positions=1,
                breakeven_at_r=0.6,
                trail_start_r=0.8,
                trail_atr_mult=1.0,
                partial_tp_at_r=1.0,
                max_bars_in_trade=24,
            ),
        }
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    ops: OpsConfig = field(default_factory=OpsConfig)

    @property
    def db_file(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else ROOT / p

    @property
    def registry_path(self) -> Path:
        p = Path(self.model.registry_dir)
        return p if p.is_absolute() else ROOT / p


def _build(cls: type, data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    # `from __future__ import annotations` makes field.type a string, so resolve.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in known:
            raise ValueError(f"unknown config key {cls.__name__}.{key}")
        ftype = hints[key]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        elif isinstance(value, dict) and _dict_value_type(ftype) is not None:
            # e.g. modes: dict[str, ModeConfig] — build each entry.
            inner = _dict_value_type(ftype)
            kwargs[key] = {
                k: _build(inner, v) if isinstance(v, dict) else v for k, v in value.items()
            }
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _dict_value_type(ftype: Any) -> type | None:
    """Value dataclass of a `dict[str, SomeDataclass]` annotation, else None."""
    if get_origin(ftype) is not dict:
        return None
    args = get_args(ftype)
    if len(args) == 2 and is_dataclass(args[1]):
        return args[1]
    return None


_ENV_PREFIX = "QUANTBOT_"


def load_dotenv(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Read a `.env` file into the environment.

    Deliberately dependency-free and deliberately *not* overriding variables
    already set in the real environment — a value exported in the shell should
    win over a file, so a one-off override doesn't require editing the file.
    """
    path = Path(path) if path else ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip matched surrounding quotes; leave inner content untouched so
        # passwords containing #, $ or spaces survive intact.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif "#" in value:
            value = value.split("#", 1)[0].strip()
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _apply_env(cfg: Config) -> Config:
    """Secrets belong in env, not in the YAML you might commit."""
    env_map = {
        f"{_ENV_PREFIX}MT5_LOGIN": ("broker", "mt5_login", int),
        f"{_ENV_PREFIX}MT5_PASSWORD": ("broker", "mt5_password", str),
        f"{_ENV_PREFIX}MT5_SERVER": ("broker", "mt5_server", str),
        f"{_ENV_PREFIX}MT5_PATH": ("broker", "mt5_path", str),
        f"{_ENV_PREFIX}BROKER": ("broker", "kind", str),
        f"{_ENV_PREFIX}INVESTING_API_KEY": ("calendar", "investing_api_key", str),
    }
    for env_key, (section, attr, cast) in env_map.items():
        raw = os.environ.get(env_key)
        if raw:
            setattr(getattr(cfg, section), attr, cast(raw))
    return cfg


def load_config(path: str | Path | None = None, env_file: str | Path | None = None) -> Config:
    # Secrets come from .env / the environment, never from the YAML — the YAML
    # is meant to be committable.
    load_dotenv(env_file)
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _build(Config, raw)
    return _apply_env(cfg)
