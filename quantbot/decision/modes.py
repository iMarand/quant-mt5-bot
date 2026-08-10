"""Trading modes — scalp and swing as genuinely separate configurations.

They differ in far more than timeframe: a scalp needs a tighter stop, a smaller
risk budget, an earlier move to breakeven, a time-based exit for trades that go
nowhere, and it only belongs in the London/NY overlap. Expressing that as one
set of parameters with a different bar size would be wrong.

A `ModeRuntime` is a deep copy of the base config with the mode's overrides
applied, so every downstream component (features, risk, trade manager) keeps
working unchanged against a plain `Config`.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from ..config import Config, ModeConfig
from .sessions import SessionPolicy

log = logging.getLogger(__name__)


@dataclass
class ModeRuntime:
    name: str
    cfg: Config
    session_policy: SessionPolicy
    max_bars_in_trade: int = 0

    @property
    def base_timeframe(self) -> str:
        return self.cfg.data.base_timeframe

    def __str__(self) -> str:
        return (
            f"{self.name}(tf={self.base_timeframe}, "
            f"min_conf={self.cfg.risk.min_confidence:.2f}, "
            f"risk={self.cfg.risk.risk_per_trade_pct}%)"
        )


def build_mode(base: Config, name: str, mode: ModeConfig) -> ModeRuntime:
    """Derive a full Config for one mode."""
    cfg = copy.deepcopy(base)

    cfg.data.base_timeframe = mode.base_timeframe
    if mode.timeframes:
        cfg.data.timeframes = list(mode.timeframes)
    # The base timeframe must be present, or feature building fails outright.
    if mode.base_timeframe not in cfg.data.timeframes:
        cfg.data.timeframes = sorted(
            set(cfg.data.timeframes) | {mode.base_timeframe},
            key=lambda tf: _tf_order(tf),
        )
    cfg.model.horizon_bars = mode.horizon_bars

    for attr in (
        "min_confidence",
        "sl_atr_mult",
        "tp_atr_mult",
        "risk_per_trade_pct",
        "breakeven_at_r",
        "trail_start_r",
        "trail_atr_mult",
        "partial_tp_at_r",
    ):
        value = getattr(mode, attr)
        if value is not None:
            setattr(cfg.risk, attr, value)
    if mode.max_positions is not None:
        cfg.risk.max_open_positions = mode.max_positions

    # Restrict the strategy book to this mode's setups.
    if mode.setups:
        unknown = set(mode.setups) - set(cfg.strategy.setups)
        if unknown:
            raise ValueError(f"mode {name!r} lists unknown setups: {sorted(unknown)}")
        cfg.strategy.setups = {
            k: v for k, v in cfg.strategy.setups.items() if k in mode.setups
        }

    policy = SessionPolicy(allowed=list(mode.sessions))
    return ModeRuntime(
        name=name,
        cfg=cfg,
        session_policy=policy,
        max_bars_in_trade=mode.max_bars_in_trade,
    )


def build_modes(cfg: Config) -> list[ModeRuntime]:
    """All enabled modes. Falls back to a single default mode when none are set."""
    out: list[ModeRuntime] = []
    for name, mode in (cfg.modes or {}).items():
        if not mode.enabled:
            continue
        try:
            out.append(build_mode(cfg, name, mode))
        except Exception as exc:
            log.error("mode %s is misconfigured and was skipped: %s", name, exc)
            raise
    if not out:
        log.warning("no modes enabled — falling back to the top-level data/risk config")
        out.append(
            ModeRuntime(name="default", cfg=copy.deepcopy(cfg), session_policy=SessionPolicy())
        )
    return out


_TF_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


def _tf_order(tf: str) -> int:
    return _TF_ORDER.index(tf) if tf in _TF_ORDER else len(_TF_ORDER)
