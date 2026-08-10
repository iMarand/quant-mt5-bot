from .base import Broker
from .paper import PaperBroker

__all__ = ["Broker", "PaperBroker", "make_broker"]


def make_broker(cfg) -> Broker:
    """Factory — the only place that knows which implementations exist."""
    kind = cfg.broker.kind.lower()
    if kind == "paper":
        return PaperBroker(
            balance=cfg.broker.paper_balance,
            spread_points=cfg.broker.paper_spread_points,
            commission_per_lot=cfg.broker.paper_commission_per_lot,
        )
    if kind == "mt5":
        from .mt5_broker import MT5Broker

        return MT5Broker(
            login=cfg.broker.mt5_login,
            password=cfg.broker.mt5_password,
            server=cfg.broker.mt5_server,
            path=cfg.broker.mt5_path,
            magic=cfg.broker.magic,
            deviation=cfg.broker.deviation,
            allow_live=cfg.broker.allow_live,
        )
    raise ValueError(f"unknown broker kind {cfg.broker.kind!r} (expected 'paper' or 'mt5')")
