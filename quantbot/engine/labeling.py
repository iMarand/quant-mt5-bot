"""Triple-barrier labeling — turns candles into supervised targets.

A fixed "did price go up in N bars" label ignores that a move has to survive a
stop first. The triple barrier (López de Prado) sets an ATR-scaled profit
barrier, an ATR-scaled stop barrier and a time barrier, and labels by whichever
is touched first — the same question the risk layer actually asks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.indicators import atr

LABEL_UP, LABEL_FLAT, LABEL_DOWN = 1, 0, -1
#: LightGBM needs contiguous class ids; keep the mapping in one place.
CLASS_ORDER = [LABEL_DOWN, LABEL_FLAT, LABEL_UP]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def triple_barrier_labels(
    candles: pd.DataFrame,
    horizon_bars: int = 8,
    atr_mult: float = 1.0,
    atr_period: int = 14,
) -> pd.DataFrame:
    """Label each bar by which barrier its *next* `horizon_bars` bars hit first.

    Returns columns: label (-1/0/1), fwd_return, bars_held, barrier_width.
    Entry is assumed at the close of the labeled bar.
    """
    if candles.empty:
        return pd.DataFrame(columns=["label", "fwd_return", "bars_held", "barrier_width"])

    close = candles["close"].to_numpy(dtype=float)
    high = candles["high"].to_numpy(dtype=float)
    low = candles["low"].to_numpy(dtype=float)
    width = (atr(candles, atr_period) * atr_mult).to_numpy(dtype=float)

    n = len(close)
    labels = np.full(n, np.nan)
    fwd = np.full(n, np.nan)
    held = np.full(n, np.nan)

    for i in range(n):
        w = width[i]
        if not np.isfinite(w) or w <= 0 or i + 1 >= n:
            continue
        end = min(i + horizon_bars, n - 1)
        entry = close[i]
        up_barrier, dn_barrier = entry + w, entry - w
        outcome, j = LABEL_FLAT, end
        for k in range(i + 1, end + 1):
            hit_up = high[k] >= up_barrier
            hit_dn = low[k] <= dn_barrier
            if hit_up and hit_dn:
                # Both barriers inside one bar: intrabar path is unknown, so
                # call it flat rather than inventing a favourable ordering.
                outcome, j = LABEL_FLAT, k
                break
            if hit_up:
                outcome, j = LABEL_UP, k
                break
            if hit_dn:
                outcome, j = LABEL_DOWN, k
                break
        labels[i] = outcome
        fwd[i] = (close[j] - entry) / entry
        held[i] = j - i

    return pd.DataFrame(
        {
            "label": labels,
            "fwd_return": fwd,
            "bars_held": held,
            "barrier_width": width,
        },
        index=candles.index,
    )


def purge_and_embargo(
    index: pd.DatetimeIndex,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    horizon_bars: int,
    embargo_bars: int,
) -> np.ndarray:
    """Drop train rows whose label window overlaps the test block, plus embargo.

    Without this, a label built from bars that also appear in the test fold
    leaks the answer and every walk-forward metric is inflated.
    """
    if len(test_idx) == 0:
        return train_idx
    lo, hi = test_idx.min(), test_idx.max()
    bad_lo = lo - horizon_bars
    bad_hi = hi + embargo_bars
    return train_idx[(train_idx < bad_lo) | (train_idx > bad_hi)]


def walk_forward_splits(
    n: int, n_splits: int = 5, min_train: int = 500
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits — always train on the past, test on the future."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    if n <= min_train + n_splits:
        return splits
    test_size = max(1, (n - min_train) // n_splits)
    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_end = min(train_end + test_size, n)
        if train_end >= n or test_end <= train_end:
            break
        splits.append((np.arange(0, train_end), np.arange(train_end, test_end)))
    return splits
