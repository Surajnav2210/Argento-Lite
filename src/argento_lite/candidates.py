"""Every allowed package on an integer weight grid.

Weights are integer units of WEIGHT_STEP, so sums are exact: with a 0.10 step a package is a
tuple of integers summing to 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np

from argento_lite.schemas import PolicyConfig


@dataclass(frozen=True)
class CandidateSet:
    """Row i of ``units`` holds the integer weight units per ticker of package i."""

    tickers: tuple[str, ...]
    grid_units: int
    units: np.ndarray
    cash_units: np.ndarray
    signatures: tuple[str, ...]

    @property
    def weights(self) -> np.ndarray:
        return self.units / self.grid_units

    @property
    def cash_weights(self) -> np.ndarray:
        return self.cash_units / self.grid_units

    @property
    def n_components(self) -> np.ndarray:
        return (self.units > 0).sum(axis=1)

    def __len__(self) -> int:
        return len(self.signatures)


def compositions(parts: int, total: int, min_units: int) -> Iterator[tuple[int, ...]]:
    """All ordered tuples of ``parts`` integers >= ``min_units`` summing to ``total``, in lexicographic order."""
    if parts == 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        if total >= min_units:
            yield (total,)
        return
    for first in range(min_units, total - min_units * (parts - 1) + 1):
        for rest in compositions(parts - 1, total - first, min_units):
            yield (first, *rest)


def make_signature(weights: dict[str, float], cash_weight: float) -> str:
    """Stable package id, e.g. ``"IEF:0.20|SPY:0.50|CASH:0.10"``."""
    parts = [f"{t}:{w:.2f}" for t, w in sorted(weights.items()) if w > 0]
    if cash_weight > 0:
        parts.append(f"CASH:{cash_weight:.2f}")
    return "|".join(parts)


def allowed_cash_units(n_components: int, policy: PolicyConfig) -> list[int]:
    """Single-ETF packages are fully invested; cash is only allowed alongside several ETFs."""
    if n_components == 1:
        return [0]
    return policy.cash_units


def enumerate_candidates(tickers: list[str], policy: PolicyConfig) -> CandidateSet:
    n = policy.grid_units
    m = policy.min_component_units
    index = {t: i for i, t in enumerate(tickers)}
    unit_rows: list[np.ndarray] = []
    cash_rows: list[int] = []
    sigs: list[str] = []

    for k in range(policy.MIN_PORTFOLIO_COMPONENTS, policy.MAX_PORTFOLIO_COMPONENTS + 1):
        for combo in combinations(tickers, k):
            for cash in allowed_cash_units(k, policy):
                for split in compositions(k, n - cash, m):
                    row = np.zeros(len(tickers), dtype=int)
                    for t, u in zip(combo, split):
                        row[index[t]] = u
                    unit_rows.append(row)
                    cash_rows.append(cash)
                    sigs.append(make_signature(dict(zip(combo, (u / n for u in split))), cash / n))

    units = np.vstack(unit_rows) if unit_rows else np.zeros((0, len(tickers)), dtype=int)
    return CandidateSet(
        tickers=tuple(tickers),
        grid_units=n,
        units=units,
        cash_units=np.asarray(cash_rows, dtype=int),
        signatures=tuple(sigs),
    )
