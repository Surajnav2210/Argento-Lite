"""Dollar feasibility: every holding must meet its minimum investment."""

from __future__ import annotations

import numpy as np


def package_minimums(units: np.ndarray, minimums: np.ndarray, grid_units: int) -> np.ndarray:
    """Smallest feasible account per package: max over held components of ceil(min_i / w_i).

    ``units`` are integer weight units, so min_i / w_i = min_i * grid_units / units_i.
    """
    u = np.atleast_2d(np.asarray(units))
    mins = np.asarray(minimums, dtype=float)
    held = u > 0
    with np.errstate(divide="ignore"):
        per_component = np.where(held, np.ceil(mins * grid_units / np.where(held, u, 1)), 0.0)
    return per_component.max(axis=1)


def feasible_mask(units: np.ndarray, minimums: np.ndarray, grid_units: int, amount: float) -> np.ndarray:
    """True where every held component satisfies amount * w_i >= min_i.

    Compared as amount * units_i >= min_i * grid_units to avoid float rounding.
    """
    u = np.atleast_2d(np.asarray(units))
    mins = np.asarray(minimums, dtype=float)
    ok = (u == 0) | (amount * u >= mins * grid_units)
    return ok.all(axis=1)
