"""Cash hurdle, dominance and the Pareto frontier."""

from __future__ import annotations

import numpy as np


def excess_scores(selection_returns, cash_hurdle: float) -> np.ndarray:
    return np.asarray(selection_returns, dtype=float) - cash_hurdle


def passes_cash_hurdle(excess) -> np.ndarray:
    return np.asarray(excess, dtype=float) >= 0.0


def pareto_frontier(risk, score, signatures, tol: float) -> np.ndarray:
    """Positions of non-dominated points, by ascending risk.

    Sweeps in (risk asc, score desc, signature asc) order and keeps a point only if its score
    beats every lower-risk point by more than ``tol``. Ties keep the lexically first signature.
    """
    r = np.asarray(risk, dtype=float)
    s = np.asarray(score, dtype=float)
    sig = np.asarray(signatures, dtype=str)
    if r.size == 0:
        return np.zeros(0, dtype=int)
    order = np.lexsort((sig, -s, r))
    keep: list[int] = []
    best = -np.inf
    for i in order:
        if s[i] > best + tol:
            keep.append(int(i))
            best = s[i]
    return np.asarray(keep, dtype=int)
