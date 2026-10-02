"""Risk targets and the forward-only nearest-point risk ladder."""

from __future__ import annotations

import numpy as np

from argento_lite.schemas import PolicyConfig


def risk_targets(policy: PolicyConfig) -> np.ndarray:
    return np.linspace(policy.RISK_SCALE_MIN, policy.RISK_SCALE_MAX, policy.RISK_LEVEL_COUNT)


def pick_nearest(
    risk: np.ndarray,
    score: np.ndarray,
    is_multi: np.ndarray,
    signatures: np.ndarray,
    target: float,
    start: int,
    tol: float,
) -> int:
    """Nearest point to ``target`` among positions >= ``start``.

    Tie-breaks in order: distance (within ``tol`` is a tie), lower risk, higher score,
    multi-portfolio, lexical signature.
    """
    idx = np.arange(start, len(risk))
    dist = np.abs(risk[idx] - target)
    tied = idx[dist <= dist.min() + tol]
    return int(min(tied, key=lambda i: (risk[i], -score[i], not is_multi[i], signatures[i])))


def select_ladder(
    risk,
    score,
    is_multi,
    signatures,
    targets,
    tol: float,
) -> list[tuple[int, float]]:
    """Frontier position and distance for each target, never moving backward.

    Inputs describe frontier points sorted by ascending risk. Adjacent levels may share a
    package but never step back to a less risky one.
    """
    r = np.asarray(risk, dtype=float)
    if r.size == 0:
        return []
    s = np.asarray(score, dtype=float)
    multi = np.asarray(is_multi, dtype=bool)
    sig = np.asarray(signatures, dtype=object)
    picks: list[tuple[int, float]] = []
    start = 0
    for t in np.asarray(targets, dtype=float):
        j = pick_nearest(r, s, multi, sig, float(t), start, tol)
        picks.append((j, float(abs(r[j] - t))))
        start = j
    return picks
