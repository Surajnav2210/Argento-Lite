"""Covariance estimates, package volatility and selection risk.

The full-history covariance uses the common history of the whole universe, not the
Argento's ragged-history method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def covariance(returns: pd.DataFrame) -> np.ndarray:
    """Sample covariance (daily units) of a returns window with no missing values."""
    if returns.isna().any().any():
        raise ValueError("covariance window must not contain missing returns")
    return returns.cov(ddof=1).to_numpy()


def package_vol(
    weights: np.ndarray,
    cov: np.ndarray,
    annualization: int,
    cash_weights: np.ndarray | None = None,
    cash_vol: float = 0.0,
) -> np.ndarray:
    """Annualized volatility sqrt(k * w' Σ w) per row of ``weights``.

    Cash is uncorrelated with every ETF and adds (w_cash * cash_vol)^2 to annual variance.
    """
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    var = annualization * np.einsum("ij,jk,ik->i", w, cov, w)
    if cash_weights is not None and cash_vol:
        var = var + (np.asarray(cash_weights, dtype=float) * cash_vol) ** 2
    return np.sqrt(np.clip(var, 0.0, None))


def selection_risk(vol_full, vol_recent) -> tuple[np.ndarray, np.ndarray]:
    """max(full, recent) volatility, plus which one bound. Ties are labelled "full"."""
    vf = np.asarray(vol_full, dtype=float)
    vr = np.asarray(vol_recent, dtype=float)
    risk = np.maximum(vf, vr)
    binding = np.where(vr > vf, "recent", "full")
    return risk, binding
