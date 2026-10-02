"""Return estimates and the asymmetric recency adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from argento_lite.schemas import PolicyConfig


def annualized_mean(returns: pd.Series, annualization: int) -> float:
    clean = returns.dropna()
    if clean.empty:
        raise ValueError(f"no returns for {returns.name!r}")
    return float(clean.mean() * annualization)


def recency_adjustment(delta, alpha_pos: float, alpha_neg: float):
    """alpha_pos * max(delta, 0) + alpha_neg * min(delta, 0): improvements count less than declines."""
    d = np.asarray(delta, dtype=float)
    out = alpha_pos * np.maximum(d, 0.0) + alpha_neg * np.minimum(d, 0.0)
    return float(out) if out.ndim == 0 else out


def common_window(returns: pd.DataFrame) -> pd.DataFrame:
    """Rows where every ETF has a return."""
    common = returns.dropna(how="any")
    if common.empty:
        raise ValueError("the universe has no common history")
    return common


def recent_window(returns: pd.DataFrame, days: int) -> pd.DataFrame:
    common = common_window(returns)
    if len(common) < days:
        raise ValueError(f"common history has {len(common)} days, fewer than RECENT_WINDOW_DAYS={days}")
    return common.iloc[-days:]


def return_stats(returns: pd.DataFrame, policy: PolicyConfig) -> pd.DataFrame:
    """Per ETF: mu_full over its own history, mu_recent over the common last window, and the blend."""
    k = policy.ANNUALIZATION_FACTOR
    recent = recent_window(returns, policy.RECENT_WINDOW_DAYS)
    rows = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()
        mu_f = annualized_mean(series, k)
        mu_r = annualized_mean(recent[ticker], k)
        delta = mu_r - mu_f
        adj = recency_adjustment(delta, policy.ALPHA_POS, policy.ALPHA_NEG)
        rows.append({
            "ticker": ticker,
            "first_date": series.index[0].date(),
            "n_obs": len(series),
            "mu_full": mu_f,
            "mu_recent": mu_r,
            "delta": delta,
            "adjustment": adj,
            "mu_selection": mu_f + adj,
        })
    return pd.DataFrame(rows).set_index("ticker")


def package_selection_return(
    weights: np.ndarray, cash_weights: np.ndarray, mu_selection: np.ndarray, cash_hurdle: float
) -> np.ndarray:
    """Weighted selection returns per row of ``weights``, with cash earning the hurdle."""
    return np.asarray(weights) @ np.asarray(mu_selection) + np.asarray(cash_weights) * cash_hurdle
