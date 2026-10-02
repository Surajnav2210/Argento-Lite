"""Holdings look-through: how much of a package is really one company.

A package holds ETFs and each ETF holds companies, so a company's exposure is the sum over ETFs
of (ETF weight in the package) x (company weight inside the ETF). ``H`` is the holdings matrix,
one row per ETF and one column per company; ETFs without company holdings are all-zero rows.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from argento_lite.schemas import CompanyIdentityConfig, HoldingsSnapshot


def holdings_matrix(
    snapshots: dict[str, HoldingsSnapshot],
    tickers: list[str],
    identity: CompanyIdentityConfig,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """ETF x company weights and a company-key -> display-name map.

    Fund and cash holdings (``identity.non_company``) are dropped, and share classes or
    cross-listings of one company (GOOGL + GOOG, TSM + 2330.TW) are merged into one column.
    """
    per_etf: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    for ticker in tickers:
        weights: dict[str, float] = defaultdict(float)
        snap = snapshots.get(ticker)
        for h in (snap.holdings if snap else []):
            if h.symbol in identity.non_company:
                continue
            alias = identity.aliases.get(h.symbol)
            key = alias.key if alias else h.symbol
            names.setdefault(key, alias.name if alias else h.name)
            weights[key] += h.weight
        per_etf[ticker] = dict(weights)

    companies = sorted({key for w in per_etf.values() for key in w})
    matrix = pd.DataFrame(0.0, index=tickers, columns=companies)
    for ticker, weights in per_etf.items():
        for key, w in weights.items():
            matrix.loc[ticker, key] = w
    return matrix, names


def max_exposure_all(weights: np.ndarray, H: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Largest single-company exposure of every package, and the company driving it ("" if none)."""
    n = len(weights)
    if H.shape[1] == 0:
        return np.zeros(n), np.full(n, "", dtype=object)
    exposure = np.asarray(weights) @ H.to_numpy()
    top = exposure.max(axis=1)
    driver = np.asarray(H.columns, dtype=object)[exposure.argmax(axis=1)]
    return top, np.where(top > 0, driver, "")


def package_exposure(weights: dict[str, float], H: pd.DataFrame, names: dict[str, str],
                     top_n: int = 10) -> pd.DataFrame:
    """Largest company exposures of one package: ``company``, total ``exposure``, then one column per ETF."""
    held = [t for t, w in weights.items() if w > 0]
    if H.shape[1] == 0 or not held:
        return pd.DataFrame(columns=["company", "exposure", *held])
    parts = H.loc[held].mul(pd.Series(weights)[held], axis=0)
    parts = parts.loc[:, parts.sum() > 0]
    total = parts.sum().sort_values(ascending=False, kind="stable").head(top_n)
    out = parts[total.index].T
    out.insert(0, "exposure", total)
    out.insert(0, "company", [names.get(key, key) for key in out.index])
    return out.reset_index(drop=True)


def known_share(weights: dict[str, float], snapshots: dict[str, HoldingsSnapshot]) -> float:
    """Share of the package's equity-ETF weight covered by known holdings (1.0 = complete)."""
    equity = {t: w for t, w in weights.items() if snapshots.get(t) and snapshots[t].holdings}
    total = sum(equity.values())
    if total == 0:
        return 1.0
    return sum(w * snapshots[t].coverage for t, w in equity.items()) / total


def overlap_matrix(H: pd.DataFrame) -> pd.DataFrame:
    """ETF x ETF overlap: per shared company the smaller of its two weights, summed."""
    a = H.to_numpy()
    shared = np.minimum(a[:, None, :], a[None, :, :]).sum(axis=2)
    return pd.DataFrame(shared, index=H.index, columns=H.index)
