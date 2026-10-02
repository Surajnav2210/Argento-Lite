"""The pipeline: config + prices -> scored candidates -> frontier -> risk ladder.

``build_catalog`` runs once and computes everything independent of the account amount.
``ladder_for_amount`` runs per amount:

    affordable_candidates -> [within_cap] -> frontier_table -> build_rungs

With a cap, the frontier is rebuilt from the packages that satisfy it, not trimmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from argento_lite.candidates import CandidateSet, enumerate_candidates
from argento_lite.config import CACHE_DIR, PROJECT_ROOT
from argento_lite.data.holdings import load_all_holdings
from argento_lite.data.prices import apply_history_starts, daily_returns, load_prices
from argento_lite.feasibility import feasible_mask, package_minimums
from argento_lite.frontier import excess_scores, pareto_frontier, passes_cash_hurdle
from argento_lite.ladder import risk_targets, select_ladder
from argento_lite.lookthrough import holdings_matrix, max_exposure_all
from argento_lite.returns import common_window, package_selection_return, recent_window, return_stats
from argento_lite.risk import covariance, package_vol, selection_risk
from argento_lite.schemas import AppConfig, HoldingsSnapshot, Ladder, LadderRung, PackageSummary

WEIGHT_PREFIX = "w:"
HOLDINGS_DIR = PROJECT_ROOT / "data" / "holdings"
CAP_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Catalog:
    """Everything computed once per (universe, policy)."""

    config: AppConfig
    stats: pd.DataFrame
    candidates: CandidateSet
    table: pd.DataFrame
    holdings: dict[str, HoldingsSnapshot]
    exposure: pd.DataFrame
    company_names: dict[str, str]
    targets: np.ndarray

    @property
    def tickers(self) -> list[str]:
        return list(self.candidates.tickers)

    @property
    def minimums(self) -> np.ndarray:
        mins = self.config.universe.minimums
        return np.array([mins[t] for t in self.tickers], dtype=float)


def fetch_universe_prices(config: AppConfig, cache_dir: Path = CACHE_DIR, refresh: bool = False,
                          allow_fetch: bool = True) -> pd.DataFrame:
    uni = config.universe
    return load_prices(uni.tickers, uni.price_start, uni.price_end, cache_dir,
                       refresh=refresh, allow_fetch=allow_fetch)


def score_candidates(cands: CandidateSet, mu_selection: np.ndarray, cov_full: np.ndarray,
                     cov_recent: np.ndarray, minimums: np.ndarray, config: AppConfig) -> pd.DataFrame:
    """One row per candidate: return, both volatilities, selection risk, hurdle flag, package minimum."""
    p = config.policy
    w, cash = cands.weights, cands.cash_weights
    selection_return = package_selection_return(w, cash, mu_selection, p.CASH_HURDLE)
    vol_full = package_vol(w, cov_full, p.ANNUALIZATION_FACTOR, cash, p.CASH_VOLATILITY)
    vol_recent = package_vol(w, cov_recent, p.ANNUALIZATION_FACTOR, cash, p.CASH_VOLATILITY)
    risk, binding = selection_risk(vol_full, vol_recent)
    excess = excess_scores(selection_return, p.CASH_HURDLE)

    table = pd.DataFrame({
        "signature": cands.signatures,
        "n_components": cands.n_components,
        "cash_weight": cash,
        "selection_return": selection_return,
        "excess_score": excess,
        "vol_full": vol_full,
        "vol_recent": vol_recent,
        "selection_risk": risk,
        "risk_binding": binding,
        "package_minimum": package_minimums(cands.units, minimums, cands.grid_units),
        "passes_hurdle": passes_cash_hurdle(excess),
    })
    weights = pd.DataFrame(w, columns=[WEIGHT_PREFIX + t for t in cands.tickers])
    return pd.concat([table, weights], axis=1)


def build_catalog(config: AppConfig, prices: pd.DataFrame | None = None,
                  holdings: dict[str, HoldingsSnapshot] | None = None,
                  cache_dir: Path = CACHE_DIR, allow_fetch: bool = True) -> Catalog:
    """Run the amount-independent steps once. ``prices`` and ``holdings`` default to the cache."""
    uni, p = config.universe, config.policy
    if prices is None:
        prices = fetch_universe_prices(config, cache_dir, allow_fetch=allow_fetch)
    prices = apply_history_starts(prices, uni.history_starts)

    returns = daily_returns(prices)[uni.tickers]
    common = common_window(returns)
    recent = recent_window(returns, p.RECENT_WINDOW_DAYS)

    stats = return_stats(returns, p)
    cov_full, cov_recent = covariance(common), covariance(recent)
    cands = enumerate_candidates(uni.tickers, p)
    minimums = np.array([uni.minimums[t] for t in uni.tickers], dtype=float)
    table = score_candidates(cands, stats["mu_selection"].to_numpy(), cov_full, cov_recent,
                             minimums, config)

    if holdings is None:
        holdings = load_all_holdings(uni.tickers, HOLDINGS_DIR, cache_dir, allow_fetch=allow_fetch)
    exposure, company_names = holdings_matrix(holdings, uni.tickers, config.companies)
    table["max_exposure"], table["max_company"] = max_exposure_all(cands.weights, exposure)

    return Catalog(
        config=config,
        stats=stats,
        candidates=cands,
        table=table,
        holdings=holdings,
        exposure=exposure,
        company_names=company_names,
        targets=risk_targets(p),
    )


def affordable_candidates(catalog: Catalog, amount: float) -> pd.DataFrame:
    """Candidates that beat the cash hurdle and are dollar-feasible at ``amount``."""
    feasible = feasible_mask(catalog.candidates.units, catalog.minimums,
                             catalog.candidates.grid_units, amount)
    return catalog.table[catalog.table["passes_hurdle"].to_numpy() & feasible]


def within_cap(pool: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Keep packages whose largest single-company look-through exposure is at most ``cap``."""
    return pool[pool["max_exposure"] <= cap + CAP_TOLERANCE]


def frontier_table(pool: pd.DataFrame, tol: float) -> pd.DataFrame:
    """The non-dominated rows of ``pool``, sorted by risk, and numbered by ``frontier_index``."""
    keep = pareto_frontier(pool["selection_risk"].to_numpy(), pool["excess_score"].to_numpy(),
                           pool["signature"].to_numpy(), tol)
    frontier = pool.iloc[keep].copy()
    frontier.insert(0, "frontier_index", np.arange(len(frontier)))
    return frontier


def package_summary(row: pd.Series, tickers: list[str]) -> PackageSummary:
    return PackageSummary(
        signature=row["signature"],
        weights={t: float(row[WEIGHT_PREFIX + t]) for t in tickers if row[WEIGHT_PREFIX + t] > 0},
        cash_weight=float(row["cash_weight"]),
        n_components=int(row["n_components"]),
        selection_return=float(row["selection_return"]),
        excess_score=float(row["excess_score"]),
        vol_full=float(row["vol_full"]),
        vol_recent=float(row["vol_recent"]),
        selection_risk=float(row["selection_risk"]),
        risk_binding=row["risk_binding"],
        package_minimum=float(row["package_minimum"]),
        max_exposure=float(row["max_exposure"]),
        max_company=str(row["max_company"]),
    )


def build_rungs(frontier: pd.DataFrame, targets: np.ndarray, tickers: list[str],
                tol: float, empty_note: str = "no affordable package") -> list[LadderRung]:
    """One rung per risk target. An empty frontier gives every rung ``empty_note`` instead of a package."""
    if frontier.empty:
        return [LadderRung(level=i + 1, target_vol=float(t), note=empty_note)
                for i, t in enumerate(targets)]
    picks = select_ladder(
        frontier["selection_risk"].to_numpy(),
        frontier["excess_score"].to_numpy(),
        (frontier["n_components"] > 1).to_numpy(),
        frontier["signature"].to_numpy(),
        targets,
        tol,
    )
    return [
        LadderRung(level=i + 1, target_vol=float(t), frontier_index=j, distance=dist,
                   package=package_summary(frontier.iloc[j], tickers))
        for i, (t, (j, dist)) in enumerate(zip(targets, picks))
    ]


def ladder_for_amount(catalog: Catalog, amount: float, cap: float | None = None
                      ) -> tuple[pd.DataFrame, Ladder]:
    """Frontier and ladder for one amount. With ``cap``, filter by the cap first and rebuild the frontier."""
    pool = affordable_candidates(catalog, amount)
    note = "no affordable package"
    if cap is not None:
        pool = within_cap(pool, cap)
        note = f"no affordable package keeps every company at or below {cap:.0%}"
    tol = catalog.config.policy.FRONTIER_TOLERANCE
    frontier = frontier_table(pool, tol)
    rungs = build_rungs(frontier, catalog.targets, catalog.tickers, tol, note)
    return frontier, Ladder(account_amount=amount, rungs=rungs)
