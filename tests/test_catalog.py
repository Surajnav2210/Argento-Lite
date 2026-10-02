"""End-to-end pipeline on synthetic prices (no network, no cache)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from argento_lite.catalog import (
    affordable_candidates, build_catalog, frontier_table, ladder_for_amount, within_cap)
from argento_lite.config import load_config
from argento_lite.report import cap_headline, compare_ladders, ladder_frame
from tests.helpers import dominates


@pytest.fixture
def catalog(toy_config, toy_prices, toy_holdings):
    return build_catalog(toy_config, prices=toy_prices, holdings=toy_holdings)


def test_history_start_trims_a_young_fund(catalog):
    ccc_first = catalog.stats.loc["CCC", "first_date"]
    assert ccc_first >= date(2020, 6, 1)


def test_selection_risk_is_max_in_catalog(catalog):
    t = catalog.table
    np.testing.assert_allclose(t["selection_risk"], np.maximum(t["vol_full"], t["vol_recent"]))


@pytest.mark.parametrize("amount", [10000, 50000])
def test_ladder_pipeline(catalog, amount):
    tol = catalog.config.policy.FRONTIER_TOLERANCE
    pool = affordable_candidates(catalog, amount)
    frontier, ladder = ladder_for_amount(catalog, amount)

    assert (pool["excess_score"] >= 0).all()
    assert (pool["package_minimum"] <= amount).all()

    r, e = pool["selection_risk"].to_numpy(), pool["excess_score"].to_numpy()
    for _, row in frontier.iterrows():
        assert not any(dominates(rb, eb, row["selection_risk"], row["excess_score"], tol)
                       for rb, eb in zip(r, e))

    idx = [rung.frontier_index for rung in ladder.rungs]
    assert idx == sorted(idx)
    assert len(ladder.rungs) == catalog.config.policy.RISK_LEVEL_COUNT

    frame = ladder_frame(ladder)
    assert len(frame) == len(ladder.rungs)


@pytest.fixture
def cap(catalog):
    """A cap that binds on the toy data: half the largest exposure on the unconstrained ladder."""
    _, base = ladder_for_amount(catalog, 50000)
    return 0.5 * max(r.package.max_exposure for r in base.rungs)


def test_capped_ladder_never_exceeds_the_cap(catalog, cap):
    assert cap > 0
    for amount in (10000, 50000):
        _, ladder = ladder_for_amount(catalog, amount, cap=cap)
        packages = [r.package for r in ladder.rungs if r.package]
        assert packages
        assert all(p.max_exposure <= cap + 1e-9 for p in packages)


def test_cap_binds_on_the_toy_data(catalog, cap):
    """The test above is not vacuous: the unconstrained ladder does break this cap."""
    _, base = ladder_for_amount(catalog, 50000)
    assert max(r.package.max_exposure for r in base.rungs) > cap


def test_capped_frontier_is_built_from_the_capped_pool(catalog, cap):
    tol = catalog.config.policy.FRONTIER_TOLERANCE
    capped_pool = within_cap(affordable_candidates(catalog, 50000), cap)
    frontier, _ = ladder_for_amount(catalog, 50000, cap=cap)
    assert (frontier["max_exposure"] <= cap + 1e-9).all()
    pd.testing.assert_frame_equal(frontier, frontier_table(capped_pool, tol))


def test_frontier_is_rebuilt_not_merely_filtered():
    """Hand-made pool. B dominates C, so C is off the full frontier. Capping removes B,
    and then C is a frontier point: filtering the old frontier would have missed it."""
    pool = pd.DataFrame({
        "signature": ["A", "B", "C"],
        "selection_risk": [0.05, 0.10, 0.12],
        "excess_score": [0.01, 0.05, 0.04],
        "max_exposure": [0.05, 0.30, 0.05],
    })
    full = frontier_table(pool, 1e-9)
    assert list(full["signature"]) == ["A", "B"]

    capped_pool = within_cap(pool, 0.10)
    assert list(frontier_table(capped_pool, 1e-9)["signature"]) == ["A", "C"]
    assert list(full[full["max_exposure"] <= 0.10]["signature"]) == ["A"]     # the wrong way


def test_capped_ladder_is_monotone(catalog, cap):
    _, ladder = ladder_for_amount(catalog, 50000, cap=cap)
    idx = [r.frontier_index for r in ladder.rungs]
    assert idx == sorted(idx)


def test_loose_cap_changes_nothing(catalog):
    _, base = ladder_for_amount(catalog, 50000)
    _, loose = ladder_for_amount(catalog, 50000, cap=1.0)
    assert [r.package.signature for r in base.rungs] == [r.package.signature for r in loose.rungs]


def test_impossible_cap_says_so_on_every_level(catalog):
    frontier, ladder = ladder_for_amount(catalog, 50000, cap=-0.01)
    assert frontier.empty
    assert all(r.package is None and "no affordable package keeps every company" in r.note
               for r in ladder.rungs)
    cmp = compare_ladders(catalog, ladder_for_amount(catalog, 50000)[1], ladder)
    assert cmp["changed"].all() and cmp["d_return"].isna().all()


def test_comparison_deltas_are_aware_minus_base(catalog, cap):
    _, base = ladder_for_amount(catalog, 50000)
    _, aware = ladder_for_amount(catalog, 50000, cap=cap)
    cmp = compare_ladders(catalog, base, aware)
    np.testing.assert_allclose(cmp["d_return"], cmp["aware_return"] - cmp["base_return"])
    np.testing.assert_allclose(cmp["d_risk"], cmp["aware_risk"] - cmp["base_risk"])
    assert (cmp["aware_exposure"] <= cap + 1e-9).all()


def test_cap_headline_loose_cap_reports_no_change(catalog):
    _, base = ladder_for_amount(catalog, 50000)
    _, loose = ladder_for_amount(catalog, 50000, cap=1.0)
    head = cap_headline(compare_ladders(catalog, base, loose))
    assert head["n_changed"] == 0 and head["n_unavailable"] == 0 and head["biggest"] is None
    cmp = compare_ladders(catalog, base, loose)
    assert head["top"]["exposure"] == pytest.approx(cmp["base_exposure"].max())


def test_cap_headline_impossible_cap_flags_every_level(catalog):
    _, base = ladder_for_amount(catalog, 50000)
    _, none = ladder_for_amount(catalog, 50000, cap=-0.01)
    head = cap_headline(compare_ladders(catalog, base, none))
    assert head["n_unavailable"] == head["levels"] == 10 and head["biggest"] is None


def test_cap_headline_biggest_is_the_largest_return_move(catalog, cap):
    _, base = ladder_for_amount(catalog, 50000)
    _, aware = ladder_for_amount(catalog, 50000, cap=cap)
    cmp = compare_ladders(catalog, base, aware)
    head = cap_headline(cmp)
    assert head["n_changed"] == int(cmp["changed"].sum())
    if head["biggest"] is not None:
        moved = cmp[cmp["changed"] & cmp["d_return"].notna()]
        assert abs(head["biggest"]["d_return"]) == pytest.approx(moved["d_return"].abs().max())


def test_candidate_table_exposure_matches_package_weights(catalog):
    """Every candidate's max_exposure equals max over companies of (weights @ H)."""
    w = catalog.candidates.weights
    expected = (w @ catalog.exposure.to_numpy()).max(axis=1)
    np.testing.assert_allclose(catalog.table["max_exposure"], expected)


def test_project_config_loads():
    cfg = load_config()
    assert len(cfg.universe.tickers) == 12
    assert cfg.policy.CONCENTRATION_CAP == 0.10
    assert cfg.companies.aliases["GOOG"].key == cfg.companies.aliases["GOOGL"].key
    assert cfg.companies.aliases["TSM"].key == cfg.companies.aliases["2330.TW"].key
    assert "VRTPX" in cfg.companies.non_company
