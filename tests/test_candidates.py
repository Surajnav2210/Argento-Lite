from math import comb

import numpy as np
import pytest

from argento_lite.candidates import compositions, enumerate_candidates, make_signature
from argento_lite.schemas import PolicyConfig
from tests.conftest import POLICY_KWARGS

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "XLE", "VNQ", "XLK", "SMH"]


@pytest.fixture
def cands(policy):
    return enumerate_candidates(TICKERS, policy)


def test_weights_sum_to_one_including_cash(cands):
    total_units = cands.units.sum(axis=1) + cands.cash_units
    assert (total_units == cands.grid_units).all()
    np.testing.assert_allclose(cands.weights.sum(axis=1) + cands.cash_weights, 1.0)


def test_component_floor_and_grid(cands, policy):
    held = cands.units[cands.units > 0]
    assert (held >= policy.min_component_units).all()
    assert np.issubdtype(cands.units.dtype, np.integer)


def test_cash_rules(cands, policy):
    single = cands.n_components == 1
    assert (cands.cash_units[single] == 0).all(), "single-component packages are fully invested"
    assert set(cands.cash_weights[~single]) <= set(policy.ALLOWED_CASH_WEIGHTS)
    assert (cands.n_components >= 1).all(), "no cash-only packages"


def test_component_counts(cands, policy):
    assert cands.n_components.min() == policy.MIN_PORTFOLIO_COMPONENTS
    assert cands.n_components.max() == policy.MAX_PORTFOLIO_COMPONENTS


def test_candidate_count_matches_plan(cands):
    n = len(TICKERS)
    expected = n * 1 + comb(n, 2) * (9 + 8) + comb(n, 3) * (36 + 28)
    assert expected == 15214
    assert len(cands) == expected


def test_signatures_unique_and_deterministic(cands, policy):
    assert len(set(cands.signatures)) == len(cands)
    again = enumerate_candidates(TICKERS, policy)
    assert again.signatures == cands.signatures


def test_signature_format():
    assert make_signature({"SPY": 0.5, "IEF": 0.2, "GLD": 0.2}, 0.1) == "GLD:0.20|IEF:0.20|SPY:0.50|CASH:0.10"
    assert make_signature({"SPY": 1.0}, 0.0) == "SPY:1.00"


def test_compositions_small_case():
    assert list(compositions(2, 4, 1)) == [(1, 3), (2, 2), (3, 1)]
    assert list(compositions(3, 2, 1)) == []


def test_policy_rejects_off_grid_floor():
    bad = {**POLICY_KWARGS, "MIN_COMPONENT_WEIGHT": 0.15}
    with pytest.raises(ValueError):
        PolicyConfig(**bad)


def test_policy_rejects_cash_only():
    bad = {**POLICY_KWARGS, "ALLOWED_CASH_WEIGHTS": [0.0, 1.0]}
    with pytest.raises(ValueError):
        PolicyConfig(**bad)


def test_policy_requires_alpha_neg_above_alpha_pos():
    bad = {**POLICY_KWARGS, "ALPHA_NEG": 0.25}
    with pytest.raises(ValueError):
        PolicyConfig(**bad)
