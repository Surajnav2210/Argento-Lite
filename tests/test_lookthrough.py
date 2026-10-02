import numpy as np
import pandas as pd
import pytest

from argento_lite.data.holdings import get_holdings
from argento_lite.lookthrough import (
    holdings_matrix,
    known_share,
    max_exposure_all,
    overlap_matrix,
    package_exposure,
)
from argento_lite.schemas import Holding, HoldingsSnapshot
from tests.conftest import TOY_COMPANIES

TICKERS = ["AAA", "BBB", "CCC"]


@pytest.fixture
def matrix(toy_holdings):
    return holdings_matrix(toy_holdings, TICKERS, TOY_COMPANIES)


def test_aliases_merge_share_classes(matrix):
    H, names = matrix
    # XA 0.30 + XB 0.20 are one company inside AAA; CCC's XA is the same company
    assert H.loc["AAA", "X"] == pytest.approx(0.50)
    assert H.loc["CCC", "X"] == pytest.approx(0.10)
    assert names["X"] == "Company X"
    assert "XA" not in H.columns and "XB" not in H.columns


def test_non_company_holdings_are_dropped(matrix):
    H, _ = matrix
    assert "FUND" not in H.columns
    assert H.loc["AAA"].sum() == pytest.approx(0.60)     # 0.50 + 0.10, the fund's 0.15 is excluded


def test_two_etf_toy_exposure(matrix):
    """Exposure to X in a 60% AAA + 40% CCC package = 0.6 * 0.50 + 0.4 * 0.10 = 0.34."""
    H, names = matrix
    w = np.array([[0.6, 0.0, 0.4]])
    top, driver = max_exposure_all(w, H)
    assert top[0] == pytest.approx(0.34)
    assert driver[0] == "X"

    table = package_exposure({"AAA": 0.6, "CCC": 0.4}, H, names)
    x = table[table["company"] == "Company X"].iloc[0]
    assert x["exposure"] == pytest.approx(0.34)
    assert x["AAA"] == pytest.approx(0.30) and x["CCC"] == pytest.approx(0.04)
    assert table["exposure"].is_monotonic_decreasing


def test_etf_with_no_holdings_contributes_zero(matrix):
    H, _ = matrix
    assert (H.loc["BBB"] == 0).all()
    top, driver = max_exposure_all(np.array([[0.0, 1.0, 0.0]]), H)     # 100% BBB
    assert top[0] == 0 and driver[0] == ""
    # adding BBB only dilutes: 50% AAA + 50% BBB halves AAA's exposure
    top, _ = max_exposure_all(np.array([[0.5, 0.5, 0.0]]), H)
    assert top[0] == pytest.approx(0.25)


def test_max_exposure_all_matches_one_at_a_time(matrix):
    H, _ = matrix
    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(3), size=40)
    top, driver = max_exposure_all(W, H)
    for w, t, d in zip(W, top, driver):
        e = w @ H.to_numpy()
        assert t == pytest.approx(e.max())
        assert d == H.columns[e.argmax()]


def test_no_company_holdings_anywhere():
    empty = {t: HoldingsSnapshot(ticker=t, holdings=[], source="none") for t in TICKERS}
    H, _ = holdings_matrix(empty, TICKERS, TOY_COMPANIES)
    top, driver = max_exposure_all(np.array([[0.5, 0.3, 0.2]]), H)
    assert H.shape == (3, 0) and top[0] == 0 and driver[0] == ""


def test_overlap_matrix_is_sum_of_minimums(matrix):
    H, _ = matrix
    ov = overlap_matrix(H)
    assert ov.loc["AAA", "CCC"] == pytest.approx(0.10)       # only X is shared: min(0.50, 0.10)
    assert ov.loc["AAA", "CCC"] == ov.loc["CCC", "AAA"]
    assert ov.loc["AAA", "AAA"] == pytest.approx(H.loc["AAA"].sum())
    assert (ov.loc["BBB"] == 0).all()


def test_known_share_ignores_etfs_without_holdings(toy_holdings):
    # AAA covers 0.75 of its fund (0.30+0.20+0.10+0.15), BBB has no holdings and is ignored
    assert known_share({"AAA": 0.5, "BBB": 0.5}, toy_holdings) == pytest.approx(0.75)
    assert known_share({"BBB": 1.0}, toy_holdings) == 1.0


def test_snapshot_partial_flag():
    snap = HoldingsSnapshot(ticker="X", source="yfinance", holdings=[Holding(symbol="A", name="A", weight=0.4)])
    assert snap.is_partial and snap.coverage == pytest.approx(0.4)
    assert not HoldingsSnapshot(ticker="B", source="none", holdings=[]).is_partial


def test_get_holdings_falls_back_to_csv_then_none(tmp_path):
    csv_dir, cache = tmp_path / "csv", tmp_path / "cache"
    csv_dir.mkdir()
    pd.DataFrame({"symbol": ["AAA", "BBB"], "name": ["A", "B"], "weight": [0.3, 0.2]}).to_csv(
        csv_dir / "ZZZ.csv", index=False)
    snap = get_holdings("ZZZ", csv_dir, cache, allow_fetch=False)
    assert snap.source == "csv" and snap.coverage == pytest.approx(0.5)
    none = get_holdings("NOPE", csv_dir, cache, allow_fetch=False)
    assert none.source == "none" and none.holdings == []
