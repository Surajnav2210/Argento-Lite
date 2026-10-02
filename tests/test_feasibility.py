import numpy as np

from argento_lite.feasibility import feasible_mask, package_minimums

# tickers: GLD, IEF, SPY with hypothetical minimums
MINS = np.array([10000.0, 5000.0, 5000.0])


def test_package_minimum_is_max_of_min_over_weight():
    units = np.array([[2, 2, 5]])  # 20% GLD, 20% IEF, 50% SPY (+10% cash)
    # GLD 10000/0.2 = 50000, IEF 5000/0.2 = 25000, SPY 5000/0.5 = 10000
    assert package_minimums(units, MINS, 10)[0] == 50000


def test_package_minimum_rounds_up():
    units = np.array([[3, 0, 7]])  # 10000 / 0.3 = 33333.33 -> 33334
    assert package_minimums(units, MINS, 10)[0] == 33334


def test_feasibility_boundary():
    units = np.array([[2, 2, 5]])
    assert feasible_mask(units, MINS, 10, 50000)[0]
    assert not feasible_mask(units, MINS, 10, 49999)[0]


def test_feasible_iff_amount_at_least_package_minimum():
    rng = np.random.default_rng(0)
    units = rng.integers(0, 6, size=(500, 3))
    units[units.sum(axis=1) == 0, 0] = 1
    pmin = package_minimums(units, MINS, 10)
    for amount in (10000, 25000, 33333, 33334, 50000, 100000):
        np.testing.assert_array_equal(feasible_mask(units, MINS, 10, amount), amount >= pmin)


def test_unheld_components_are_ignored():
    units = np.array([[0, 0, 10]])
    assert package_minimums(units, MINS, 10)[0] == 5000
    assert feasible_mask(units, MINS, 10, 5000)[0]
