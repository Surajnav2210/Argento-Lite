import numpy as np

from argento_lite.frontier import excess_scores, pareto_frontier, passes_cash_hurdle
from tests.helpers import dominates

TOL = 1e-9


def test_dominance_toy():
    assert dominates(0.10, 0.06, 0.12, 0.05, TOL)
    assert dominates(0.10, 0.06, 0.10, 0.05, TOL)
    assert not dominates(0.10, 0.05, 0.10, 0.05, TOL)
    assert not dominates(0.08, 0.04, 0.12, 0.06, TOL)
    assert not dominates(0.12, 0.06, 0.08, 0.04, TOL)
    assert not dominates(0.10, 0.05 + 1e-12, 0.10, 0.05, TOL)


def test_pareto_frontier_hand_example():
    #            A     B     C     D     E
    risk = [0.05, 0.08, 0.10, 0.12, 0.15]
    score = [0.01, 0.03, 0.02, 0.05, 0.04]
    sigs = ["A", "B", "C", "D", "E"]
    # C is dominated by B; E is dominated by D.
    assert pareto_frontier(risk, score, sigs, TOL).tolist() == [0, 1, 3]


def test_pareto_frontier_matches_brute_force():
    rng = np.random.default_rng(1)
    risk = rng.uniform(0.03, 0.3, 300)
    score = 0.5 * risk + rng.normal(0, 0.03, 300)
    sigs = [f"P{i:03d}" for i in range(300)]
    fast = set(pareto_frontier(risk, score, sigs, TOL).tolist())
    brute = {
        a for a in range(300)
        if not any(dominates(risk[b], score[b], risk[a], score[a], TOL) for b in range(300) if b != a)
    }
    assert fast == brute


def test_frontier_sorted_by_risk_with_increasing_score():
    rng = np.random.default_rng(2)
    risk = rng.uniform(0.03, 0.3, 200)
    score = rng.normal(0.05, 0.03, 200)
    pos = pareto_frontier(risk, score, [str(i) for i in range(200)], TOL)
    assert np.all(np.diff(risk[pos]) >= 0)
    assert np.all(np.diff(score[pos]) > 0)


def test_cash_hurdle():
    e = excess_scores([0.03, 0.04, 0.05], 0.04)
    np.testing.assert_allclose(e, [-0.01, 0.0, 0.01])
    assert passes_cash_hurdle(e).tolist() == [False, True, True]
