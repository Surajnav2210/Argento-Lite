import numpy as np
import pytest

from argento_lite.ladder import pick_nearest, risk_targets, select_ladder

TOL = 1e-9


def _arrays(risk, score, multi, sigs):
    return (np.asarray(risk, float), np.asarray(score, float), np.asarray(multi, bool),
            np.asarray(sigs, dtype=object))


def test_risk_targets_evenly_spaced(policy):
    t = risk_targets(policy)
    assert len(t) == policy.RISK_LEVEL_COUNT
    assert t[0] == pytest.approx(policy.RISK_SCALE_MIN)
    assert t[-1] == pytest.approx(policy.RISK_SCALE_MAX)
    assert np.allclose(np.diff(t), np.diff(t)[0])


def test_ladder_is_monotone_on_random_frontiers():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = rng.integers(1, 30)
        risk = np.sort(rng.uniform(0.02, 0.4, n))
        score = np.sort(rng.uniform(0.0, 0.3, n))
        sigs = [f"S{i}" for i in range(n)]
        picks = select_ladder(risk, score, np.ones(n, bool), sigs, np.linspace(0.04, 0.24, 10), TOL)
        idx = [j for j, _ in picks]
        assert idx == sorted(idx)


def test_forward_only_blocks_backward_move():
    risk = [0.05, 0.10, 0.20]
    # Targets deliberately descending: the second target is nearest to position 0,
    # but the forward-only rule keeps it at or after the first pick.
    picks = select_ladder(risk, [0.01, 0.02, 0.03], [True] * 3, ["a", "b", "c"], [0.19, 0.06], TOL)
    assert [j for j, _ in picks] == [2, 2]


def test_adjacent_levels_may_share_a_package():
    picks = select_ladder([0.05, 0.30], [0.01, 0.05], [True, True], ["a", "b"], [0.04, 0.08, 0.12], TOL)
    assert [j for j, _ in picks] == [0, 0, 0]


def test_tiebreak_lower_volatility_first():
    r, s, m, g = _arrays([0.09, 0.11], [0.05, 0.06], [True, True], ["b", "a"])
    assert pick_nearest(r, s, m, g, 0.10, 0, TOL) == 0


def test_tiebreak_higher_score_second():
    r, s, m, g = _arrays([0.10, 0.10], [0.04, 0.06], [True, True], ["a", "b"])
    assert pick_nearest(r, s, m, g, 0.10, 0, TOL) == 1


def test_tiebreak_multi_portfolio_third():
    r, s, m, g = _arrays([0.10, 0.10], [0.05, 0.05], [False, True], ["a", "b"])
    assert pick_nearest(r, s, m, g, 0.10, 0, TOL) == 1


def test_tiebreak_lexical_signature_last():
    r, s, m, g = _arrays([0.10, 0.10], [0.05, 0.05], [True, True], ["b", "a"])
    assert pick_nearest(r, s, m, g, 0.10, 0, TOL) == 1


def test_distance_within_tolerance_is_a_tie():
    r, s, m, g = _arrays([0.10 - 1e-12, 0.10 + 1e-12], [0.05, 0.06], [True, True], ["a", "b"])
    # equidistant within tol -> falls through to lower volatility
    assert pick_nearest(r, s, m, g, 0.10, 0, TOL) == 0


def test_empty_frontier_gives_no_picks():
    assert select_ladder([], [], [], [], [0.1, 0.2], TOL) == []
