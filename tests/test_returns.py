import numpy as np
import pandas as pd
import pytest

from argento_lite.returns import (
    annualized_mean,
    package_selection_return,
    recency_adjustment,
    recent_window,
    return_stats,
)


def test_recency_asymmetry():
    up = recency_adjustment(0.10, alpha_pos=0.25, alpha_neg=0.50)
    down = recency_adjustment(-0.10, alpha_pos=0.25, alpha_neg=0.50)
    assert up == pytest.approx(0.025)
    assert down == pytest.approx(-0.05)
    assert abs(down) > abs(up)
    assert recency_adjustment(0.0, 0.25, 0.50) == 0.0


def test_recency_adjustment_vectorized():
    out = recency_adjustment(np.array([-0.2, 0.0, 0.2]), 0.25, 0.5)
    np.testing.assert_allclose(out, [-0.1, 0.0, 0.05])


def test_annualized_mean_ignores_missing():
    s = pd.Series([np.nan, 0.001, 0.003])
    assert annualized_mean(s, 252) == pytest.approx(0.002 * 252)


def test_return_stats_full_vs_recent(policy):
    idx = pd.bdate_range("2024-01-01", periods=100)
    r = pd.DataFrame({"A": 0.001, "B": 0.001}, index=idx)
    r.loc[idx[-policy.RECENT_WINDOW_DAYS:], "A"] = 0.003   # A improves recently
    r.loc[idx[-policy.RECENT_WINDOW_DAYS:], "B"] = -0.001  # B deteriorates recently
    r.loc[idx[:10], "B"] = np.nan                          # B has a shorter history
    s = return_stats(r, policy)

    mu_f_a = r["A"].mean() * 252
    assert s.loc["A", "mu_full"] == pytest.approx(mu_f_a)
    assert s.loc["A", "mu_recent"] == pytest.approx(0.003 * 252)
    assert s.loc["A", "mu_selection"] == pytest.approx(mu_f_a + 0.25 * (0.003 * 252 - mu_f_a))

    mu_f_b = r["B"].dropna().mean() * 252
    assert s.loc["B", "n_obs"] == 90
    assert s.loc["B", "mu_selection"] == pytest.approx(mu_f_b + 0.50 * (-0.001 * 252 - mu_f_b))


def test_recent_window_requires_enough_common_days():
    r = pd.DataFrame({"A": [0.01] * 5, "B": [0.01] * 5})
    with pytest.raises(ValueError):
        recent_window(r, 10)


def test_package_selection_return_includes_cash():
    w = np.array([[0.5, 0.4]])
    out = package_selection_return(w, np.array([0.1]), np.array([0.10, 0.02]), 0.04)
    assert out[0] == pytest.approx(0.5 * 0.10 + 0.4 * 0.02 + 0.1 * 0.04)
