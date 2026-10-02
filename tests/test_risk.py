import numpy as np
import pandas as pd
import pytest

from argento_lite.risk import covariance, package_vol, selection_risk


def test_selection_risk_is_max_of_two_vols():
    vf = np.array([0.10, 0.20, 0.15])
    vr = np.array([0.12, 0.18, 0.15])
    risk, binding = selection_risk(vf, vr)
    np.testing.assert_allclose(risk, [0.12, 0.20, 0.15])
    assert list(binding) == ["recent", "full", "full"]


def test_package_vol_two_assets_by_hand():
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0001]])
    w = np.array([[0.6, 0.4]])
    expected = np.sqrt(252 * (0.36 * 0.0004 + 2 * 0.24 * 0.0001 + 0.16 * 0.0001))
    assert package_vol(w, cov, 252)[0] == pytest.approx(expected)


def test_zero_vol_cash_scales_vol_linearly():
    cov = np.array([[0.0004]])
    full = package_vol(np.array([[1.0]]), cov, 252)[0]
    ninety = package_vol(np.array([[0.9]]), cov, 252, np.array([0.1]), cash_vol=0.0)[0]
    assert ninety == pytest.approx(0.9 * full)


def test_covariance_rejects_missing_values():
    df = pd.DataFrame({"A": [0.01, np.nan, 0.02], "B": [0.0, 0.01, 0.02]})
    with pytest.raises(ValueError):
        covariance(df)
