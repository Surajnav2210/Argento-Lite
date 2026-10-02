from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from argento_lite.schemas import AppConfig, CompanyAlias, CompanyIdentityConfig, EtfSpec, PolicyConfig, UniverseConfig

POLICY_KWARGS = dict(
    ANNUALIZATION_FACTOR=252,
    RECENT_WINDOW_DAYS=20,
    ALPHA_POS=0.25,
    ALPHA_NEG=0.50,
    MIN_PORTFOLIO_COMPONENTS=1,
    MAX_PORTFOLIO_COMPONENTS=3,
    WEIGHT_STEP=0.10,
    MIN_COMPONENT_WEIGHT=0.10,
    ALLOWED_CASH_WEIGHTS=[0.0, 0.10],
    CASH_HURDLE=0.04,
    CASH_VOLATILITY=0.0,
    RISK_LEVEL_COUNT=10,
    RISK_SCALE_MIN=0.04,
    RISK_SCALE_MAX=0.24,
    FRONTIER_TOLERANCE=1e-9,
    ACCOUNT_AMOUNTS=[10000, 50000],
    CONCENTRATION_CAP=0.08,
)


TOY_COMPANIES = CompanyIdentityConfig(
    aliases={"XA": CompanyAlias(key="X", name="Company X"), "XB": CompanyAlias(key="X", name="Company X")},
    non_company=["FUND"],
)


@pytest.fixture
def policy() -> PolicyConfig:
    return PolicyConfig(**POLICY_KWARGS)


@pytest.fixture
def toy_config(policy: PolicyConfig) -> AppConfig:
    universe = UniverseConfig(
        price_start=date(2020, 1, 1),
        price_end=date(2021, 12, 31),
        etfs=[
            EtfSpec(ticker="AAA", name="Stock-like", min_investment=5000),
            EtfSpec(ticker="BBB", name="Bond-like", min_investment=5000),
            EtfSpec(ticker="CCC", name="Young fund", min_investment=10000,
                    history_start=date(2020, 6, 1)),
        ],
    )
    return AppConfig(policy=policy, universe=universe, companies=TOY_COMPANIES)


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    """Deterministic synthetic prices: three assets with different drift and volatility."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    rets = pd.DataFrame({
        "AAA": rng.normal(0.0006, 0.012, len(idx)),
        "BBB": rng.normal(0.0002, 0.004, len(idx)),
        "CCC": rng.normal(0.0008, 0.018, len(idx)),
    }, index=idx)
    return 100 * (1 + rets).cumprod()


@pytest.fixture
def toy_holdings():
    """AAA holds company X (as two share classes) and Y; BBB has none; CCC holds X and Z."""
    from argento_lite.schemas import Holding, HoldingsSnapshot

    def snap(ticker, rows, source="yfinance"):
        return HoldingsSnapshot(ticker=ticker, source=source,
                                holdings=[Holding(symbol=s, name=n, weight=w) for s, n, w in rows])

    return {
        "AAA": snap("AAA", [("XA", "X class A", 0.30), ("XB", "X class B", 0.20), ("Y", "Company Y", 0.10),
                            ("FUND", "A fund", 0.15)]),
        "BBB": snap("BBB", [], source="none"),
        "CCC": snap("CCC", [("XA", "X class A", 0.10), ("Z", "Company Z", 0.20)]),
    }
