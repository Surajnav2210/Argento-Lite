"""Pydantic models for validated configuration and for ladder results."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_GRID_EPS = 1e-9


def _is_multiple(value: float, step: float) -> bool:
    ratio = value / step
    return abs(ratio - round(ratio)) < _GRID_EPS


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PolicyConfig(_Frozen):
    """Illustrative policy constants (config/policy.yaml). Plutus withholds its real values."""

    ANNUALIZATION_FACTOR: int = Field(gt=0)
    RECENT_WINDOW_DAYS: int = Field(gt=1)
    ALPHA_POS: float = Field(ge=0, le=1)
    ALPHA_NEG: float = Field(ge=0, le=1)
    MIN_PORTFOLIO_COMPONENTS: int = Field(ge=1)
    MAX_PORTFOLIO_COMPONENTS: int = Field(ge=1)
    WEIGHT_STEP: float = Field(gt=0, le=1)
    MIN_COMPONENT_WEIGHT: float = Field(gt=0, le=1)
    ALLOWED_CASH_WEIGHTS: list[float]
    CASH_HURDLE: float
    CASH_VOLATILITY: float = Field(ge=0)
    RISK_LEVEL_COUNT: int = Field(ge=1)
    RISK_SCALE_MIN: float = Field(ge=0)
    RISK_SCALE_MAX: float = Field(gt=0)
    FRONTIER_TOLERANCE: float = Field(ge=0)
    ACCOUNT_AMOUNTS: list[float]
    CONCENTRATION_CAP: float = Field(gt=0, le=1)

    @field_validator("ALLOWED_CASH_WEIGHTS")
    @classmethod
    def _cash_weights_valid(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("ALLOWED_CASH_WEIGHTS must not be empty")
        if any(w < 0 or w >= 1 for w in v):
            raise ValueError("cash weights must be in [0, 1); cash-only packages are prohibited")
        return sorted(set(v))

    @field_validator("ACCOUNT_AMOUNTS")
    @classmethod
    def _amounts_positive(cls, v: list[float]) -> list[float]:
        if not v or any(a <= 0 for a in v):
            raise ValueError("ACCOUNT_AMOUNTS must be a non-empty list of positive amounts")
        return sorted(v)

    @model_validator(mode="after")
    def _cross_checks(self) -> PolicyConfig:
        if self.ALPHA_NEG <= self.ALPHA_POS:
            raise ValueError("ALPHA_NEG must be greater than ALPHA_POS (deterioration weighs more)")
        if self.MIN_PORTFOLIO_COMPONENTS > self.MAX_PORTFOLIO_COMPONENTS:
            raise ValueError("MIN_PORTFOLIO_COMPONENTS must be <= MAX_PORTFOLIO_COMPONENTS")
        if not _is_multiple(1.0, self.WEIGHT_STEP):
            raise ValueError("1 / WEIGHT_STEP must be an integer")
        if not _is_multiple(self.MIN_COMPONENT_WEIGHT, self.WEIGHT_STEP):
            raise ValueError("MIN_COMPONENT_WEIGHT must lie on the WEIGHT_STEP grid")
        if any(not _is_multiple(w, self.WEIGHT_STEP) for w in self.ALLOWED_CASH_WEIGHTS):
            raise ValueError("ALLOWED_CASH_WEIGHTS must lie on the WEIGHT_STEP grid")
        if self.RISK_SCALE_MIN >= self.RISK_SCALE_MAX:
            raise ValueError("RISK_SCALE_MIN must be < RISK_SCALE_MAX")
        if self.MIN_PORTFOLIO_COMPONENTS * self.MIN_COMPONENT_WEIGHT > 1 + _GRID_EPS:
            raise ValueError("MIN_PORTFOLIO_COMPONENTS * MIN_COMPONENT_WEIGHT exceeds 1")
        return self

    @property
    def grid_units(self) -> int:
        """Number of WEIGHT_STEP units in a fully invested package (10 for a 0.10 step)."""
        return round(1.0 / self.WEIGHT_STEP)

    @property
    def min_component_units(self) -> int:
        return round(self.MIN_COMPONENT_WEIGHT / self.WEIGHT_STEP)

    @property
    def cash_units(self) -> list[int]:
        return [round(w / self.WEIGHT_STEP) for w in self.ALLOWED_CASH_WEIGHTS]


class EtfSpec(_Frozen):
    ticker: str
    name: str
    min_investment: float = Field(gt=0, description="Hypothetical minimum, illustration only")
    history_start: date | None = None


class UniverseConfig(_Frozen):
    price_start: date
    price_end: date
    etfs: list[EtfSpec]

    @model_validator(mode="after")
    def _checks(self) -> UniverseConfig:
        tickers = [e.ticker for e in self.etfs]
        if len(set(tickers)) != len(tickers):
            raise ValueError("duplicate tickers in universe")
        if not tickers:
            raise ValueError("universe must contain at least one ETF")
        if self.price_start >= self.price_end:
            raise ValueError("price_start must be before price_end")
        return self

    @property
    def tickers(self) -> list[str]:
        return [e.ticker for e in self.etfs]

    @property
    def minimums(self) -> dict[str, float]:
        return {e.ticker: e.min_investment for e in self.etfs}

    @property
    def history_starts(self) -> dict[str, date]:
        return {e.ticker: e.history_start for e in self.etfs if e.history_start is not None}


class CompanyAlias(_Frozen):
    key: str
    name: str


class CompanyIdentityConfig(_Frozen):
    """How holdings map to companies: share-class / cross-listing merges, and non-company holdings."""

    aliases: dict[str, CompanyAlias] = Field(default_factory=dict)
    non_company: list[str] = Field(default_factory=list)


class AppConfig(_Frozen):
    """Everything read from config/."""

    policy: PolicyConfig
    universe: UniverseConfig
    companies: CompanyIdentityConfig


class Holding(_Frozen):
    symbol: str
    name: str
    weight: float = Field(ge=0, le=1)


class HoldingsSnapshot(_Frozen):
    """What is known about one ETF's holdings. ``none`` means no company holdings (bonds, gold)."""

    ticker: str
    holdings: list[Holding]
    source: Literal["yfinance", "csv", "none"]

    @property
    def coverage(self) -> float:
        """Share of the ETF these holdings add up to (about 0.46 for QQQ's top 10)."""
        return float(sum(h.weight for h in self.holdings))

    @property
    def is_partial(self) -> bool:
        return self.source != "none" and self.coverage < 0.999


class PackageSummary(_Frozen):
    """One candidate package with its selection statistics."""

    signature: str
    weights: dict[str, float]
    cash_weight: float
    n_components: int
    selection_return: float
    excess_score: float
    vol_full: float
    vol_recent: float
    selection_risk: float
    risk_binding: Literal["full", "recent"]
    package_minimum: float
    max_exposure: float = 0.0      # largest look-through weight in any single company
    max_company: str = ""          # key of that company ("" if the package has no company holdings)


class LadderRung(_Frozen):
    level: int
    target_vol: float
    frontier_index: int | None = None
    distance: float | None = None
    package: PackageSummary | None = None
    note: str | None = None


class Ladder(_Frozen):
    account_amount: float
    rungs: list[LadderRung]
