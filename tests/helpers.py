"""Reference definitions used as oracles in tests."""


def dominates(risk_b: float, score_b: float, risk_a: float, score_a: float, tol: float) -> bool:
    """True if b is no riskier, no worse, and strictly better in one. Differences within ``tol`` are ties."""
    no_worse = risk_b <= risk_a + tol and score_b >= score_a - tol
    strictly_better = risk_b < risk_a - tol or score_b > score_a + tol
    return no_worse and strictly_better
