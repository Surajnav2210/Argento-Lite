"""Turn catalog results into display tables. Nothing here affects selection."""

from __future__ import annotations

import pandas as pd

from argento_lite.catalog import Catalog
from argento_lite.schemas import Ladder, PackageSummary


def composition_label(package: PackageSummary) -> str:
    """e.g. '50% SPY + 20% GLD + 10% cash', largest first."""
    parts = sorted(package.weights.items(), key=lambda kv: (-kv[1], kv[0]))
    text = [f"{round(w * 100)}% {t}" for t, w in parts]
    if package.cash_weight > 0:
        text.append(f"{round(package.cash_weight * 100)}% cash")
    return " + ".join(text)


def ladder_frame(ladder: Ladder) -> pd.DataFrame:
    """One row per risk level with the chosen package's numbers."""
    rows = []
    for rung in ladder.rungs:
        row: dict = {"level": rung.level, "target_vol": rung.target_vol}
        pkg = rung.package
        if pkg is None:
            row["composition"] = rung.note or "no qualifying package"
        else:
            row.update({
                "composition": composition_label(pkg),
                "selection_return": pkg.selection_return,
                "selection_risk": pkg.selection_risk,
                "risk_binding": pkg.risk_binding,
                "package_minimum": pkg.package_minimum,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def compare_ladders(catalog: Catalog, base: Ladder, aware: Ladder) -> pd.DataFrame:
    """Per level: unconstrained vs capped package. Deltas are capped minus unconstrained, blank if no package fits."""
    names = catalog.company_names
    rows = []
    for b, a in zip(base.rungs, aware.rungs):
        bp, ap = b.package, a.package
        row = {
            "level": b.level,
            "target_vol": b.target_vol,
            "base": composition_label(bp) if bp else (b.note or "none"),
            "aware": composition_label(ap) if ap else (a.note or "none"),
            "changed": bool(bp and ap and bp.signature != ap.signature) or (ap is None),
            "base_return": bp.selection_return if bp else float("nan"),
            "aware_return": ap.selection_return if ap else float("nan"),
            "base_risk": bp.selection_risk if bp else float("nan"),
            "aware_risk": ap.selection_risk if ap else float("nan"),
            "base_exposure": bp.max_exposure if bp else float("nan"),
            "aware_exposure": ap.max_exposure if ap else float("nan"),
            "base_company": names.get(bp.max_company, bp.max_company) if bp else "",
            "aware_company": names.get(ap.max_company, ap.max_company) if ap else "",
        }
        row["d_return"] = row["aware_return"] - row["base_return"]
        row["d_risk"] = row["aware_risk"] - row["base_risk"]
        rows.append(row)
    return pd.DataFrame(rows)


def cap_headline(cmp: pd.DataFrame) -> dict:
    """Headline facts from a ``compare_ladders`` table.

    ``top`` is the largest single-company share in the unconstrained ladder, ``biggest`` the changed
    level with the largest move in selection return, ``n_unavailable`` the levels where no package fits.
    """
    has_base = cmp[cmp["base_exposure"].notna()]
    top = None
    if not has_base.empty:
        r = has_base.loc[has_base["base_exposure"].idxmax()]
        top = {"level": int(r["level"]), "company": r["base_company"], "exposure": float(r["base_exposure"])}

    n_unavailable = int((cmp["base_exposure"].notna() & cmp["aware_exposure"].isna()).sum())
    moved = cmp[cmp["changed"] & cmp["d_return"].notna()]
    biggest = None
    if not moved.empty:
        r = moved.loc[moved["d_return"].abs().idxmax()]
        biggest = {
            "level": int(r["level"]), "d_return": float(r["d_return"]), "d_risk": float(r["d_risk"]),
            "base_company": r["base_company"], "base_exposure": float(r["base_exposure"]),
            "aware_company": r["aware_company"], "aware_exposure": float(r["aware_exposure"]),
        }
    return {"top": top, "n_changed": int(cmp["changed"].sum()), "levels": len(cmp),
            "n_unavailable": n_unavailable, "biggest": biggest}
