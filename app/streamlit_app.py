"""Argento-Lite Streamlit app. UI only; the logic lives in src/argento_lite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from argento_lite.catalog import Catalog, affordable_candidates, build_catalog, ladder_for_amount
from argento_lite.config import CACHE_DIR, load_config
from argento_lite.data.prices import PriceCacheMissing
from argento_lite.lookthrough import known_share, overlap_matrix, package_exposure
from argento_lite.plots import exposure_bar, frontier_figure, overlap_heatmap
from argento_lite.report import cap_headline, compare_ladders, composition_label, ladder_frame
from argento_lite.schemas import Ladder

DISCLAIMER = "Educational research prototype, not investment advice. Constants and ETF minimums are illustrative."
CAP_RANGE = (0.05, 0.15)

st.set_page_config(page_title="Argento-Lite", layout="wide")


@st.cache_resource(show_spinner="Building candidate catalog…")
def get_catalog() -> Catalog:
    return build_catalog(load_config(), cache_dir=CACHE_DIR, allow_fetch=False)


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:.2%}"


def signed_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+.2%}"


def money(x: float) -> str:
    return "" if pd.isna(x) else f"${x:,.0f}"


def md_money(x: float) -> str:
    """money() that is safe inside st.markdown (a pair of bare $ signs would render as math)."""
    return money(x).replace("$", r"\$")


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def section(plain: str, accent: str) -> None:
    st.markdown(f'<h2 class="section">{plain} <em>{accent}</em></h2>', unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
<div class="hero">
  <div class="eyebrow">Argento-Lite</div>
  <h1>Which mix of funds, at every level of risk.</h1>
  <p>Argento rebuilt on 12 public funds, plus one idea: what would a single-company cap cost?</p>
</div>
""", unsafe_allow_html=True)


def render_sidebar(catalog: Catalog) -> tuple[float, float]:
    p = catalog.config.policy
    with st.sidebar:
        amount = st.selectbox("Account amount", p.ACCOUNT_AMOUNTS,
                              index=min(2, len(p.ACCOUNT_AMOUNTS) - 1), format_func=money,
                              help="How much money the client has. Bigger accounts can afford more "
                                   "of the funds, so the choices change.")
        default = min(max(p.CONCENTRATION_CAP, CAP_RANGE[0]), CAP_RANGE[1])
        cap = st.slider("Max share in any one company (my addition)", *CAP_RANGE, value=default,
                        step=0.01, format="%.2f",
                        help="Used by the Overview headline and the Concentration cap tab. "
                             "0.10 means no single company may be more than 10% of the client's money.")
    return amount, cap


def render_overview(catalog: Catalog, amount: float, cap: float) -> None:
    _, base = ladder_for_amount(catalog, amount)
    _, aware = ladder_for_amount(catalog, amount, cap)
    head = cap_headline(compare_ladders(catalog, base, aware))

    eyebrow("Thinking out loud")
    section("What I noticed,", "and what I tried.")
    st.markdown("""
Argento picks a mix of funds for each risk level. It sets no limit on how much of a mix can sit in one company.

Funds overlap. SPY, QQQ, XLK and SMH all hold NVIDIA, so a mix of them can quietly pile a client's money into one name.

So I tried an optional cap: drop every mix that breaks it, rebuild the frontier, redo the ladder and compare.
""")

    eyebrow("What I found")
    st.markdown(f"At {md_money(amount)} with a {cap:.0%} cap (both in the sidebar):")
    c1, c2, c3 = st.columns(3)
    top, big = head["top"], head["biggest"]
    c1.metric("Biggest single-company share, no cap", pct(top["exposure"]) if top else "n/a")
    if top:
        c1.caption(f"{top['company']}, at risk level {top['level']}")
    c2.metric("Risk levels that change", f"{head['n_changed']} of {head['levels']}")
    c3.metric("Biggest change in return", f"{big['d_return'] * 100:+.1f} pts" if big else "none")
    if big:
        c3.caption(f"Level {big['level']}: {big['base_company']} {pct(big['base_exposure'])} → "
                   f"{big['aware_company']} {pct(big['aware_exposure'])}")
    if head["n_unavailable"]:
        st.warning(f"At this cap, {head['n_unavailable']} level(s) have no qualifying mix at all.")
    elif head["n_changed"] == 0:
        st.info("At this account size the cap never binds: the unconstrained picks already respect it.")
    st.caption("pts = percentage points of yearly return. Negative means return given up for the cap. "
               "Caveat: the free data gives only each fund's top 10 holdings, so company shares are lower bounds, "
               "and all constants and fund minimums are illustrative.")


def render_frontier(catalog: Catalog, amount: float) -> None:
    frontier, ladder = ladder_for_amount(catalog, amount)
    pool = affordable_candidates(catalog, amount)
    if pool.empty:
        st.warning("No package is feasible at this account amount.")
        return
    st.markdown("Each dot is one mix of funds: higher earns more, further right is bumpier. The line is the "
                "frontier, the best deals. Stars are the mixes picked for each comfort level (L1 calm, L10 wild).")
    st.plotly_chart(frontier_figure(pool, frontier, ladder, catalog.targets,
                                    catalog.config.policy.CASH_HURDLE), width="stretch")


def render_ladder(catalog: Catalog, amount: float) -> None:
    _, ladder = ladder_for_amount(catalog, amount)
    frame = ladder_frame(ladder)
    if "selection_risk" not in frame:
        st.warning("No affordable package at this account amount.")
        return
    st.dataframe(pd.DataFrame({
        "Level": frame["level"],
        "Target vol": frame["target_vol"].map(pct),
        "Composition": frame["composition"],
        "Selection return": frame["selection_return"].map(pct),
        "Selection risk": frame["selection_risk"].map(pct),
        "Binding estimate": frame["risk_binding"],
        "Package minimum": frame["package_minimum"].map(money),
    }), hide_index=True, width="stretch", column_config={
        "Target vol": st.column_config.Column(help="The amount of ups and downs this comfort level aims for."),
        "Selection return": st.column_config.Column(
            help="Expected yearly return of the mix, using Argento's recency-adjusted estimates."),
        "Selection risk": st.column_config.Column(
            help="Ups and downs of the mix actually chosen. The closer to the target, the better."),
        "Binding estimate": st.column_config.Column(
            help="Which risk measurement was larger: 'full' (all history) or 'recent' (last year)."),
        "Package minimum": st.column_config.Column(
            help="Smallest account that meets every fund's minimum investment for this mix."),
    })


def holdings_note(catalog: Catalog) -> str:
    partial = [t for t, s in catalog.holdings.items() if s.is_partial]
    none = [t for t, s in catalog.holdings.items() if not s.holdings]
    sources = sorted({catalog.holdings[t].source for t in partial})
    text = f"Look-through is partial: {', '.join(sources)} top holdings only, for {', '.join(partial)}."
    if none:
        text += f" No company holdings for {', '.join(none)} (zero company exposure)."
    return text + " Exposures are therefore lower bounds. Holdings are today's snapshot."


def render_exposure(catalog: Catalog, ladder: Ladder, level: int, title: str, cap: float | None) -> None:
    st.markdown(title)
    pkg = ladder.rungs[level - 1].package
    if pkg is None:
        st.warning(ladder.rungs[level - 1].note)
        return
    st.markdown(f"{composition_label(pkg)}  \nlargest: {catalog.company_names.get(pkg.max_company, '-')} "
                f"at {pct(pkg.max_exposure)}")
    exposure = package_exposure(pkg.weights, catalog.exposure, catalog.company_names)
    if exposure.empty:
        st.info("No company holdings in this package.")
        return
    st.plotly_chart(exposure_bar(exposure, cap), width="stretch", key=f"bar_{title}_{level}")
    st.caption(f"Partial: known holdings cover {known_share(pkg.weights, catalog.holdings):.0%} of "
               f"this package's equity-ETF weight.")


def render_look_inside(catalog: Catalog, amount: float, cap: float) -> None:
    _, base = ladder_for_amount(catalog, amount)
    _, aware = ladder_for_amount(catalog, amount, cap)
    st.markdown(f"If a client said no single company may be more than {cap:.0%} of their money, what changes? "
                "Left is the normal pick for the level you choose, right is the pick under the cap. "
                "The dashed line is the cap.")
    st.caption(holdings_note(catalog))

    level = st.selectbox("Risk level", [r.level for r in base.rungs], index=len(base.rungs) - 1,
                         key="inside_level")
    left, right = st.columns(2)
    with left:
        render_exposure(catalog, base, level, "Unconstrained", cap)
    with right:
        render_exposure(catalog, aware, level, f"Capped at {cap:.0%}", cap)

    st.markdown(f"#### Ladders side by side at {md_money(amount)}, cap {cap:.0%}")
    cmp = compare_ladders(catalog, base, aware)
    if all(r.package is None for r in aware.rungs):
        st.warning(aware.rungs[0].note)
    st.dataframe(pd.DataFrame({
        "Level": cmp["level"],
        "Unconstrained": cmp["base"],
        "Concentration-aware": cmp["aware"],
        "Δ return": cmp["d_return"].map(signed_pct),
        "Δ risk": cmp["d_risk"].map(signed_pct),
        "Max company before": cmp["base_exposure"].map(pct),
        "Max company after": cmp["aware_exposure"].map(pct),
        "Driver before": cmp["base_company"],
        "Driver after": cmp["aware_company"],
    }), hide_index=True, width="stretch", column_config={
        "Unconstrained": st.column_config.Column(help="Argento's normal pick for this level."),
        "Concentration-aware": st.column_config.Column(help="The pick when no company may exceed the cap."),
        "Δ return": st.column_config.Column(help="Return of the capped pick minus the normal pick."),
        "Δ risk": st.column_config.Column(help="Risk of the capped pick minus the normal pick."),
        "Driver before": st.column_config.Column(help="The company with the largest share in the normal pick."),
        "Driver after": st.column_config.Column(help="The company with the largest share in the capped pick."),
    })
    st.caption("Δ = capped minus normal, so a negative Δ return is the cost of the cap.")

    st.markdown("#### ETF overlap")
    overlap = overlap_matrix(catalog.exposure)
    with_holdings = [t for t in overlap.index if catalog.exposure.loc[t].sum() > 0]
    st.plotly_chart(overlap_heatmap(overlap.loc[with_holdings, with_holdings]), width="stretch")
    st.caption("Darker means two funds hold more of the same companies, so buying both diversifies less. "
               "Only each fund's top 10 holdings are counted, so real overlap is higher, and the diagonal is left blank.")


def main() -> None:
    st.html(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")
    render_hero()
    st.caption(DISCLAIMER)

    try:
        catalog = get_catalog()
    except PriceCacheMissing as exc:
        st.error(str(exc))
        st.stop()

    amount, cap = render_sidebar(catalog)
    tab_overview, tab_frontier, tab_ladder, tab_cap = st.tabs(
        ["Overview", "Frontier", "Risk ladder", "Concentration cap (my addition)"])
    with tab_overview:
        render_overview(catalog, amount, cap)
    with tab_frontier:
        render_frontier(catalog, amount)
    with tab_ladder:
        render_ladder(catalog, amount)
    with tab_cap:
        render_look_inside(catalog, amount, cap)


main()
