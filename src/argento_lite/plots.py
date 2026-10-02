"""Plotly figure builders (no Streamlit imports)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from argento_lite.schemas import Ladder

NAVY = "#111D32"
GOLD = "#B8923A"
MIST = "#ECEFF1"
ETF_COLORS = [NAVY, "#3A4A66", "#6B7A94", "#9FAABD", GOLD, "#D8C28A", "#C9D0DB", "#7A6A3A"]

_HOVER = (
    "%{customdata[0]}<br>selection risk %{x:.2%}<br>selection return %{y:.2%}"
    "<br>binding: %{customdata[1]}<br>package minimum $%{customdata[2]:,.0f}<extra></extra>"
)


def _custom(df: pd.DataFrame) -> np.ndarray:
    return df[["signature", "risk_binding", "package_minimum"]].to_numpy()


def _style(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", color=NAVY, size=12),
        paper_bgcolor="white", plot_bgcolor="white", **layout,
    )
    fig.update_xaxes(gridcolor=MIST, zeroline=False, linecolor=MIST)
    fig.update_yaxes(gridcolor=MIST, zeroline=False, linecolor=MIST)
    return fig


def frontier_figure(pool: pd.DataFrame, frontier: pd.DataFrame, ladder: Ladder,
                    targets: np.ndarray, cash_hurdle: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=pool["selection_risk"], y=pool["selection_return"], mode="markers",
        name=f"Affordable mixes ({len(pool):,})",
        marker=dict(size=4, color="rgba(17,29,50,0.16)"),
        customdata=_custom(pool), hovertemplate=_HOVER,
    ))
    fig.add_trace(go.Scatter(
        x=frontier["selection_risk"], y=frontier["selection_return"], mode="lines+markers",
        name=f"Frontier ({len(frontier)})",
        line=dict(color=NAVY, width=2), marker=dict(size=6, color=NAVY),
        customdata=_custom(frontier), hovertemplate=_HOVER,
    ))

    chosen = [r for r in ladder.rungs if r.package is not None]
    if chosen:
        labels: dict[str, list[str]] = {}
        for r in chosen:
            labels.setdefault(r.package.signature, []).append(str(r.level))
        firsts = {r.package.signature: r.package for r in chosen}
        pk = list(firsts.values())
        fig.add_trace(go.Scatter(
            x=[p.selection_risk for p in pk], y=[p.selection_return for p in pk],
            mode="markers+text", name="Chosen for a comfort level",
            text=["L" + ",".join(labels[p.signature]) for p in pk], textposition="top left",
            marker=dict(size=13, color=GOLD, symbol="star", line=dict(width=1, color="white")),
            customdata=[[p.signature, p.risk_binding, p.package_minimum] for p in pk],
            hovertemplate=_HOVER,
        ))

    for i, t in enumerate(targets, start=1):
        fig.add_vline(x=float(t), line=dict(color="rgba(17,29,50,0.2)", dash="dot", width=1),
                      annotation_text=f"L{i}", annotation_position="top", annotation_font_size=10)
    fig.add_hline(y=cash_hurdle, line=dict(color=GOLD, dash="dash", width=1),
                  annotation_text="cash hurdle", annotation_position="bottom right")

    return _style(
        fig,
        xaxis_title="Risk (annualized volatility)",
        yaxis_title="Expected return (annualized)",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        height=600, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )


def exposure_bar(exposure: pd.DataFrame, cap: float | None = None) -> go.Figure:
    """Top company exposures of one package, stacked by the ETF they come through."""
    fig = go.Figure()
    etfs = [c for c in exposure.columns if c not in ("company", "exposure")]
    ordered = exposure.iloc[::-1]
    for i, etf in enumerate(etfs):
        fig.add_trace(go.Bar(y=ordered["company"], x=ordered[etf], name=etf, orientation="h",
                             marker_color=ETF_COLORS[i % len(ETF_COLORS)],
                             hovertemplate=f"{etf}: %{{x:.2%}}<extra></extra>"))
    if cap is not None:
        fig.add_vline(x=cap, line=dict(color=GOLD, dash="dash", width=1.5),
                      annotation_text=f"cap {cap:.0%}", annotation_position="top")
    return _style(
        fig, barmode="stack", xaxis_tickformat=".0%", xaxis_title="Share of the client's money",
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )


def overlap_heatmap(overlap: pd.DataFrame) -> go.Figure:
    z = overlap.to_numpy(dtype=float).copy()
    np.fill_diagonal(z, np.nan)
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.1%}")(z))
    fig = go.Figure(go.Heatmap(
        z=z, x=list(overlap.columns), y=list(overlap.index),
        colorscale=[[0, "#FFFFFF"], [1, "#8795AE"]], zmin=0, text=text,
        texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z:.2%}<extra></extra>",
        hoverongaps=False, colorbar=dict(tickformat=".0%")))
    _style(fig, height=480, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    return fig.update_layout(plot_bgcolor=MIST)
