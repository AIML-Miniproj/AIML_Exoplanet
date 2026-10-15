"""Plotly figures in the app's dark theme (transparent backgrounds, faint grid, Inter)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from ui.theme import ACCENT, CYAN, color, label, rgba

POINTS = "rgba(226,232,240,0.42)"
GRID = "rgba(255,255,255,0.06)"
AXIS = "#94A3B8"
RED = "#FF5470"

pio.templates["exo_dark"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#04060C",
    font=dict(family="Inter, sans-serif", color=AXIS, size=14),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(255,255,255,0.12)"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(255,255,255,0.12)"),
    legend=dict(bgcolor="rgba(7,10,17,0.85)", bordercolor="#202A38", borderwidth=1),
    hoverlabel=dict(bgcolor="#0D1424", font=dict(family="Inter, sans-serif")),
    margin=dict(l=60, r=20, t=40, b=50), colorway=[CYAN, ACCENT, "#F59E0B", "#EC4899"],
))
TEMPLATE = "exo_dark"
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def _fig(height: int, title: str = "", **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, height=height,
                      title=dict(text=title, font=dict(size=15)) if title else None, **layout)
    return fig


def _glow(fig: go.Figure, x, y, col: str, name: str, width: float = 2.2, **kw) -> None:
    """A thick translucent line under a thin bright one: a soft neon glow."""
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=rgba(col, 0.25),
                             width=width * 4), hoverinfo="skip", showlegend=False, **kw))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=width),
                             name=name, **kw))


def _points(fig: go.Figure, x, y, name: str = "Data", col: str = POINTS, size: float = 2.5,
            **kw) -> None:
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", name=name,
                               marker=dict(color=col, size=size), **kw))


def sector_spans(lc_norm: dict[str, Any]) -> list[tuple[int, float, float]]:
    t = np.asarray(lc_norm.get("time") or [], float)
    s = np.asarray(lc_norm.get("sector") or [], float)
    if not len(t) or len(s) != len(t):
        return []
    return [(int(k), float(t[s == k].min()), float(t[s == k].max()))
            for k in sorted(set(s[np.isfinite(s)].tolist()))]


def light_curve(lc: dict[str, Any], spans: list[tuple[int, float, float]], title: str,
                col: str = POINTS) -> go.Figure:
    fig = _fig(330, title, xaxis_title="Time (BTJD, days)", yaxis_title="Normalized flux",
               showlegend=False)
    for i, (sec, t0, t1) in enumerate(spans):
        fig.add_vrect(x0=t0, x1=t1, fillcolor=rgba(ACCENT, 0.10 if i % 2 else 0.05),
                      line_width=0, layer="below")
        fig.add_vline(x=t0, line=dict(color="rgba(200,205,255,0.25)", dash="dash", width=1))
        fig.add_annotation(x=0.5 * (t0 + t1), y=1.0, yref="paper", yanchor="bottom",
                           text=f"S{sec}", showarrow=False,
                           font=dict(size=12, color="#94A3B8", family="JetBrains Mono"))
    _points(fig, lc.get("time"), lc.get("flux"), col=col, size=2)
    return fig


def bls_periodogram(pg: dict[str, Any], best: float, sde: float, col: str) -> go.Figure:
    fig = _fig(380, "", xaxis_title="Period (days)", yaxis_title="BLS power",
               showlegend=False)
    _glow(fig, pg.get("period"), pg.get("power"), col, "BLS power", width=1.4)
    fig.update_xaxes(type="log")
    if math.isfinite(best):
        fig.add_vline(x=best, line=dict(color=RED, dash="dash", width=2))
        fig.add_annotation(x=math.log10(best), y=1, yref="paper", yanchor="bottom",
                           text=f"Best: {best:.5f} d · SDE {sde:.1f}", showarrow=False,
                           font=dict(color=RED, size=13, family="JetBrains Mono"))
    return fig


def ls_periodogram(ls: dict[str, Any], main: float) -> go.Figure:
    col = color("VARIABLE")
    fig = _fig(320, "", xaxis_title="Period (days)", yaxis_title="Lomb-Scargle power",
               showlegend=False)
    _glow(fig, ls.get("period"), ls.get("power"), col, "Lomb-Scargle power", width=1.4)
    fig.update_xaxes(type="log")
    if math.isfinite(main):
        fig.add_vline(x=main, line=dict(color=RED, dash="dash", width=2))
        fig.add_annotation(x=math.log10(main), y=1, yref="paper", yanchor="bottom",
                           text=f"Main period: {main:.4f} d", showarrow=False,
                           font=dict(color=RED, size=13, family="JetBrains Mono"))
    return fig


def fold(view: dict[str, Any], col: str, windows: list[tuple[float, float]], title: str,
         x_range: tuple[float, float] | None = None) -> go.Figure:
    fig = _fig(380, title, xaxis_title="Phase (orbits)", yaxis_title="Normalized flux",
               legend=dict(orientation="h", y=-0.22))
    for centre, half in windows:
        fig.add_vrect(x0=centre - half, x1=centre + half, fillcolor=rgba(col, 0.13),
                      line_width=0, layer="below")
    _points(fig, view.get("phase"), view.get("flux"))
    _glow(fig, view.get("bin_phase"), view.get("bin_flux"), col, "Binned median")
    if windows:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Dip window",
                                 marker=dict(size=12, symbol="square", color=rgba(col, 0.35))))
    if x_range:
        fig.update_xaxes(range=list(x_range))
    return fig


def overlay(a: dict[str, Any], b: dict[str, Any], names: tuple[str, str],
            cols: tuple[str, str], title: str) -> go.Figure:
    fig = _fig(350, title, xaxis_title="Hours from eclipse centre",
               yaxis_title="Normalized flux", legend=dict(orientation="h", y=-0.42),
               margin=dict(l=60, r=20, t=40, b=100))
    for v, name, col in ((a, names[0], cols[0]), (b, names[1], cols[1])):
        _points(fig, v.get("hours"), v.get("flux"), name=f"{name} (points)",
                col=rgba(col, 0.22), size=2, showlegend=False)
    for v, name, col in ((a, names[0], cols[0]), (b, names[1], cols[1])):
        _glow(fig, v.get("bin_hours"), v.get("bin_flux"), col, name, width=2)
    return fig


def probabilities(probs: dict[str, float]) -> go.Figure:
    order = ["NOISE", "VARIABLE", "EB", "PLANET"]
    vals = [float(probs.get(c, 0) or 0) for c in order]
    fig = _fig(260, "", showlegend=False, bargap=0.42, margin=dict(l=150, r=60, t=10, b=44))
    fig.add_trace(go.Bar(
        x=vals, y=[label(c) for c in order], orientation="h",
        marker=dict(color=[rgba(color(c), 0.85) for c in order],
                    line=dict(color=[color(c) for c in order], width=1.5),
                    cornerradius=6),
        text=[f"  {100 * v:.0f}%" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(family="JetBrains Mono", color="#DDE2F5")))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", title="Classifier probability")
    # Room between the class names and the bars
    fig.update_yaxes(ticksuffix="    ", ticklabelstandoff=10, automargin=True)
    return fig


def confusion(cm_norm: list[list[float]], counts: list[list[int]], classes: list[str]
              ) -> go.Figure:
    names = [label(c) for c in classes]
    z = np.asarray(cm_norm, float)
    text = [[f"{z[i, j]:.0%}<br>({counts[i][j]})" for j in range(len(classes))]
            for i in range(len(classes))]
    fig = _fig(420, "", xaxis_title="Predicted class", yaxis_title="True class",
               margin=dict(l=130, r=20, t=20, b=70))
    fig.add_trace(go.Heatmap(
        z=z, x=names, y=names, text=text, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=14, color="#F4F6FF"),
        colorscale=[[0, "#04060C"], [0.35, "#14305A"], [0.7, ACCENT], [1, CYAN]],
        zmin=0, zmax=1, showscale=False, hovertemplate="True %{y}<br>Predicted %{x}<br>"
                                                         "%{z:.0%}<extra></extra>"))
    for i in range(len(classes)):          # emphasise the diagonal (correct answers)
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color=CYAN, width=2.5))
    fig.update_yaxes(autorange="reversed")
    return fig


def importance(items: list[tuple[str, float]], names: dict[str, str]) -> go.Figure:
    items = list(items)[::-1]
    fig = _fig(380, "", showlegend=False, margin=dict(l=260, r=30, t=10, b=50))
    fig.add_trace(go.Bar(x=[v for _, v in items], y=[names.get(f, f) for f, _ in items],
                         orientation="h",
                         marker=dict(color=rgba(ACCENT, 0.75), line=dict(color=CYAN, width=1))))
    fig.update_xaxes(title="Mean |SHAP| (summed over classes)")
    return fig
