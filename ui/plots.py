"""Plotly figures built from a report's ``plot_data`` (already thinned for display)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from ui.text import CLASS_INFO

DOT = "rgba(128,128,128,0.45)"
LINE = "#333333"
SECTOR_FILL = "rgba(128,128,128,0.10)"
DIP = "#D62728"


def _layout(fig: go.Figure, x: str, y: str, height: int = 340) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_title=x, yaxis_title=y, template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def class_colour(report: dict[str, Any]) -> str:
    cls = (report.get("routing") or {}).get("primary_class") or "NEEDS_REVIEW"
    return CLASS_INFO.get(cls, CLASS_INFO["NEEDS_REVIEW"])[1]


def light_curve(report: dict[str, Any]) -> go.Figure | None:
    """Full light curve with sectors shaded and labelled and predicted dips marked."""
    pdata = (report.get("plot_data") or {}).get("lc_norm") or {}
    t, f = np.asarray(pdata.get("time", []), float), np.asarray(pdata.get("flux", []), float)
    if t.size == 0:
        return None
    sec = np.asarray(pdata.get("sector", [0] * t.size))
    # TESS sectors are years apart: plot days since each sector began, side by side
    fig = go.Figure()
    offset, ticks = 0.0, []
    sig = (report.get("signals") or [{}])[0]
    for s in np.unique(sec):
        m = sec == s
        ts = t[m]
        x0 = ts.min()
        x = ts - x0 + offset
        fig.add_vrect(x0=x.min(), x1=x.max(), fillcolor=SECTOR_FILL, line_width=0,
                      annotation_text=f"Sector {int(s)}", annotation_position="top left",
                      annotation_font_size=10)
        fig.add_trace(go.Scattergl(x=x, y=f[m], mode="markers",
                                   marker=dict(size=2, color=DOT), hoverinfo="skip"))
        p, t0 = sig.get("period"), sig.get("t0")
        if p and t0 and p > 0:
            n0, n1 = math.ceil((ts.min() - t0) / p), math.floor((ts.max() - t0) / p)
            for n in range(n0, n1 + 1):
                ticks.append(t0 + n * p - x0 + offset)
        offset = x.max() + 2.0
    if ticks and len(ticks) < 400:
        lo = float(np.nanpercentile(f, 0.5))
        fig.add_trace(go.Scatter(x=ticks, y=[lo] * len(ticks), mode="markers",
                                 marker=dict(symbol="triangle-up", size=7, color=DIP),
                                 hovertemplate="predicted dip<extra></extra>"))
    return _layout(fig, "Days (sectors placed side by side)", "Brightness (1 = normal)")


def fold(report: dict[str, Any]) -> go.Figure | None:
    """All dips stacked: measurements, binned average, fitted dip shape."""
    pdata = report.get("plot_data") or {}
    fz = pdata.get("fold_zoom") or {}
    trap = pdata.get("trapezoid") or {}
    x, y = np.asarray(fz.get("hours", []), float), np.asarray(fz.get("flux", []), float)
    if x.size == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", marker=dict(size=3, color=DOT),
                               hoverinfo="skip"))
    edges = np.linspace(x.min(), x.max(), 60)
    idx = np.digitize(x, edges)
    xb = [x[idx == i].mean() for i in range(1, len(edges)) if (idx == i).sum() >= 3]
    yb = [y[idx == i].mean() for i in range(1, len(edges)) if (idx == i).sum() >= 3]
    fig.add_trace(go.Scatter(x=xb, y=yb, mode="lines", line=dict(color=LINE, width=2),
                             hovertemplate="%{x:.1f} h: %{y:.5f}<extra>average</extra>"))
    if trap.get("x") and trap.get("model"):
        fig.add_trace(go.Scatter(x=np.asarray(trap["x"]) * 24, y=trap["model"], mode="lines",
                                 line=dict(color=class_colour(report), width=3),
                                 hovertemplate="fitted dip<extra></extra>"))
    return _layout(fig, "Hours from the middle of the dip", "Brightness (1 = normal)")


def planet_size(report: dict[str, Any]) -> go.Figure | None:
    mod = (report.get("modules") or {}).get("planet") or {}
    mc = (mod.get("details") or {}).get("monte_carlo", {}).get("rp_rearth")
    if not mc or not all(v is not None and math.isfinite(v) for v in mc):
        return None
    lo, med, hi = mc
    names = ["Earth", "Neptune", "Jupiter", "This object"]
    sizes = [1.0, 3.88, 11.2, med]
    colours = ["#7F7F7F", "#7F7F7F", "#7F7F7F", class_colour(report)]
    fig = go.Figure(go.Bar(x=names, y=sizes, marker_color=colours,
                           error_y=dict(type="data", symmetric=False, array=[0, 0, 0, hi - med],
                                        arrayminus=[0, 0, 0, med - lo]),
                           text=[f"{v:.1f}" for v in sizes], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f} Earth radii<extra></extra>"))
    return _layout(fig, "", "Size (Earth = 1)", height=300)


def eb_eclipses(report: dict[str, Any]) -> go.Figure | None:
    """Primary and secondary eclipse side by side, at the binary's true period."""
    from plotly.subplots import make_subplots

    mod = (report.get("modules") or {}).get("eb") or {}
    det = mod.get("details") or {}
    pdata = (report.get("plot_data") or {}).get("fold") or {}
    ph, fl = np.asarray(pdata.get("phase", []), float), np.asarray(pdata.get("flux", []), float)
    sig = (report.get("signals") or [{}])[0]
    if ph.size == 0 or not det:
        return None
    factor = (det.get("true_period") or {}).get("factor", 1)
    p_det, dur = sig.get("period", 1.0), sig.get("duration", 0.1)
    # The stored fold is at the detected period; at twice it, alternate dips are
    # primary and secondary: show the detected dip and the stretch half an orbit later.
    sec_phase = (det.get("eclipses") or {}).get("secondary_phase")
    win = min(0.25, 2.5 * dur / p_det)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=("Main eclipse", "Half an orbit later"))
    m1 = np.abs(ph) < win
    fig.add_trace(go.Scattergl(x=ph[m1] * p_det * 24, y=fl[m1], mode="markers",
                               marker=dict(size=3, color=class_colour(report))), 1, 1)
    if factor == 2:
        m2 = m1                       # folded together: the other eclipse is the same dip
        title_note = "(at the detected period both eclipses overlap)"
    else:
        c = (sec_phase if sec_phase is not None and math.isfinite(sec_phase) else 0.5)
        c = c - 1 if c > 0.5 else c
        m2 = np.abs(ph - c) < win
        title_note = ""
    x2 = (ph[m2] - (0 if factor == 2 else (c if factor != 2 else 0))) * p_det * 24
    fig.add_trace(go.Scattergl(x=x2, y=fl[m2], mode="markers",
                               marker=dict(size=3, color="#7F7F7F")), 1, 2)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Hours from eclipse centre " + title_note)
    fig.update_yaxes(title_text="Brightness", col=1)
    return fig


def periodogram_ls(report: dict[str, Any]) -> go.Figure | None:
    ls = (report.get("plot_data") or {}).get("ls") or {}
    p, pw = np.asarray(ls.get("period", []), float), np.asarray(ls.get("power", []), float)
    if p.size == 0:
        return None
    feats = report.get("features") or {}
    main = feats.get("var_period_d") or feats.get("ls_period_1")
    fig = go.Figure(go.Scattergl(x=p, y=pw, mode="lines", line=dict(color=LINE, width=1)))
    if main:
        fig.add_vline(x=main, line=dict(color=class_colour(report), width=3, dash="dot"),
                      annotation_text=f"main period {main:.3f} d")
    fig.update_xaxes(type="log")
    return _layout(fig, "Period (days)", "Strength of the rhythm")


def periodogram_bls(report: dict[str, Any]) -> go.Figure | None:
    pg = (report.get("plot_data") or {}).get("periodogram") or {}
    p, pw = np.asarray(pg.get("period", []), float), np.asarray(pg.get("power", []), float)
    if p.size == 0:
        return None
    sig = (report.get("signals") or [{}])[0]
    fig = go.Figure(go.Scattergl(x=p, y=pw, mode="lines", line=dict(color=LINE, width=1)))
    if sig.get("period"):
        fig.add_vline(x=sig["period"], line=dict(color=DIP, dash="dot"),
                      annotation_text=f"best {sig['period']:.4f} d")
    fig.update_xaxes(type="log")
    return _layout(fig, "Trial period (days)", "Dip-search power")


def per_sector_depths(report: dict[str, Any]) -> go.Figure | None:
    rows = (report.get("plot_data") or {}).get("per_sector") or []
    rows = [r for r in rows if r.get("depth") is not None]
    if not rows:
        return None
    fig = go.Figure(go.Bar(x=[f"S{r.get('sector')}" for r in rows],
                           y=[r["depth"] * 1e6 for r in rows],
                           error_y=dict(array=[(r.get("depth_err") or 0) * 1e6 for r in rows]),
                           marker_color=class_colour(report)))
    return _layout(fig, "Sector", "Dip depth (ppm)", height=280)


def module_chart(report: dict[str, Any]) -> tuple[str, go.Figure | None]:
    """The chart for the routed class: planet size, EB eclipses or variable rhythms."""
    mods = (report.get("routing") or {}).get("modules") or []
    first = mods[0] if mods else None
    if first == "planet":
        return "planet", planet_size(report)
    if first == "eb":
        return "eb", eb_eclipses(report)
    if first == "variable":
        return "variable", periodogram_ls(report)
    return "variable", periodogram_ls(report)
