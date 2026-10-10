"""Reusable UI pieces: verdict card, step indicator, key facts, star card, errors."""

from __future__ import annotations

import html
import math
import traceback
from typing import Any

import streamlit as st

from ui import text as T
from ui.explain import catalog_statement, reasons


def badge(cls: str, big: bool = False) -> str:
    """A coloured pill that always carries its text label."""
    label, bg, fg, _ = T.CLASS_INFO.get(cls, T.CLASS_INFO["NEEDS_REVIEW"])
    size = "1.25rem" if big else "0.85rem"
    pad = "0.35rem 0.9rem" if big else "0.15rem 0.6rem"
    return (f'<span style="background:{bg};color:{fg};border-radius:999px;padding:{pad};'
            f'font-weight:600;font-size:{size};white-space:nowrap">{html.escape(label)}</span>')


def step_indicator(current: int) -> None:
    parts = []
    for i, name in enumerate(T.STEP_NAMES, start=1):
        on = i <= current
        style = ("font-weight:700" if i == current else "opacity:0.55" if not on else "")
        parts.append(f'<span style="{style}">{i} {html.escape(name)}</span>')
    st.markdown(" &nbsp;&rarr;&nbsp; ".join(parts), unsafe_allow_html=True)


def confidence_meter(p: float | None, label: str) -> None:
    filled = {"High": 3, "Medium": 2}.get(label, 1)
    dots = "●" * filled + "○" * (3 - filled)
    st.markdown(f"**Confidence:** {label} &nbsp; <span aria-hidden='true'>{dots}</span>",
                unsafe_allow_html=True, help=T.GLOSSARY["confidence"])


def verdict_card(report: dict[str, Any]) -> None:
    routing = report.get("routing") or {}
    cls = routing.get("primary_class") or "NEEDS_REVIEW"
    with st.container(border=True):
        st.caption(T.VERDICT_TITLE)
        st.markdown(badge(cls, big=True), unsafe_allow_html=True)
        st.markdown(f"### {html.escape(report.get('plain_summary') or '')}")
        confidence_meter(routing.get("top_probability"), routing.get("confidence_label", "Low"))


def why_block(report: dict[str, Any]) -> None:
    st.subheader(T.WHY_TITLE)
    rs = reasons(report)
    if not rs:
        st.write("No single measurement stands out; the verdict comes from many small clues.")
    for r in rs:
        st.markdown(f"- {r}")


def catalog_block(report: dict[str, Any]) -> None:
    res = catalog_statement(report)
    if not res:
        return
    kind, sentence = res
    st.subheader(T.CATALOG_TITLE)
    icon = {"agree": "✔", "disagree": "✘", "unsure": "?", "none": "–"}[kind]
    st.markdown(f"{icon} {sentence}")


def key_facts(facts: list) -> None:
    if not facts:
        return
    st.subheader(T.FACTS_TITLE)
    cols = st.columns(3)
    for i, f in enumerate(facts):
        label, value, unit, tip = (list(f) + ["", "", "", ""])[:4]
        with cols[i % 3]:
            st.metric(label, f"{value} {unit}".strip() if value not in ("", None) else "unknown",
                      help=tip or None)


def temperature_words(teff: float | None) -> str:
    if teff is None or not math.isfinite(teff):
        return "unknown"
    if teff < 3900:
        return "much cooler than the Sun (red)"
    if teff < 5300:
        return "cooler than the Sun (orange)"
    if teff < 6200:
        return "Sun-like (yellow-white)"
    if teff < 7500:
        return "hotter than the Sun (white)"
    return "much hotter than the Sun (blue-white)"


def temperature_colour(teff: float | None) -> str:
    if teff is None or not math.isfinite(teff):
        return "#999999"
    for limit, colour in ((3900, "#ff7f4f"), (5300, "#ffb36b"), (6200, "#fff1c9"),
                          (7500, "#f4f6ff"), (99999, "#a9c4ff")):
        if teff < limit:
            return colour
    return "#999999"


def star_card(star: dict[str, Any], sectors: list[dict[str, Any]],
              available: list[dict[str, Any]] | None = None) -> None:
    name = star.get("common_name") or f"TIC {star.get('tic_id')}"
    teff, rad = star.get("teff"), star.get("rad")
    with st.container(border=True):
        st.caption(T.STAR_CARD_TITLE)
        swatch = (f'<span style="display:inline-block;width:0.9rem;height:0.9rem;'
                  f'border-radius:50%;background:{temperature_colour(teff)};'
                  f'border:1px solid #888;vertical-align:middle"></span>')
        st.markdown(f"**{html.escape(name)}** &nbsp; TIC {star.get('tic_id')} &nbsp; {swatch}",
                    unsafe_allow_html=True, help=T.GLOSSARY["TIC ID"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(T.STAR_BRIGHTNESS, _fmt(star.get("tmag"), 1), help=T.STAR_BRIGHTNESS_HELP)
        c2.metric(T.STAR_TEMP, f"{_fmt(teff, 0)} K",
                  help=temperature_words(teff).capitalize())
        c3.metric(T.STAR_SIZE, f"{_fmt(rad, 2)} × Sun",
                  help="The star's radius compared with the Sun's.")
        secs = available if available is not None else sectors
        c4.metric(T.STAR_SECTORS, str(len(secs)), help=T.STAR_SECTORS_HELP)
        sector_timeline(secs, used={s.get("sector") for s in sectors})


def sector_timeline(sectors: list[dict[str, Any]], used: set | None = None) -> None:
    """A strip of small boxes, one per TESS sector, filled where data exist."""
    nums = sorted({int(s["sector"]) for s in sectors if s.get("sector") is not None})
    if not nums:
        return
    used = used or set()
    hi = max(max(nums), 1)
    cells = []
    for s in range(1, hi + 1):
        if s in used:
            colour, title = "#0072B2", f"Sector {s}: used"
        elif s in nums:
            colour, title = "#8fb8de", f"Sector {s}: available"
        else:
            colour, title = "rgba(128,128,128,0.18)", f"Sector {s}"
        cells.append(f'<span title="{title}" style="display:inline-block;width:6px;height:14px;'
                     f'margin-right:1px;background:{colour}"></span>')
    st.markdown(f'<div style="line-height:14px">{"".join(cells)}</div>'
                f'<div style="font-size:0.75rem;opacity:0.7">Sectors 1-{hi}: dark = used, '
                f'light = available</div>', unsafe_allow_html=True)


def friendly_error(kind: str, err: BaseException | None = None) -> None:
    title, body, hint = T.ERRORS.get(kind, T.ERRORS["generic"])
    st.error(f"**{title}**\n\n{body}\n\n{hint}")
    if err is not None:
        with st.expander(T.TECH_DETAILS):
            st.code("".join(traceback.format_exception(type(err), err, err.__traceback__))[-4000:])


def error_kind(err: BaseException) -> str:
    """Map an exception to one of the friendly error kinds."""
    from src.errors import DataDownloadError, NoDataError, TargetNotFoundError

    if isinstance(err, TargetNotFoundError):
        return "unknown_star"
    if isinstance(err, NoDataError):
        return "no_data"
    msg = str(err).lower()
    if isinstance(err, (DataDownloadError, TimeoutError)) or "timeout" in msg \
            or "timed out" in msg:
        return "timeout"
    return "generic"


def glossary(terms: list[str]) -> None:
    with st.expander("Words used on this page"):
        for t in terms:
            st.markdown(f"**{t}**: {T.GLOSSARY[t]}")


def _fmt(v: Any, digits: int) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "unknown"
    return f"{v:.{digits}f}" if math.isfinite(v) else "unknown"


def html_report(report: dict[str, Any], figures: list[tuple[str, Any]]) -> str:
    """A single self-contained HTML summary with the charts."""
    routing = report.get("routing") or {}
    cls = routing.get("primary_class") or "NEEDS_REVIEW"
    name = (report.get("star") or {}).get("common_name") or report.get("target")
    parts = [f"<h1>{html.escape(str(name))}</h1>", badge(cls, big=True),
             f"<h2>{html.escape(report.get('plain_summary') or '')}</h2>",
             f"<p><b>Confidence:</b> {routing.get('confidence_label', '')}</p>",
             f"<h3>{T.WHY_TITLE}</h3><ul>"]
    parts += [f"<li>{html.escape(r)}</li>" for r in reasons(report)]
    parts.append(f"</ul><h3>{T.FACTS_TITLE}</h3><table>")
    for f in report.get("key_facts") or []:
        label, value, unit = (list(f) + ["", "", ""])[:3]
        parts.append(f"<tr><td>{html.escape(str(label))}</td>"
                     f"<td><b>{html.escape(str(value))} {html.escape(str(unit))}</b></td></tr>")
    parts.append("</table>")
    first = True
    for title, fig in figures:
        if fig is None:
            continue
        parts.append(f"<h3>{html.escape(title)}</h3>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="inline" if first else False))
        first = False
    parts.append(f"<p style='color:#666'>{html.escape(T.FOOTER)}</p>")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>"
            f"{html.escape(str(name))}</title><style>body{{font-family:sans-serif;max-width:900px;"
            "margin:2rem auto;padding:0 1rem}}td{{padding:4px 12px}}</style></head><body>"
            + "".join(parts) + "</body></html>")
