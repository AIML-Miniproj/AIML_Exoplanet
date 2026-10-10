"""The app's look: class colours, one CSS block, the hero header and the verdict card.

Pure CSS (no JavaScript). Animations are slow and subtle and switch off when the
viewer asks for reduced motion.
"""

from __future__ import annotations

import html

# Class -> (label, colour, icon). A colour is always shown with its text label.
CLASSES = {
    "PLANET": ("Planet", "#22D3EE", "🪐"),
    "EB": ("Eclipsing binary", "#F59E0B", "🌗"),
    "VARIABLE": ("Variable star", "#E879F9", "✨"),
    "NOISE": ("Noise", "#94A3B8", "〰️"),
    "NEEDS_REVIEW": ("Needs review", "#A78BFA", "❓"),
}
VIOLET, CYAN = "#7C5CFF", "#22D3EE"


def label(cls: str) -> str:
    return CLASSES.get(cls, (cls.title(), "", ""))[0]


def color(cls: str) -> str:
    return CLASSES.get(cls, ("", VIOLET, ""))[1]


def icon(cls: str) -> str:
    return CLASSES.get(cls, ("", "", "•"))[2]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# Tiny stars as repeating radial gradients; each layer drifts by exactly its tile
# height per loop, so the motion is seamless. Different speeds give parallax.
_STARS_SMALL = ", ".join(
    f"radial-gradient(1px 1px at {x}px {y}px, rgba(255,255,255,{a}), transparent)"
    for x, y, a in [(20, 30, .8), (140, 80, .6), (60, 170, .7), (230, 210, .5), (90, 260, .6),
                    (270, 40, .7), (180, 140, .5), (10, 290, .6), (300, 120, .4), (120, 220, .5)])
_STARS_MID = ", ".join(
    f"radial-gradient(1.5px 1.5px at {x}px {y}px, rgba({c},{a}), transparent)"
    for x, y, c, a in [(50, 400, "200,210,255", .8), (330, 90, "255,255,255", .7),
                       (420, 300, "190,240,255", .7), (200, 470, "230,200,255", .6),
                       (480, 180, "255,255,255", .6), (120, 30, "200,230,255", .5)])
_STARS_BRIGHT = ", ".join(
    f"radial-gradient(2px 2px at {x}px {y}px, rgba(255,255,255,{a}), transparent)"
    for x, y, a in [(150, 600, .9), (620, 120, .8), (480, 520, .7), (60, 260, .8)])

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
  --violet: {VIOLET}; --cyan: {CYAN}; --text: #E6E9FF; --muted: #B4BAE0;
  --glass: rgba(13,16,38,0.62); --glass-border: rgba(124,92,255,0.28);
  --cls: {VIOLET};
}}
html, body, [class*="css"], .stMarkdown, .stText, p, li, td, th, label {{
  font-family: 'Inter', system-ui, sans-serif;
}}
.stApp {{
  background:
    radial-gradient(ellipse 900px 600px at 100% 0%, rgba(124,92,255,0.16), transparent 70%),
    radial-gradient(ellipse 700px 500px at 92% 8%, rgba(34,211,238,0.07), transparent 70%),
    #05060F;
}}
.stApp::before, .stApp::after {{
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
}}
.stApp::before {{
  background-image: {_STARS_SMALL}, {_STARS_MID};
  background-size: 320px 320px, 520px 520px;
  animation: drift-a 240s linear infinite;
  opacity: 0.55;
}}
.stApp::after {{
  background-image: {_STARS_BRIGHT};
  background-size: 700px 700px;
  animation: drift-b 300s linear infinite;
  opacity: 0.45;
}}
@keyframes drift-a {{ from {{ background-position: 0 0, 0 0; }}
                     to {{ background-position: 0 -320px, 0 -1040px; }} }}
@keyframes drift-b {{ from {{ background-position: 0 0; }} to {{ background-position: 0 -1400px; }} }}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
  background: transparent; z-index: 1;
}}
[data-testid="stMain"] {{ background: transparent; }}

/* Readable on a dim projector: body text at least 15px */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{ font-size: 16px; line-height: 1.55; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  font-size: 15px !important; color: var(--muted) !important;
}}
h1, h2, h3, h4 {{ font-family: 'Orbitron', 'Inter', sans-serif !important; letter-spacing: 0.06em; }}
h3 {{ text-transform: uppercase; font-size: 1.05rem !important; color: #CFD4FF; }}
.block-container {{ max-width: 1500px; padding-top: 2.2rem; }}

/* Sidebar: glass panel */
[data-testid="stSidebar"] > div:first-child {{
  background: rgba(10,12,30,0.55);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  border-right: 1px solid rgba(124,92,255,0.35);
  box-shadow: 0 0 24px rgba(124,92,255,0.18);
}}
[data-testid="stSidebar"] {{ background: transparent; }}
.side-title {{ font-family: 'Orbitron', sans-serif; letter-spacing: .12em; font-size: 13px;
  text-transform: uppercase; color: #CFD4FF; margin: 4px 0 6px; }}
.star-desc {{ font-size: 15px; color: var(--muted); line-height: 1.5; margin: 2px 0 10px; }}

/* Analyze button: violet -> cyan gradient with a glow */
[data-testid="stSidebar"] .stButton button {{
  background: linear-gradient(90deg, var(--violet), var(--cyan)); color: #05060F;
  font-family: 'Orbitron', sans-serif; font-weight: 700; letter-spacing: .08em;
  border: 0; border-radius: 12px; padding: .6rem 1rem;
  transition: box-shadow .25s ease, transform .25s ease;
}}
[data-testid="stSidebar"] .stButton button p {{ font-size: 16px; color: #05060F; font-weight: 700; }}
[data-testid="stSidebar"] .stButton button:hover {{
  box-shadow: 0 0 18px rgba(124,92,255,.7), 0 0 30px rgba(34,211,238,.35);
  transform: translateY(-1px); color: #05060F;
}}

/* Tabs: pills with a glowing active border */
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: 0; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{
  background: rgba(13,16,38,0.6); border: 1px solid rgba(124,92,255,0.25);
  border-radius: 999px; padding: 8px 18px; height: auto;
  transition: box-shadow .2s ease, border-color .2s ease;
}}
.stTabs [data-baseweb="tab"] p {{
  font-family: 'Orbitron', sans-serif; letter-spacing: .06em; font-size: 14px;
}}
.stTabs [data-baseweb="tab"]:hover {{ box-shadow: 0 0 12px rgba(124,92,255,.45); }}
.stTabs [aria-selected="true"] {{
  border-color: var(--cyan); box-shadow: 0 0 14px rgba(34,211,238,.45);
  background: rgba(124,92,255,0.18);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ animation: fade-in .45s ease both; padding-top: 14px; }}
@keyframes fade-in {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}

/* Glass cards: every st.container(key="glass_...") */
[class*="st-key-glass"] {{
  background: var(--glass); border: 1px solid rgba(124,92,255,0.22);
  border-radius: 16px; padding: 18px 20px;
  box-shadow: 0 0 22px rgba(124,92,255,0.10), inset 0 0 0 1px rgba(255,255,255,0.02);
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
}}
[class*="st-key-glass-cls"] {{
  border-color: color-mix(in srgb, var(--cls) 45%, transparent);
  box-shadow: 0 0 26px color-mix(in srgb, var(--cls) 18%, transparent);
}}

/* HUD metric readouts */
[data-testid="stMetric"] {{
  background: rgba(8,10,26,0.7); border: 1px solid rgba(124,92,255,0.22);
  border-top: 2px solid var(--cyan); border-radius: 12px; padding: 12px 14px;
  box-shadow: 0 -2px 14px rgba(34,211,238,0.18);
}}
[data-testid="stMetricLabel"] p {{
  text-transform: uppercase; letter-spacing: .12em; font-size: 13px !important; color: var(--muted);
}}
[data-testid="stMetricValue"] {{
  font-family: 'JetBrains Mono', monospace !important; font-size: 1.7rem !important; color: #F4F6FF;
}}

/* Tables (markdown and st.table) */
.stMarkdown table, [data-testid="stTable"] table {{
  border-collapse: collapse; width: 100%; font-size: 15px;
}}
.stMarkdown th, [data-testid="stTable"] th {{
  font-family: 'Orbitron', sans-serif; letter-spacing: .06em; font-size: 13px;
  text-transform: uppercase; color: #CFD4FF; background: rgba(124,92,255,0.12);
}}
.stMarkdown td, .stMarkdown th, [data-testid="stTable"] td, [data-testid="stTable"] th {{
  border: 1px solid rgba(124,92,255,0.18) !important; padding: 6px 10px;
}}
.stMarkdown td code {{ font-family: 'JetBrains Mono', monospace; color: #E6E9FF;
  background: rgba(124,92,255,0.12); }}

/* Progress bar in the gradient */
[data-testid="stProgress"] div[role="progressbar"] > div > div > div {{
  background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
}}

/* Hero */
.hero {{ display: flex; align-items: center; gap: 22px; margin: 0 0 6px; flex-wrap: wrap; }}
.hero-title {{
  font-family: 'Orbitron', sans-serif; font-weight: 800; letter-spacing: .07em;
  font-size: clamp(1.5rem, 2.6vw, 2.5rem); line-height: 1.15; margin: 0;
  background: linear-gradient(90deg, var(--violet), var(--cyan), var(--violet));
  background-size: 200% auto; -webkit-background-clip: text; background-clip: text;
  color: transparent; animation: shimmer 9s linear infinite;
  filter: drop-shadow(0 0 14px rgba(124,92,255,0.45));
}}
@keyframes shimmer {{ to {{ background-position: 200% center; }} }}
.hero-sub {{ font-size: 17px; color: #CDD2F5; margin: 4px 0 10px; }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }}
.chip {{ font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #DDE2FF;
  border: 1px solid rgba(124,92,255,0.4); border-radius: 999px; padding: 3px 12px;
  background: rgba(124,92,255,0.10); }}
.transit svg {{ display: block; }}
.transit .star {{ animation: dim 7s ease-in-out infinite; transform-origin: 50px 50px; }}
.transit .planet {{ animation: cross 7s linear infinite; }}
@keyframes cross {{ 0% {{ transform: translateX(-62px); }} 100% {{ transform: translateX(62px); }} }}
@keyframes dim {{ 0%, 30%, 70%, 100% {{ opacity: 1; }} 42%, 58% {{ opacity: .72; }} }}

/* Verdict card */
.verdict {{
  display: flex; align-items: center; gap: 18px; padding: 18px 20px; border-radius: 16px;
  border: 1.5px solid var(--vc); background: linear-gradient(135deg, rgba(13,16,38,.85), rgba(13,16,38,.55));
  box-shadow: 0 0 24px color-mix(in srgb, var(--vc) 35%, transparent);
  animation: pulse 1.6s ease-out 1;
}}
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--vc) 60%, transparent); }}
  100% {{ box-shadow: 0 0 24px color-mix(in srgb, var(--vc) 35%, transparent); }}
}}
.v-icon {{ font-size: 2.6rem; filter: drop-shadow(0 0 10px var(--vc)); }}
.v-main {{ flex: 1; min-width: 0; }}
.v-kicker {{ font-size: 13px; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }}
.v-class {{ font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 1.8rem;
  letter-spacing: .06em; color: var(--vc); text-shadow: 0 0 14px color-mix(in srgb, var(--vc) 50%, transparent); }}
.v-conf {{ font-size: 15px; color: #DDE2FF; }}
.v-ring {{ --p: 50; width: 84px; height: 84px; border-radius: 50%; flex: none;
  background: conic-gradient(var(--vc) calc(var(--p) * 1%), rgba(255,255,255,0.08) 0);
  display: grid; place-items: center; box-shadow: 0 0 16px color-mix(in srgb, var(--vc) 40%, transparent); }}
.v-ring span {{ width: 66px; height: 66px; border-radius: 50%; background: #0A0C1E; display: grid;
  place-items: center; font-family: 'JetBrains Mono', monospace; font-size: 17px; color: #F4F6FF; }}
.v-summary {{ font-size: 18px !important; line-height: 1.55; color: #EEF0FF; margin: 14px 2px 4px; }}
.catalog {{ font-size: 16px; padding: 10px 14px; border-radius: 12px; margin-top: 8px;
  background: rgba(8,10,26,0.6); border: 1px solid rgba(124,92,255,0.22); }}
.hint {{ font-size: 16px; padding: 10px 14px; border-radius: 12px; margin: 6px 0;
  background: rgba(167,139,250,0.12); border: 1px solid rgba(167,139,250,0.45); }}

@media (prefers-reduced-motion: reduce) {{
  .stApp::before, .stApp::after, .hero-title, .transit .star, .transit .planet, .verdict,
  .stTabs [data-baseweb="tab-panel"] {{ animation: none !important; }}
}}
</style>
"""

# A star with a planet crossing it; the star dims while the planet is in front.
TRANSIT_SVG = """
<div class="transit" aria-hidden="true">
<svg width="100" height="100" viewBox="0 0 100 100">
  <defs>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFF6D8"/><stop offset="45%" stop-color="#FFD27A"/>
      <stop offset="70%" stop-color="#F59E0B" stop-opacity="0.55"/>
      <stop offset="100%" stop-color="#7C5CFF" stop-opacity="0"/>
    </radialGradient>
    <clipPath id="sky"><rect width="100" height="100"/></clipPath>
  </defs>
  <g class="star"><circle cx="50" cy="50" r="46" fill="url(#glow)"/>
    <circle cx="50" cy="50" r="22" fill="#FFF3CC"/></g>
  <g clip-path="url(#sky)"><circle class="planet" cx="50" cy="52" r="7" fill="#0A0C1E"
    stroke="#22D3EE" stroke-width="1" stroke-opacity="0.8"/></g>
</svg></div>
"""


def hero_html(title: str, subtitle: str, chips: list[str]) -> str:
    chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    return (f'<div class="hero">{TRANSIT_SVG}<h1 class="hero-title">{html.escape(title)}</h1>'
            f'</div><div class="hero-sub">{html.escape(subtitle)}</div>'
            f'<div class="chips">{chips_html}</div>')


def verdict_html(cls: str, confidence: str, prob: float | None, summary: str) -> str:
    pct = "" if prob is None else f"{100 * prob:.0f}%"
    ring = (f'<div class="v-ring" style="--p:{0 if prob is None else 100 * prob:.0f}">'
            f'<span>{pct or "–"}</span></div>')
    return (f'<div class="verdict" style="--vc:{color(cls)}">'
            f'<div class="v-icon">{icon(cls)}</div>'
            f'<div class="v-main"><div class="v-kicker">Verdict</div>'
            f'<div class="v-class">{html.escape(label(cls).upper())}</div>'
            f'<div class="v-conf">Confidence: <b>{html.escape(confidence or "unknown")}</b>'
            f'{" · class probability " + pct if pct else ""}</div></div>{ring}</div>'
            f'<p class="v-summary">{html.escape(summary or "")}</p>')


def class_color_css(cls: str) -> str:
    """Sets the colour used by the class-tinted glass cards."""
    return f"<style>:root {{ --cls: {color(cls)}; }}</style>"
