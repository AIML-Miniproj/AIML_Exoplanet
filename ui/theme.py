"""The app's look: class colours and icons, one CSS block, the hero header and the verdict card.

Styled after the Stitch "Galactic Quiet Luxury" design used by the other pages:
deep navy, frosted glass panels with hairline borders, light wide-tracked
Montserrat headings, Inter text, a sapphire accent and Material Symbols icons
(no emoji). Pure CSS (no JavaScript); animations are slow and subtle and switch
off when the viewer asks for reduced motion.
"""

from __future__ import annotations

import html

# Class -> (label, colour, Material Symbols icon). A colour is always shown with its label.
CLASSES = {
    "PLANET": ("Planet", "#22D3EE", "public"),
    "EB": ("Eclipsing binary", "#F59E0B", "contrast"),
    "VARIABLE": ("Variable star", "#EC4899", "auto_awesome"),
    "NOISE": ("Noise", "#94A3B8", "graphic_eq"),
    "NEEDS_REVIEW": ("Needs review", "#A78BFA", "help"),
}
ACCENT, CYAN = "#3B82F6", "#22D3EE"          # sapphire accent, electric cyan


def label(cls: str) -> str:
    return CLASSES.get(cls, (cls.title(), "", ""))[0]


def color(cls: str) -> str:
    return CLASSES.get(cls, ("", ACCENT, ""))[1]


def icon(cls: str) -> str:
    """Material Symbols name for the class (render with icon_html)."""
    return CLASSES.get(cls, ("", "", "radio_button_unchecked"))[2]


def icon_html(name: str, col: str | None = None, size: int = 18) -> str:
    style = f"font-size:{size}px;" + (f"color:{col};" if col else "")
    return f'<span class="msi" style="{style}" aria-hidden="true">{html.escape(name)}</span>'


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# Faint stars as repeating radial gradients; each layer drifts by exactly its tile
# height per loop, so the motion is seamless.
_STARS_SMALL = ", ".join(
    f"radial-gradient(1px 1px at {x}px {y}px, rgba(255,255,255,{a}), transparent)"
    for x, y, a in [(20, 30, .7), (140, 80, .5), (60, 170, .6), (230, 210, .4), (90, 260, .5),
                    (270, 40, .6), (180, 140, .4), (10, 290, .5), (300, 120, .35), (120, 220, .4)])
_STARS_BRIGHT = ", ".join(
    f"radial-gradient(1.5px 1.5px at {x}px {y}px, rgba(255,255,255,{a}), transparent)"
    for x, y, a in [(150, 600, .8), (620, 120, .7), (480, 520, .6), (60, 260, .7)])

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@200;300;400;500;600&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,300..500,0..1,0&display=swap');

:root {{
  --accent: {ACCENT}; --sapphire: #2563EB; --cyan: {CYAN};
  --text: #DDE2F5; --text-hi: #F2F5FA; --muted: #9AA8BA; --dim: #475569;
  --frame: #070A11; --panel: #090D17; --inset: #04060C; --border: #1A2333; --border-hi: #202A38;
  --glass: #090D17; --glass-hi: #0D1424;
  --hair: #1A2333; --hair-blue: #202A38;
  --cls: {ACCENT};
}}
html, body, [class*="css"], .stMarkdown, .stText, p, li, td, th, label {{
  font-family: 'Inter', system-ui, sans-serif;
}}
.msi {{ font-family: 'Material Symbols Outlined'; font-weight: 400; font-style: normal; line-height: 1;
  display: inline-block; vertical-align: -0.2em; letter-spacing: normal; text-transform: none;
  font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }}
.stApp {{ background: #030407; }}
.stApp::before, .stApp::after {{
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
}}
.stApp::before {{
  background-image: {_STARS_SMALL}; background-size: 320px 320px;
  animation: drift-a 260s linear infinite; opacity: 0.45;
}}
.stApp::after {{
  background-image: {_STARS_BRIGHT}; background-size: 700px 700px;
  animation: drift-b 320s linear infinite; opacity: 0.35;
}}
@keyframes drift-a {{ from {{ background-position: 0 0; }} to {{ background-position: 0 -320px; }} }}
@keyframes drift-b {{ from {{ background-position: 0 0; }} to {{ background-position: 0 -1400px; }} }}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: transparent; z-index: 1; }}
[data-testid="stMain"] {{ background: transparent; }}

/* Readable on a dim projector: body text at least 15px */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {{ font-size: 15.5px; line-height: 1.6; color: var(--text); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  font-size: 14.5px !important; color: var(--muted) !important;
}}
h1, h2, h3, h4 {{ font-family: 'Montserrat', 'Inter', sans-serif !important; font-weight: 300 !important;
  letter-spacing: 0.14em; color: var(--text-hi); }}
h3 {{ text-transform: uppercase; font-size: 1.05rem !important; font-weight: 400 !important; letter-spacing: .2em; }}
/* The page sits in a rounded "viewport frame", like the Stitch pages */
.block-container {{
  max-width: 1440px; margin: 24px auto 32px; padding: 1.6rem 2.6rem 2.4rem !important;
  background: var(--frame); border: 1px solid #161D2B; border-radius: 28px;
  box-shadow: 0 20px 80px rgba(0,0,0,0.85); position: relative;
  overflow-x: clip; min-height: auto;   /* clip the glows sideways only, so the page scrolls */
}}
.block-container::before {{ content: ""; position: absolute; top: -160px; left: -160px; width: 600px; height: 600px;
  background: rgba(37,99,235,0.10); border-radius: 50%; filter: blur(140px); pointer-events: none; }}
.block-container::after {{ content: ""; position: absolute; top: 25%; right: -160px; width: 550px; height: 550px;
  background: rgba(0,203,230,0.07); border-radius: 50%; filter: blur(160px); pointer-events: none; }}

/* Top navigation capsule (same as the Stitch pages) */
.ea-nav {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; height: 72px;
  padding: 0 14px 0 10px; margin-bottom: 26px; border-radius: 9999px; background: rgba(11,15,25,0.8);
  border: 1px solid var(--border); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.6); position: relative; z-index: 2; }}
.ea-nav nav {{ display: flex; align-items: center; gap: 4px; }}
.ea-nav nav a {{ padding: 9px 16px; border-radius: 9999px; font-family: 'Montserrat', sans-serif; font-size: 11px;
  font-weight: 500; letter-spacing: .18em; text-transform: uppercase; color: #C3C6D7 !important;
  text-decoration: none !important; white-space: nowrap; transition: background .3s ease, color .3s ease; }}
.ea-nav nav a:hover {{ background: #202944; color: #DBE1FF !important; }}
.ea-nav nav a.on {{ background: var(--sapphire); color: #EEEFFF !important; font-size: 13px; letter-spacing: .08em; }}
.ea-brand {{ font-family: 'Montserrat', sans-serif; font-weight: 300; font-size: 17px; letter-spacing: .2em;
  text-transform: uppercase; color: #DBE1FF; white-space: nowrap; }}
.ea-brand b {{ font-weight: 300; color: #5DE6FF; }}
.ea-cta {{ display: inline-flex; align-items: center; gap: 10px; padding: 5px 5px 5px 16px; border-radius: 9999px;
  background: var(--sapphire); color: #EEEFFF !important; text-decoration: none !important;
  font-family: 'Montserrat', sans-serif; font-size: 11px; font-weight: 500; letter-spacing: .18em;
  text-transform: uppercase; box-shadow: 0 0 20px rgba(37,99,235,0.6); white-space: nowrap; }}
.ea-cta i {{ width: 28px; height: 28px; border-radius: 999px; background: rgba(3,13,39,0.3); display: grid;
  place-items: center; font-style: normal; }}
@media (max-width: 1100px) {{ .ea-brand {{ display: none; }} }}

/* Selected-star picture card (same look as the Demo Stars cards) */
.star-card {{ position: relative; border-radius: 28px; overflow: hidden; border: 1px solid var(--border-hi);
  background: #05070D; aspect-ratio: 1 / 0.92; box-shadow: 0 20px 60px rgba(0,0,0,0.9), 0 0 80px rgba(84,216,232,0.08); }}
.star-card {{ animation: ea-rise .9s cubic-bezier(.2,.7,.2,1) both; }}
.star-card img {{ width: 100%; height: 100%; object-fit: cover; transition: transform .7s ease;
  animation: ea-drift 18s ease-in-out infinite alternate; }}
@keyframes ea-rise {{ from {{ opacity: 0; transform: translateY(24px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes ea-drift {{ from {{ transform: scale(1.02); }} to {{ transform: scale(1.09) translate(-1.5%, -1%); }} }}
.hero-title, .hero-sub, .chips {{ animation: ea-rise .9s cubic-bezier(.2,.7,.2,1) both; }}
.hero-sub {{ animation-delay: .12s; }} .chips {{ animation-delay: .22s; }}
.star-card:hover img {{ transform: scale(1.04); }}
.star-card .shade {{ position: absolute; inset: 0; background: linear-gradient(to top, #070A11 5%, transparent 60%); opacity: .8; }}
.star-card .top {{ position: absolute; top: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between;
  gap: 8px; align-items: center; }}
.star-card .tag {{ padding: 3px 10px; border-radius: 6px; background: rgba(3,4,7,0.8); border: 1px solid var(--border-hi);
  font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .06em; color: #DDE2F5; text-transform: uppercase; }}
.star-card .badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 9999px;
  font-family: 'Montserrat', sans-serif; font-size: 10.5px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
  color: var(--bc); background: rgba(3,4,7,0.8); border: 1px solid color-mix(in srgb, var(--bc) 40%, transparent); }}
.star-card .badge::before {{ content: ""; width: 6px; height: 6px; border-radius: 999px; background: var(--bc);
  box-shadow: 0 0 6px var(--bc); }}
.star-card .foot {{ position: absolute; left: 14px; right: 14px; bottom: 14px; padding: 12px 14px; border-radius: 18px;
  background: rgba(7,10,17,0.88); border: 1px solid var(--border-hi); backdrop-filter: blur(12px);
  display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; }}
.star-card .k {{ font-family: 'Montserrat', sans-serif; font-size: 10.5px; font-weight: 600; letter-spacing: .18em;
  text-transform: uppercase; color: #54D8E8; }}
.star-card .n {{ font-size: 15px; color: #FFFFFF; font-weight: 500; }}
.star-card .t {{ font-size: 12px; color: var(--muted); }}
.star-card .r {{ text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--muted); line-height: 1.6; }}
.star-card .r b {{ color: #54D8E8; font-weight: 500; }}
.star-credit {{ font-size: 11px; color: var(--dim); margin-top: 8px; }}

/* Page navigation (glass capsules) */
[class*="st-key-nav_"] button {{
  border-radius: 9999px; background: var(--glass); border: 1px solid var(--hair);
  backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); transition: all .25s ease;
}}
[class*="st-key-nav_"] button p {{ font-family: 'Montserrat', sans-serif; font-size: 12px; font-weight: 500;
  letter-spacing: .18em; text-transform: uppercase; color: #C3C6D7; white-space: nowrap; }}
[class*="st-key-nav_"] button:hover {{ border-color: rgba(59,130,246,0.45); background: var(--glass-hi); }}
[class*="st-key-nav_"] button:hover p {{ color: var(--text-hi); }}

/* Sidebar: frosted glass */
[data-testid="stSidebar"] > div:first-child {{
  background: #070A11; border-right: 1px solid #161D2B;
}}
[data-testid="stSidebar"] {{ background: transparent; }}
.side-title {{ font-family: 'Montserrat', sans-serif; letter-spacing: .22em; font-size: 11px; font-weight: 600;
  text-transform: uppercase; color: var(--cyan); margin: 4px 0 10px; display: flex; align-items: center; gap: 8px; }}
.side-title::before {{ content: ""; width: 6px; height: 6px; border-radius: 999px; background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan); }}
.star-desc {{ font-size: 14.5px; color: var(--muted); line-height: 1.55; margin: 2px 0 10px; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background: rgba(2,4,10,0.6); border: 1px solid rgba(255,255,255,0.1); border-radius: 9999px;
}}

/* Analyze button: primary sapphire glow pill */
[data-testid="stSidebar"] .stButton button {{
  background: var(--sapphire); color: #FFFFFF; border: 0; border-radius: 9999px; padding: .6rem 1rem;
  box-shadow: 0 0 20px rgba(37,99,235,0.6); transition: box-shadow .25s ease, transform .25s ease;
}}
[data-testid="stSidebar"] .stButton button p {{ font-family: 'Montserrat', sans-serif; font-size: 13px;
  font-weight: 500; letter-spacing: .18em; text-transform: uppercase; color: #FFFFFF; }}
[data-testid="stSidebar"] .stButton button:hover {{
  box-shadow: 0 0 24px -2px rgba(37,99,235,0.8), 0 0 48px -12px rgba(37,99,235,0.5);
  transform: translateY(-1px); color: #FFFFFF;
}}

/* Tabs: glass pills, the active one filled sapphire */
.stTabs [data-baseweb="tab-list"] {{ gap: 10px; border-bottom: 0; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{
  background: var(--glass); border: 1px solid var(--hair); border-radius: 9999px; padding: 8px 18px;
  height: auto; transition: border-color .2s ease, background .2s ease;
}}
.stTabs [data-baseweb="tab"] p {{ font-family: 'Montserrat', sans-serif; font-weight: 500; letter-spacing: .14em;
  font-size: 12.5px; text-transform: uppercase; }}
.stTabs [data-baseweb="tab"]:hover {{ border-color: rgba(59,130,246,0.45); }}
.stTabs [aria-selected="true"] {{ background: var(--sapphire); border-color: var(--sapphire);
  box-shadow: 0 0 20px rgba(37,99,235,0.5); }}
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {{ color: #FFFFFF; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ animation: fade-in .45s ease both; padding-top: 14px; }}
@keyframes fade-in {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}

/* Glass cards: every st.container(key="glass_...") */
[class*="st-key-glass"] {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 28px; padding: 22px 24px;
  box-shadow: 0 4px 25px rgba(0,0,0,0.35), inset 0 1px 1px rgba(255,255,255,0.04);
}}
[class*="st-key-glass-cls"] {{ border-color: color-mix(in srgb, var(--cls) 40%, transparent); }}

/* Telemetry metric pods */
[data-testid="stMetric"] {{
  background: var(--inset); border: 1px solid var(--border); border-radius: 20px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ font-family: 'Montserrat', sans-serif; text-transform: uppercase;
  letter-spacing: .18em; font-size: 11.5px !important; font-weight: 500; color: var(--muted); }}
[data-testid="stMetricValue"] {{ font-family: 'Inter', sans-serif !important; font-weight: 400;
  font-feature-settings: 'tnum' 1; font-size: 1.7rem !important; color: var(--text-hi); }}

/* Tables: hairline telemetry rows */
.stMarkdown table, [data-testid="stTable"] table {{ border-collapse: collapse; width: 100%; font-size: 15px;
  font-feature-settings: 'tnum' 1; }}
.stMarkdown th, [data-testid="stTable"] th {{
  font-family: 'Montserrat', sans-serif; letter-spacing: .16em; font-size: 11.5px; font-weight: 500;
  text-transform: uppercase; color: var(--muted); background: rgba(2,4,10,0.5);
}}
.stMarkdown td, .stMarkdown th, [data-testid="stTable"] td, [data-testid="stTable"] th {{
  border: 1px solid rgba(255,255,255,0.06) !important; padding: 7px 10px;
}}
.stMarkdown td code {{ font-family: 'JetBrains Mono', monospace; color: var(--text-hi);
  background: rgba(59,130,246,0.12); }}

/* Alerts (info / warning / error) as glass */
[data-testid="stAlert"] > div {{ border-radius: 20px; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); }}

/* Progress bar */
[data-testid="stProgress"] div[role="progressbar"] > div > div > div {{
  background: linear-gradient(90deg, var(--sapphire), var(--cyan)) !important;
}}

/* Hero */
.hero {{ display: flex; align-items: center; gap: 22px; margin: 18px 0 6px; flex-wrap: wrap; }}
.hero-kicker {{ display: inline-flex; align-items: center; gap: 8px; padding: 5px 14px; border-radius: 9999px;
  border: 1px solid var(--hair); background: var(--glass); font-family: 'Montserrat', sans-serif; font-size: 11px;
  font-weight: 500; letter-spacing: .22em; text-transform: uppercase; color: #9AA8BA; margin-top: 8px; }}
.hero-kicker::before {{ content: ""; width: 6px; height: 6px; border-radius: 999px; background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan); }}
.hero-title {{
  font-family: 'Montserrat', sans-serif; font-weight: 200; letter-spacing: .2em; text-transform: uppercase;
  font-size: clamp(1.7rem, 3.2vw, 3.2rem); line-height: 1.12; margin: 0; color: var(--text-hi);
}}
.hero-sub {{ font-size: 17px; font-weight: 300; color: #9AA8BA; margin: 10px 0 12px; }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }}
.chip {{ font-family: 'Montserrat', sans-serif; font-size: 11px; font-weight: 500; letter-spacing: .16em;
  text-transform: uppercase; color: #C3C6D7; border: 1px solid var(--hair); border-radius: 9999px;
  padding: 5px 14px; background: var(--glass); }}
.transit svg {{ display: block; }}
.transit .star {{ animation: dim 7s ease-in-out infinite; transform-origin: 50px 50px; }}
.transit .planet {{ animation: cross 7s linear infinite; }}
@keyframes cross {{ 0% {{ transform: translateX(-62px); }} 100% {{ transform: translateX(62px); }} }}
@keyframes dim {{ 0%, 30%, 70%, 100% {{ opacity: 1; }} 42%, 58% {{ opacity: .72; }} }}

/* Verdict card */
.verdict {{
  display: flex; align-items: center; gap: 18px; padding: 18px 22px; border-radius: 28px;
  border: 1px solid color-mix(in srgb, var(--vc) 45%, transparent); background: var(--glass-hi);
  box-shadow: 0 0 24px -2px color-mix(in srgb, var(--vc) 35%, transparent);
  animation: pulse 1.6s ease-out 1;
}}
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 color-mix(in srgb, var(--vc) 55%, transparent); }}
  100% {{ box-shadow: 0 0 24px -2px color-mix(in srgb, var(--vc) 35%, transparent); }}
}}
.v-icon {{ width: 58px; height: 58px; flex: none; border-radius: 999px; display: grid; place-items: center;
  color: var(--vc); background: color-mix(in srgb, var(--vc) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--vc) 40%, transparent); box-shadow: 0 0 18px color-mix(in srgb, var(--vc) 30%, transparent); }}
.v-icon .msi {{ font-size: 30px !important; }}
.v-main {{ flex: 1; min-width: 0; }}
.v-kicker {{ font-family: 'Montserrat', sans-serif; font-size: 11px; font-weight: 500; letter-spacing: .22em;
  text-transform: uppercase; color: var(--muted); }}
.v-class {{ font-family: 'Montserrat', sans-serif; font-weight: 300; font-size: 1.8rem; letter-spacing: .16em;
  color: var(--vc); }}
.v-conf {{ font-size: 15px; color: var(--text); }}
.v-ring {{ --p: 50; width: 84px; height: 84px; border-radius: 50%; flex: none;
  background: conic-gradient(var(--vc) calc(var(--p) * 1%), rgba(255,255,255,0.07) 0);
  display: grid; place-items: center; box-shadow: 0 0 16px color-mix(in srgb, var(--vc) 35%, transparent); }}
.v-ring span {{ width: 66px; height: 66px; border-radius: 50%; background: #06102A; display: grid;
  place-items: center; font-family: 'Inter', sans-serif; font-feature-settings: 'tnum' 1; font-size: 17px; color: var(--text-hi); }}
.v-summary {{ font-size: 17px !important; font-weight: 300; line-height: 1.6; color: var(--text-hi); margin: 14px 2px 4px; }}
.catalog {{ font-size: 15.5px; padding: 11px 16px; border-radius: 9999px; margin-top: 8px;
  background: rgba(2,4,10,0.55); border: 1px solid var(--hair); }}
.hint {{ font-size: 15.5px; padding: 11px 16px; border-radius: 20px; margin: 6px 0;
  background: rgba(167,139,250,0.10); border: 1px solid rgba(167,139,250,0.40); }}

@media (prefers-reduced-motion: reduce) {{
  .stApp::before, .stApp::after, .transit .star, .transit .planet, .verdict, .star-card, .star-card img,
  .hero-title, .hero-sub, .chips,
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
      <stop offset="0%" stop-color="#FFFFFF"/><stop offset="40%" stop-color="#FFDDB8"/>
      <stop offset="68%" stop-color="#FFB95F" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#2563EB" stop-opacity="0"/>
    </radialGradient>
    <clipPath id="sky"><rect width="100" height="100"/></clipPath>
  </defs>
  <g class="star"><circle cx="50" cy="50" r="46" fill="url(#glow)"/>
    <circle cx="50" cy="50" r="22" fill="#FFF3DD"/></g>
  <g clip-path="url(#sky)"><circle class="planet" cx="50" cy="52" r="7" fill="#02040A"
    stroke="#22D3EE" stroke-width="1" stroke-opacity="0.8"/></g>
</svg></div>
"""


def hero_html(title: str, subtitle: str, chips: list[str]) -> str:
    chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    head, _, tail = title.partition(" & ")
    title_html = (f'{html.escape(head)} &amp;<br><span>{html.escape(tail)}</span>' if tail
                  else html.escape(title))
    return (f'<div class="hero-kicker">NASA TESS photometry · False positive recycler</div>'
            f'<div class="hero">{TRANSIT_SVG}</div><h1 class="hero-title">{title_html}</h1>'
            f'<div class="hero-sub">{html.escape(subtitle)}</div>'
            f'<div class="chips">{chips_html}</div>')


NAV_PAGES = [("home", "Home"), ("learn", "Learn"), ("analyzer", "Analyzer"),
             ("demo_stars", "Demo Stars"), ("how_it_works", "How It Works")]


def nav_html(current: str) -> str:
    """The Stitch navigation capsule. Links reload the app on that page (?page=...)."""
    links = "".join(f'<a href="?page={k}" target="_self" class="{"on" if k == current else ""}">'
                    f'{name}</a>' for k, name in NAV_PAGES)
    return (f'<div class="ea-nav"><nav>{links}</nav>'
            f'<div class="ea-brand">Exoplanet <b>Atlas</b></div>'
            f'<a class="ea-cta" href="?page=demo_stars" target="_self">Demo stars'
            f'<i>{icon_html("north_east", size=16)}</i></a></div>')


def star_card_html(name: str, tic: int, cls: str, kind: str, facts: dict,
                   img_src: str, img_alt: str) -> str:
    """Picture card for the selected star (same images as the Demo Stars cards)."""
    return (f'<div class="star-card"><img src="{html.escape(img_src)}" alt="{html.escape(img_alt)}">'
            f'<div class="shade"></div><div class="top"><span class="tag">TIC {tic}</span>'
            f'<span class="badge" style="--bc:{color(cls)}">{html.escape(label(cls))}</span></div>'
            f'<div class="foot"><div><div class="k">Selected star</div>'
            f'<div class="n">{html.escape(name)}</div><div class="t">{html.escape(kind)}</div></div>'
            f'<div class="r">P = <b>{html.escape(facts.get("period", "—"))}</b><br>'
            f'Depth {html.escape(facts.get("depth", "—"))}<br>'
            f'{html.escape(facts.get("sectors", "—"))} sectors</div></div></div>'
            f'<div class="star-credit">Image: {html.escape(img_alt)}</div>')


def verdict_html(cls: str, confidence: str, prob: float | None, summary: str) -> str:
    pct = "" if prob is None else f"{100 * prob:.0f}%"
    ring = (f'<div class="v-ring" style="--p:{0 if prob is None else 100 * prob:.0f}">'
            f'<span>{pct or "–"}</span></div>')
    return (f'<div class="verdict" style="--vc:{color(cls)}">'
            f'<div class="v-icon">{icon_html(icon(cls))}</div>'
            f'<div class="v-main"><div class="v-kicker">Verdict</div>'
            f'<div class="v-class">{html.escape(label(cls).upper())}</div>'
            f'<div class="v-conf">Confidence: <b>{html.escape(confidence or "unknown")}</b>'
            f'{" · class probability " + pct if pct else ""}</div></div>{ring}</div>'
            f'<p class="v-summary">{html.escape(summary or "")}</p>')


def class_color_css(cls: str) -> str:
    """Sets the colour used by the class-tinted glass cards."""
    return f"<style>:root {{ --cls: {color(cls)}; }}</style>"
