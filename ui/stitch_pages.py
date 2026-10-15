"""The Stitch-designed pages (Home, Learn, Demo Stars, How It Works).

The HTML in ui/stitch/ is the Stitch export ("Exoplanet Atlas Web App"), with its
images saved under static/stitch/. It is shown as-is in a full-width frame; this
module only
  * points the Stitch navigation links at the app's pages (?page=...),
  * fills the Demo Stars cards with the six real demo stars from the repo,
  * lets the frame grow to the page's height.
The Analyzer page is not part of this: it is the original app (app.py).
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

from ui.views import eb_period, final_period

HTML_DIR = Path(__file__).parent / "stitch"
REPO_URL = "https://github.com/mahi-mahalle/AIML_Exoplanet"
PAGES = {"home": "Home", "learn": "Learn", "analyzer": "Analyzer",
         "demo_stars": "Demo Stars", "how_it_works": "How It Works"}
STITCH_PATHS = {"home": "home", "learn": "learn", "analyzer": "analyzer",
                "demo-stars": "demo_stars", "how-it-works": "how_it_works"}

# Streamlit chrome around the Stitch pages: no header, sidebar or padding.
PAGE_CSS = """
<style>
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stToolbar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stDecoration"] { display: none !important; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #030407 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stMainBlockContainer"] { padding: 0 !important; }
iframe { display: block; border: 0; }
</style>
"""

# Injected into each Stitch page: links switch app pages, the frame fits the content.
FRAME_SCRIPT = """
<style>body{min-height:0!important}body>div{min-height:0!important}</style>
<script>
(function () {
  function go(href) {                      // navigate the app (the top page), not the frame
    var top = window.parent.document, a = top.createElement("a");
    a.href = href; a.style.display = "none"; top.body.appendChild(a); a.click();
  }
  document.addEventListener("click", function (e) {
    var a = e.target.closest("a[data-app-href]");
    if (a) { e.preventDefault(); go(a.getAttribute("data-app-href")); }
  });
  function fit() {
    var f = window.frameElement;
    if (f) f.style.height = Math.ceil(document.documentElement.scrollHeight) + "px";
  }
  window.addEventListener("load", fit);
  window.addEventListener("resize", fit);
  if (window.ResizeObserver) new ResizeObserver(fit).observe(document.body);
  setTimeout(fit, 300); setTimeout(fit, 1500);
})();
</script>
"""

# Demo Stars: which Stitch card design each class uses (index of the card in the
# Stitch export), its badge, the filter it belongs to and its telemetry labels.
CARD_SKIN = {"PLANET": 0, "EB": 2, "VARIABLE": 3}
REVIEW_SKINS = [1, 4]                       # planet-like candidate, noisy candidate
EB_EXTRA_SKIN = 5
BADGE = {"PLANET": ("Confirmed Planet", "84,216,232", "#54d8e8", "confirmed"),
         "EB": ("Eclipsing Binary", "245,158,11", "#f59e0b", "binary"),
         "VARIABLE": ("Variable Star", "236,72,153", "#ec4899", "variable"),
         "NOISE": ("Noise", "148,163,184", "#94a3b8", "review"),
         "NEEDS_REVIEW": ("Needs Review", "167,139,250", "#a78bfa", "review")}
ROW_LABELS = {"PLANET": ("Orbital Period (P)", "Transit Depth"),
              "EB": ("Orbital Period (P)", "Primary Depth"),
              "VARIABLE": ("Pulsation Period", "Amplitude"),
              "NEEDS_REVIEW": ("Candidate Period", "Dip Depth")}
# NASA images for cards whose Stitch design has none (credits in static/stitch/CREDITS.md).
CARD_IMAGE = {
    "VARIABLE": ("app/static/stitch/nasa_v838_monocerotis.jpg",
                 "V838 Monocerotis light echo, Hubble photo (NASA, ESA, H. E. Bond)"),
    "NEEDS_REVIEW": ("app/static/stitch/nasa_pia26601_young_exoplanet.jpg",
                     "Young exoplanet, artist's concept (NASA/JPL-Caltech)"),
}
IMAGE_BLOCK = (
    '<div class="w-full h-32 rounded-DEFAULT mb-4 overflow-hidden relative group/img border '
    'border-[#202a38]"><img src="{src}" alt="{alt}" title="{alt}" class="w-full h-full '
    'object-cover object-center opacity-85 transition-all duration-500 group-hover/img:scale-105 '
    'group-hover/img:opacity-100"><div class="absolute inset-0 bg-gradient-to-t from-[#080d17] '
    'via-transparent to-transparent"></div><span class="absolute bottom-2 left-2.5 '
    'font-label-caps text-[10px] tracking-widest uppercase text-secondary/90 bg-[#080d17]/80 '
    'px-2 py-0.5 rounded-full border border-secondary/20"></span></div>'
)
LABEL_SPAN = ('<span class="text-on-surface-variant uppercase font-label-caps text-label-caps '
              'tracking-widest">')


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _fmt(v: float, spec: str, unit: str) -> str:
    return f"{v:{spec}}{unit}" if math.isfinite(v) else "—"


def star_facts(report: dict) -> dict:
    """Period, depth and SNR as the Analyzer's Findings section computes them."""
    features = report.get("features") or {}
    signal = (report.get("signals") or [{}])[0]
    modules = report.get("modules") or {}
    period, kind = final_period(report)
    if kind == "eb" and math.isfinite(eb_period(report)):
        period = eb_period(report)
    if kind == "variable":
        depth = _num(features.get("amp_p2p_ppm")) / 1e4
    elif kind == "eb":
        ecl = ((modules.get("eb") or {}).get("details") or {}).get("eclipses") or {}
        depth = _num(ecl.get("primary_depth")) * 100
    else:
        depth = _num(signal.get("depth_ppm", features.get("depth_ppm"))) / 1e4
    snr = _num(features.get("depth_snr", signal.get("depth_snr")))
    return {"period": _fmt(period, ".5f", " d"), "depth": _fmt(depth, ".2f", " %"),
            "snr": _fmt(snr, ".1f", ""), "sectors": str(len(report.get("sectors") or []))}


def _card(skin: str, ex: dict, report: dict) -> str:
    """One Stitch card re-filled with a real demo star."""
    cls = ex.get("verdict", "NEEDS_REVIEW")
    name, rgb, hexc, category = BADGE.get(cls, BADGE["NEEDS_REVIEW"])
    facts = star_facts(report)
    c = re.sub(r'data-category="[^"]*"', f'data-category="{category}"', skin, count=1)
    c = re.sub(r"(<h2[^>]*>)[^<]*(</h2>)", rf"\g<1>{html.escape(ex['display_name'])}\g<2>",
               c, count=1)
    if ex["display_name"] != "HD 209458 b":
        c = re.sub(r'\s*<span class="font-label-caps text-label-caps text-secondary[^"]*">'
                   r'\(OSIRIS\)</span>', "", c)
    c = re.sub(r">TIC \d+<", f">TIC {ex['tic_id']}<", c, count=1)
    badge = (f'<span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full shadow-sm" '
             f'style="color:{hexc};background:rgba({rgb},0.08);border:1px solid rgba({rgb},0.25);">'
             f'<span class="w-1.5 h-1.5 rounded-full" style="background:{hexc};box-shadow:0 0 6px '
             f'rgba({rgb},0.8);"></span><span class="font-label-caps text-label-caps uppercase '
             f'tracking-wider">{name}</span></span>')
    c = re.sub(r'<span class="inline-flex items-center gap-1\.5 px-3 py-1 rounded-full.*?'
               r'</span>\s*</span>', badge, c, count=1, flags=re.S)
    if "<img" not in c:                     # some Stitch card designs have no picture
        src, alt = CARD_IMAGE.get(cls, CARD_IMAGE["NEEDS_REVIEW"])
        c = re.sub(r'(<p class="font-body-sm)', IMAGE_BLOCK.format(src=src, alt=alt) + r"\g<1>",
                   c, count=1)
    c = re.sub(r'(<span class="absolute bottom-2 left-2\.5[^>]*>)[^<]*(</span>)',
               rf"\g<1>{html.escape(ex['type'])}\g<2>", c, count=1)
    c = re.sub(r'(<p class="font-body-sm[^>]*>)\s*[^<]*(</p>)',
               rf"\g<1>{html.escape(ex['hint'])}\g<2>", c, count=1)
    c = re.sub(r">SNR: [\d.]+<", f">SNR: {facts['snr']}<", c)
    labels = ROW_LABELS.get(cls, ROW_LABELS["NEEDS_REVIEW"]) + ("Sectors Used",)
    values = (facts["period"], facts["depth"], facts["sectors"])
    rows = iter(zip(labels, values))

    def row(m: re.Match) -> str:
        lab, val = next(rows, (None, None))
        if lab is None:
            return m.group(0)
        return f"{LABEL_SPAN}{lab}</span>{m.group(1)}{m.group(2)}{val}</span>"

    c = re.sub(re.escape(LABEL_SPAN) + r"[^<]*</span>(\s*)(<span[^>]*>)[^<]*</span>", row, c)
    c = c.replace('href="#"', f'href="#" data-app-href="?page=analyzer&star={ex["tic_id"]}"', 1)
    return c


def demo_stars_html(page: str, examples: list[dict], load_demo) -> str:
    """Rebuild the Demo Stars grid from examples.json and the precomputed reports."""
    parts = re.split(r'(?=<article class="star-card)', page)
    head, cards = parts[0], parts[1:]
    tail = ""
    if cards:
        last = cards[-1]
        cut = last.find("</article>") + len("</article>")
        cards[-1], tail = last[:cut], last[cut:]
    reviews, ebs = iter(REVIEW_SKINS), 0
    out = []
    for ex in examples:
        cls = ex.get("verdict", "NEEDS_REVIEW")
        if cls == "EB":
            skin = CARD_SKIN["EB"] if ebs == 0 else EB_EXTRA_SKIN
            ebs += 1
        elif cls in CARD_SKIN:
            skin = CARD_SKIN[cls]
        else:
            skin = next(reviews, REVIEW_SKINS[-1])
        out.append(_card(cards[skin], ex, load_demo(ex["tic_id"])))
    groups = [BADGE.get(e.get("verdict"), BADGE["NEEDS_REVIEW"])[3] for e in examples]
    counts = {k: groups.count(k) for k in ("confirmed", "binary", "variable", "review")}
    for label, key in (("Confirmed Planets", "confirmed"), ("Eclipsing Binaries", "binary"),
                       ("Variable Stars", "variable"), ("Needs Review", "review")):
        head = re.sub(rf"{label} \(\d+\)", f"{label} ({counts[key]})", head)
    head = re.sub(r"All Targets \(\d+\)", f"All Targets ({len(examples)})", head)
    return head + "\n".join(out) + tail


def _wire_nav(page_html: str, current: str) -> str:
    """Stitch nav links (data-path="...") open the matching app page."""
    def link(m: re.Match) -> str:
        key = STITCH_PATHS.get(m.group(1), "home")
        return f'data-path="{m.group(1)}" data-app-href="?page={key}"'
    page_html = re.sub(r'data-path="([^"]+)"', link, page_html)
    page_html = re.sub(r"Browse 100 Demo Stars", "Browse the 6 Demo Stars", page_html,
                       flags=re.I)
    page_html = re.sub(r'href="#"(\s*>\s*GitHub Open Science)',
                       rf'href="{REPO_URL}" target="_blank" rel="noopener"\g<1>', page_html)
    return page_html.replace("</body>", FRAME_SCRIPT + "</body>")


def render(page: str, examples: list[dict], load_demo) -> str:
    """The finished HTML for one Stitch page."""
    text = (HTML_DIR / f"{page}.html").read_text(encoding="utf-8")
    if page == "demo_stars":
        text = demo_stars_html(text, examples, load_demo)
    return _wire_nav(text, page)


def star_images(examples: list[dict], load_demo) -> dict[int, tuple[str, str]]:
    """{tic: (image src, alt text)} exactly as used on the Demo Stars cards."""
    page = demo_stars_html((HTML_DIR / "demo_stars.html").read_text(encoding="utf-8"),
                           examples, load_demo)
    out = {}
    for card in re.split(r'(?=<article class="star-card)', page)[1:]:
        tic = re.search(r">TIC (\d+)<", card)
        img = re.search(r'<img src="([^"]+)" alt="([^"]*)"', card)
        if tic and img:
            out[int(tic.group(1))] = (img.group(1), html.unescape(img.group(2)))
    return out
