"""Capture the README screenshots of the app (dev tool; needs Playwright).

    pip install playwright && python -m playwright install chromium
    python scripts/take_screenshots.py [--port 8599]

Starts the app headless, waits until it is healthy, opens it in Chromium at
1600x1000 and saves docs/screenshots/: the four pages (home, learn, demo_stars,
how_it_works) and the Analyzer (analyzer, lightcurves, bls, phasefold, results)
from the precomputed examples (no network needed for the analysis). The server
and the browser are always shut down at the end.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
VIEWPORT = {"width": 1600, "height": 1000}
HIDE_CHROME = """
[data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], .stDeployButton { display: none !important; }
"""


def wait_healthy(url: str, timeout_s: float = 90) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=2) as r:
                if r.read().strip() == b"ok":
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"app at {url} did not become healthy in {timeout_s:.0f} s")


def settle(page, charts: int = 1) -> None:
    """Network idle, at least ``charts`` rendered Plotly charts in the open tab, fonts, +2 s."""
    page.wait_for_load_state("networkidle")
    page.wait_for_function(
        """n => {
            const panel = [...document.querySelectorAll('[role="tabpanel"]')]
                .find(p => p.offsetParent !== null && !p.hidden);
            const root = panel || document;
            return root.querySelectorAll('.js-plotly-plot .main-svg').length >= n;
        }""", arg=charts, timeout=60_000)
    page.evaluate("document.fonts.ready")
    page.wait_for_timeout(2000)


def analyze(page, star: str) -> None:
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').click()
    page.locator('li[role="option"]', has_text=star).first.click()
    page.locator('[data-testid="stSidebar"] button', has_text="Analyze").click()
    page.locator('button[role="tab"]', has_text="Light Curves").wait_for(timeout=60_000)


def stitch_page(page, url: str, name: str, height: int) -> None:
    """A Stitch page (Home / Learn / Demo Stars / How It Works) in its full-width frame."""
    page.set_viewport_size({"width": VIEWPORT["width"], "height": height})
    page.goto(f"{url}/?page={name}")
    page.wait_for_load_state("networkidle")
    page.locator("iframe").first.wait_for(timeout=60_000)
    frame = page.locator("iframe").first.content_frame
    frame.locator("main").wait_for(timeout=60_000)
    # Show every section at once (the scroll-reveal animation would hide lower ones).
    frame.locator("body").evaluate(
        "b => b.querySelectorAll('.ea-reveal').forEach(e => e.classList.add('ea-in'))")
    page.wait_for_timeout(3500)
    page.screenshot(path=OUT / f"{name}.png")
    page.set_viewport_size(VIEWPORT)


def open_tab(page, name: str, charts: int = 1) -> None:
    page.locator('button[role="tab"]', has_text=name).click()
    settle(page, charts)


def main() -> None:
    from playwright.sync_api import sync_playwright

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8599)
    args = ap.parse_args()
    url = f"http://localhost:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(args.port)], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_healthy(url)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
                stitch_page(page, url, "home", 1000)
                stitch_page(page, url, "learn", 1000)
                stitch_page(page, url, "demo_stars", 1500)
                stitch_page(page, url, "how_it_works", 1600)

                page.goto(f"{url}/?page=analyzer")
                page.wait_for_load_state("networkidle")
                page.add_style_tag(content=HIDE_CHROME)
                page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').wait_for()

                analyze(page, "HD 209458 b")
                page.wait_for_timeout(2500)
                page.screenshot(path=OUT / "analyzer.png")
                open_tab(page, "Light Curves", charts=2)
                page.screenshot(path=OUT / "lightcurves.png")
                open_tab(page, "BLS Detection", charts=1)
                page.screenshot(path=OUT / "bls.png")
                open_tab(page, "Results", charts=3)
                # Streamlit scrolls inside its own container, so make the window tall
                # enough for the verdict card and the confusion matrix together.
                page.set_viewport_size({"width": VIEWPORT["width"], "height": 2100})
                settle(page, charts=3)
                page.screenshot(path=OUT / "results.png")
                page.set_viewport_size(VIEWPORT)

                analyze(page, "TOI 184.01")
                open_tab(page, "Phase-Folded", charts=3)
                page.set_viewport_size({"width": VIEWPORT["width"], "height": 1560})
                settle(page, charts=3)
                page.screenshot(path=OUT / "phasefold.png")
            finally:
                browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    for name in ("home", "learn", "demo_stars", "how_it_works", "analyzer", "lightcurves",
                 "bls", "results", "phasefold"):
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
