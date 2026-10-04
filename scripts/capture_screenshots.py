"""Capture the README screenshots and the animated tour of the dashboard.

Needs a running dashboard (`sp500-web`), Google Chrome, and `pip install playwright` (it drives the
installed Chrome, so no browser download). The agent screenshot asks one real question, so an LLM key
must be configured on the server for it to show a real answer.

    python scripts/capture_screenshots.py --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "media"
SHOTS = [
    ("overview", "/", "light"),
    ("stock", "/stock/NVDA", "light"),
    ("backtest", "/backtest", "light"),
    ("model-lab", "/model", "dark"),
    ("screener", "/screener", "dark"),
]
AGENT_QUESTION = "Compare NVDA and AMD on growth, valuation and recent filings"
SHOT_WIDTH = 1600
GIF_WIDTH = 960


def open_page(browser, theme: str) -> tuple:
    context = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme=theme)
    context.add_init_script(f"try {{ localStorage.setItem('theme', '{theme}'); sessionStorage.clear(); }} catch (e) {{}}")
    return context, context.new_page()


def settle(page: Page, extra_ms: int = 1500) -> None:
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(extra_ms)


def grab(page: Page) -> Image.Image:
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


def save(image: Image.Image, name: str) -> None:
    if image.width > SHOT_WIDTH:
        image = image.resize((SHOT_WIDTH, round(image.height * SHOT_WIDTH / image.width)), Image.LANCZOS)
    path = OUT / f"{name}.png"
    image.save(path, optimize=True)
    print(f"saved {path.relative_to(OUT.parents[1])} ({path.stat().st_size // 1024} KB)")


def gif_frame(image: Image.Image) -> Image.Image:
    small = image.resize((GIF_WIDTH, round(image.height * GIF_WIDTH / image.width)), Image.LANCZOS)
    return small.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--no-agent", action="store_true", help="Skip the live agent question.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tour: list[tuple[Image.Image, int]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for name, path, theme in SHOTS:
            context, page = open_page(browser, theme)
            page.goto(args.url + path)
            settle(page)
            image = grab(page)
            save(image, name)
            tour.append((image, 2200))
            context.close()

        if not args.no_agent:
            context, page = open_page(browser, "dark")
            page.goto(args.url + "/agent")
            settle(page, 500)
            page.fill("#agent-input", AGENT_QUESTION)
            page.keyboard.press("Enter")
            deadline = time.time() + 120
            done = page.locator("text=/tokens|Rule-based assistant/")
            while time.time() < deadline and done.count() == 0:
                page.wait_for_timeout(1500)
                tour.append((grab(page), 700))
            page.wait_for_timeout(1000)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(500)
            image = grab(page)
            save(image, "agent")
            tour.append((image, 3500))
            context.close()
        browser.close()

    # Tour order: overview, stock, agent at work, backtest, model lab, screener.
    order = tour[:2] + tour[len(SHOTS):] + tour[2 : len(SHOTS)]
    frames = [gif_frame(image) for image, _ in order]
    durations = [ms for _, ms in order]
    path = OUT / "tour.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)
    print(f"saved {path.relative_to(OUT.parents[1])} ({path.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
