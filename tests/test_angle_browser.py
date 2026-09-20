"""Optional real-browser coverage: requires Playwright and Firefox or a local Chrome binary."""
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]


def test_draw_and_resize_rectangle_in_browser(tmp_path):
    chrome = os.environ.get("EELS_CHROME_PATH") or shutil.which("google-chrome") or shutil.which("chromium")
    engine = os.environ.get("EELS_BROWSER", "chromium")
    if engine == "chromium" and not chrome:
        pytest.skip("Chrome/Chromium is not installed")
    energy, rows, columns = np.indices((41, 60, 80))
    np.save(tmp_path / "scan_T300K.npy", 1.0 + energy + rows / 10 + columns / 100)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
        "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=0.5):
                    break
            except OSError:
                if server.poll() is not None:
                    pytest.fail("Streamlit server exited")
                time.sleep(0.1)
        with playwright.sync_playwright() as driver:
            browser = getattr(driver, engine).launch(
                **({"executable_path": chrome} if engine == "chromium" else {}), headless=True)
            page = browser.new_page(viewport={"width": 1700, "height": 1300})
            page.set_default_timeout(30000)
            page.goto(f"http://127.0.0.1:{port}")
            folder = page.get_by_role("textbox", name="Data folder", exact=True)
            folder.fill(str(tmp_path))
            folder.press("Enter")
            page.get_by_role("tab", name="Angle-resolved EELS", exact=True).click()
            # Enabled by default now; no click needed.
            playwright.expect(page.get_by_role("checkbox", name="Enable angle-resolved map", exact=True)).to_be_checked()
            chart = page.locator(".st-key-angle_rectangle_target .js-plotly-plot")
            chart.wait_for(state="visible")
            chart.scroll_into_view_if_needed()
            # Wait for the component's listener, not just the visible Plotly image.
            page.wait_for_function("""() => {
                const chart = document.querySelector('.st-key-angle_rectangle_target .js-plotly-plot');
                return chart?._ev?.listenerCount('plotly_relayout') > 0;
            }""")
            def screen_point(x, y):
                return chart.evaluate("""(chart, point) => {
                    const rect = chart.getBoundingClientRect();
                    const layout = chart._fullLayout;
                    return {x: rect.x + layout.xaxis._offset + layout.xaxis.l2p(point[0]),
                            y: rect.y + layout.yaxis._offset + layout.yaxis.l2p(point[1])};
                }""", [x, y])
            start = screen_point(19.5, 11.5)
            end = screen_point(50.5, 30.5)
            page.mouse.move(start["x"], start["y"])
            page.mouse.down()
            page.mouse.move(end["x"], end["y"], steps=15)
            page.mouse.up()
            for label, expected in [("Row min (px)", "12"), ("Row max (px)", "30"),
                                     ("Column min (py)", "20"), ("Column max (py)", "50")]:
                playwright.expect(page.get_by_role("spinbutton", name=label, exact=True)).to_have_value(expected)
            playwright.expect(page.get_by_text("Map shape: 41 energy bins × 31 pixels.", exact=False)).to_be_visible()
            # Exercise Plotly's incremental shape-edit payload as emitted by resize/move.
            chart.evaluate("chart => window.Plotly.relayout(chart, {'shapes[0].x1': 40.5})")
            playwright.expect(page.get_by_role("spinbutton", name="Column max (py)", exact=True)).to_have_value("40")
            playwright.expect(page.get_by_text("Map shape: 41 energy bins × 21 pixels.", exact=False)).to_be_visible()
            assert not page.locator('[data-testid="stException"]').count()
            assert not page.locator('[data-testid="stAlertContainer"]').filter(has_text="Could not").count()
            # Save an optional review image without adding generated binaries to the repo.
            screenshot = os.environ.get("EELS_BROWSER_SCREENSHOT")
            if screenshot:
                page.screenshot(path=screenshot, full_page=True)
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)
