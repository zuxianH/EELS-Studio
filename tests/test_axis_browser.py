"""Optional mouse-gesture coverage; set EELS_BROWSER=firefox or use Chrome."""
import math
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


@pytest.fixture
def spectrum_page(tmp_path):
    chrome = os.environ.get("EELS_CHROME_PATH") or shutil.which("google-chrome") or shutil.which("chromium")
    engine = os.environ.get("EELS_BROWSER", "chromium")
    if engine == "chromium" and not chrome:
        pytest.skip("Chrome/Chromium is not installed")
    energy, rows, columns = np.indices((41, 12, 14))
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
        else:
            pytest.fail("Streamlit server did not become healthy")
        with playwright.sync_playwright() as driver:
            browser = getattr(driver, engine).launch(
                **({"executable_path": chrome} if engine == "chromium" else {}), headless=True)
            page = browser.new_page(viewport={"width": 1700, "height": 1300})
            page.set_default_timeout(30000)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}")
            folder = page.get_by_role("textbox", name="Data folder", exact=True)
            folder.fill(str(tmp_path))
            folder.press("Enter")
            page.locator(".st-key-spectrum_axis_target .js-plotly-plot").wait_for(state="visible")
            yield page
            assert not page.locator('[data-testid="stException"]').count()
            assert not errors
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=10)


def test_shift_drag_scales_only_chosen_axis_and_preserves_native_controls(spectrum_page):
    page = spectrum_page
    chart = page.locator(".st-key-spectrum_axis_target .js-plotly-plot")

    def ranges():
        return chart.evaluate("c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]")

    def gesture(axis, distance, shift=True):
        handle = chart.locator(".ewdrag" if axis == 0 else ".nsdrag")
        handle.scroll_into_view_if_needed()
        box = handle.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        if shift:
            page.keyboard.down("Shift")
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + distance if axis == 0 else x,
                        y - distance if axis == 1 else y, steps=10)
        page.mouse.up()
        if shift:
            page.keyboard.up("Shift")

    def wait_for_span(axis, expected):
        try:
            page.wait_for_function("""([axis, expected]) => {
                const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
                const r = c._fullLayout[axis === 0 ? 'xaxis' : 'yaxis'].range;
                return Math.abs((r[1] - r[0]) / expected - 1) < 1e-6;
            }""", arg=[axis, expected])
        except playwright.TimeoutError:
            pytest.fail(f"Axis {axis}: expected span {expected}; actual ranges {ranges()}")

    initial = ranges()
    for axis in (1, 0):
        before = ranges()
        length = chart.evaluate("(c, a) => c._fullLayout[a]._length", "xaxis" if axis == 0 else "yaxis")
        span = before[axis][1] - before[axis][0]
        gesture(axis, 60)
        wait_for_span(axis, span * math.exp(-60 / length * math.log(4)))
        after = ranges()
        assert sum(after[axis]) == pytest.approx(sum(before[axis]))
        assert after[1 - axis] == pytest.approx(before[1 - axis])
        # Plotly can resize the axis when a new set of tick labels changes margins.
        length = chart.evaluate("(c, a) => c._fullLayout[a]._length", "xaxis" if axis == 0 else "yaxis")
        zoomed_span = after[axis][1] - after[axis][0]
        gesture(axis, -60)
        wait_for_span(axis, zoomed_span * math.exp(60 / length * math.log(4)))
        assert sum(ranges()[axis]) == pytest.approx(sum(before[axis]))
        assert ranges()[1 - axis] == pytest.approx(before[1 - axis])

    # Unmodified drag still pans without scaling.
    before = ranges()
    gesture(1, 40, shift=False)
    wait_for_span(1, before[1][1] - before[1][0])
    assert ranges()[1][0] != pytest.approx(before[1][0])
    assert ranges()[0] == pytest.approx(before[0])

    # Plotly's native reset control still restores the original ranges.
    chart.locator('[data-title="Reset axes"]').click(force=True)
    page.wait_for_function("""expected => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return ['xaxis', 'yaxis'].every((axis, i) =>
            c._fullLayout[axis].range.every((v, j) => Math.abs(v - expected[i][j]) < 1e-6));
    }""", arg=initial, timeout=5000)

    # A Streamlit rerun must not remove or duplicate the gesture handler.
    page.get_by_text("Show hover details", exact=True).click()
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return c?._fullLayout?.hovermode === false;
    }""")
    before = ranges()
    length = chart.evaluate("c => c._fullLayout.yaxis._length")
    gesture(1, 60)
    wait_for_span(1, (before[1][1] - before[1][0]) * math.exp(-60 / length * math.log(4)))
    assert ranges()[0] == pytest.approx(before[0])


def test_zoom_survives_detector_changes_and_explicit_limits_still_apply(spectrum_page):
    page = spectrum_page
    chart = page.locator(".st-key-spectrum_axis_target .js-plotly-plot")

    def ranges():
        return chart.evaluate("c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]")

    def wait_for_view(expected):
        page.wait_for_function("""expected => {
            const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
            return ['xaxis', 'yaxis'].every((axis, i) =>
                c?._fullLayout?.[axis]?.range.every((v, j) => Math.abs(v - expected[i][j]) < 1e-8));
        }""", arg=expected, timeout=5000)

    def change_radius(value):
        first_y = chart.evaluate("c => c._fullData[0].y[0]")
        radius = page.get_by_role("spinbutton", name="radius", exact=True)
        radius.fill(str(value))
        radius.press("Enter")
        page.wait_for_function("""oldValue => {
            const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
            const value = c?._fullData?.[0]?.y?.[0];
            return Number.isFinite(value) && value !== oldValue;
        }""", arg=first_y)

    # Use the real box-zoom gesture, not a programmatic range update.
    plot = chart.locator(".nsewdrag")
    plot.scroll_into_view_if_needed()
    box = plot.bounding_box()
    initial = ranges()
    page.mouse.move(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.25)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.65,
                    box["y"] + box["height"] * 0.65, steps=10)
    page.mouse.up()
    page.wait_for_function("""initial => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return ['xaxis', 'yaxis'].every((axis, i) => {
            const r = c._fullLayout[axis].range;
            return r[1] - r[0] < 0.8 * (initial[i][1] - initial[i][0]);
        });
    }""", arg=initial)
    zoomed = ranges()
    change_radius(5)
    wait_for_view(zoomed)
    np.testing.assert_allclose(ranges(), zoomed, rtol=1e-8)

    # The custom Shift-drag gesture must persist too.
    handle = chart.locator(".nsdrag")
    handle.scroll_into_view_if_needed()
    box = handle.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.keyboard.down("Shift")
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y - 60, steps=10)
    page.mouse.up()
    page.keyboard.up("Shift")
    page.wait_for_function("""oldSpan => {
        const r = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.yaxis.range;
        return r[1] - r[0] < 0.9 * oldSpan;
    }""", arg=zoomed[1][1] - zoomed[1][0])
    scaled = ranges()
    change_radius(4)
    wait_for_view(scaled)
    np.testing.assert_allclose(ranges(), scaled, rtol=1e-8)

    # Explicit x-axis limits override only that axis; the y-axis exploration persists.
    minimum = page.get_by_role("spinbutton", name="Energy min (meV)", exact=True)
    minimum.fill("-100")
    minimum.press("Enter")
    page.wait_for_function("""() => {
        const r = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.xaxis.range;
        return r[0] === -100 && r[1] === 150;
    }""")
    np.testing.assert_allclose(ranges()[1], scaled[1], rtol=1e-8)


def test_fast_scroll_zooms_without_scrolling_the_page(spectrum_page):
    page = spectrum_page
    chart = page.locator(".st-key-spectrum_axis_target .js-plotly-plot")

    def ranges():
        return chart.evaluate("c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]")

    chart.scroll_into_view_if_needed()
    box = chart.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before = ranges()
    before_scroll = page.evaluate("() => window.scrollY")
    # A burst of large-delta wheel events emulates a fast trackpad flick, which is
    # exactly the case where Plotly's own scrollZoom handler can miss a preventDefault.
    for _ in range(20):
        page.mouse.wheel(0, -300)
    page.wait_for_function("""before => {
        const r = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.xaxis.range;
        return r[1] - r[0] < (before[1] - before[0]) * 0.5;
    }""", arg=before[0])
    assert page.evaluate("() => window.scrollY") == before_scroll
