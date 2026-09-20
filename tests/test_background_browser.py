"""Optional real-browser background layout and viewport regression checks."""
import os

import numpy as np
import pytest

from test_axis_browser import spectrum_page


def test_background_desktop_narrow_and_preserved_view(spectrum_page):
    page = spectrum_page
    chart = page.locator(".st-key-spectrum_axis_target .js-plotly-plot")
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return c?._ev?.listenerCount('plotly_relayout') > 0;
    }""")
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis.range': [20, 80], 'yaxis.range': [-4, -2]})")
    page.get_by_role("tab", name="Background", exact=True).click()
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role("option", name="SNIP", exact=True).click()
    page.get_by_role("button", name="Preview", exact=True).click()
    page.get_by_text("solver: completed", exact=False).wait_for()
    preview = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    assert preview.evaluate("c => c._fullLayout.xaxis.matches") == "x2"
    assert preview.evaluate("c => c._fullData[1].line.dash") == "dash"
    page.get_by_role("button", name="Apply to all spectra", exact=True).click()
    page.get_by_text("Applied: SNIP", exact=False).last.wait_for()
    page.get_by_role("tab", name="Spectra", exact=True).click()
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return c?._fullLayout?.yaxis.title.text.includes('Background-subtracted') ||
               c?._fullLayout?.yaxis.title.text.includes('background-subtracted');
    }""")
    page.wait_for_function("""() => {
        const l = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout;
        return l.xaxis.range[0] === 20 && l.xaxis.range[1] === 80 && l.yaxis.range[0] === -4 && l.yaxis.range[1] === -2;
    }""")
    page.get_by_role("tab", name="Background", exact=True).click()
    for width in (1700, 760, 390):
        page.set_viewport_size({"width": width, "height": 1100})
        page.wait_for_function("""() => {
            const block = document.querySelector('.st-key-background_layout');
            return block && block.getBoundingClientRect().width <= window.innerWidth;
        }""")
        # Inputs/actions must fit their containers without horizontal page scrolling.
        controls = page.locator('.st-key-background_layout input, .st-key-background_layout button')
        for control in controls.all():
            if control.is_visible():
                box = control.bounding_box()
                assert box["x"] >= -1 and box["x"] + box["width"] <= width + 1
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        if width <= 760:
            columns = page.locator('.st-key-background_layout [data-testid="stColumn"]')
            first, second = columns.nth(0).bounding_box(), columns.nth(1).bounding_box()
            assert second["y"] >= first["y"] + first["height"] - 1
        screenshot_dir = os.environ.get("EELS_BACKGROUND_SCREENSHOTS")
        if screenshot_dir:
            page.screenshot(path=f"{screenshot_dir}/background-{width}.png", full_page=True)
            preview.scroll_into_view_if_needed()
            page.screenshot(path=f"{screenshot_dir}/background-plot-{width}.png", full_page=True)


def test_background_shift_drag_linked_energy_independent_intensity(spectrum_page):
    import math

    page = spectrum_page
    main = page.locator('.st-key-spectrum_axis_target .js-plotly-plot')
    main_before = main.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]')
    page.get_by_role('tab', name='Background', exact=True).click()
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')

    def ranges():
        return chart.evaluate('c => Object.fromEntries(["xaxis", "xaxis2", "yaxis", "yaxis2"].map(a => [a, [...c._fullLayout[a].range]]))')

    def gesture(axis, distance, shift=True):
        group = '.xy' if axis in ('xaxis', 'yaxis') else '.x2y2'
        horizontal = axis.startswith('x')
        handle = chart.locator(f'.draglayer {group} ' + ('.ewdrag' if horizontal else '.nsdrag'))
        handle.scroll_into_view_if_needed()
        box = handle.bounding_box()
        x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        if shift:
            page.keyboard.down('Shift')
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + distance if horizontal else x,
                        y if horizontal else y - distance, steps=10)
        page.mouse.up()
        if shift:
            page.keyboard.up('Shift')

    def wait_ranges(expected):
        page.wait_for_function('''expected => {
            const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
            return Object.entries(expected).every(([axis, range]) =>
                c?._fullLayout?.[axis]?.range.every((v, i) => Math.abs(v - range[i]) < 1e-8));
        }''', arg=expected)

    initial = ranges()
    revision = chart.evaluate('c => c._fullLayout.meta.eels_render_revision')
    data_before = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    bounds_before = [page.get_by_role('spinbutton', name=name, exact=True).input_value()
                     for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')]
    for axis in ('yaxis', 'yaxis2', 'xaxis2', 'xaxis'):
        for distance in (45, -45):
            before = ranges()
            length = chart.evaluate('(c, axis) => c._fullLayout[axis]._length', axis)
            midpoint = sum(before[axis]) / 2
            half_span = (before[axis][1] - before[axis][0]) / 2
            half_span *= math.exp(-distance / length * math.log(4))
            expected = {**before, axis: [midpoint - half_span, midpoint + half_span]}
            if axis.startswith('x'):
                expected['xaxis'] = expected['xaxis2'] = expected[axis]
            gesture(axis, distance)
            wait_ranges(expected)

    # Native pan remains available and moves only the chosen intensity axis.
    before = ranges()
    gesture('yaxis2', 35, shift=False)
    page.wait_for_function('''old => {
        const r = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')._fullLayout.yaxis2.range;
        return Math.abs(r[0] - old[0]) > 1e-8;
    }''', arg=before['yaxis2'])
    after = ranges()
    assert after['yaxis2'][1] - after['yaxis2'][0] == pytest.approx(before['yaxis2'][1] - before['yaxis2'][0])
    for axis in ('xaxis', 'xaxis2', 'yaxis'):
        assert after[axis] == pytest.approx(before[axis])
    assert chart.evaluate('c => c._fullLayout.meta.eels_render_revision') == revision
    np.testing.assert_array_equal(
        chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), data_before)
    assert [page.get_by_role('spinbutton', name=name, exact=True).input_value()
            for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')] == bounds_before

    # A rerun retains all subplot ranges, with no interference from Spectra.
    page.get_by_role('tab', name='Spectra', exact=True).click()
    assert main.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]') == main_before
    page.get_by_text('Show hover details', exact=True).click()
    page.wait_for_function('''() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')?._fullLayout?.hovermode === false''')
    page.get_by_role('tab', name='Background', exact=True).click()
    wait_ranges(after)
    chart.locator('[data-title="Reset axes"]').click(force=True)
    wait_ranges(initial)


def test_background_log_display_keeps_energy_zoom_and_axis_gestures(spectrum_page):
    import math

    page = spectrum_page
    page.get_by_role('tab', name='Background', exact=True).click()
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')
    linear_data = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    chart.evaluate('''c => window.Plotly.relayout(c, {
        'xaxis2.range': [10, 120], 'yaxis.range': [0.01, 0.1], 'yaxis2.range': [-0.002, 0.002]
    })''')
    page.locator('.st-key-bg_display [role="combobox"]').click()
    page.get_by_role('option', name='log10', exact=True).click()
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        const l = c?._fullLayout;
        return l?.yaxis2.title.text === 'log10(positive residual)' &&
            l.xaxis.range[0] === 10 && l.xaxis2.range[1] === 120;
    }''')
    assert chart.evaluate('c => c._fullLayout.yaxis.range') != [0.01, 0.1]
    assert chart.evaluate('c => c._fullLayout.yaxis2.range') != [-0.002, 0.002]
    assert not chart.evaluate('c => c.layout.shapes.some(s => s.type === "line" && s.y0 === 0 && s.y1 === 0)')
    from background_core import corrected_display
    logged_data = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    for actual, original in zip(logged_data, linear_data):
        np.testing.assert_allclose(actual, corrected_display(original, 'log10'), rtol=1e-12, equal_nan=True)
    assert chart.evaluate('c => c._fullData.every(t => t.connectgaps === false)')

    handle = chart.locator('.draglayer .x2y2 .nsdrag')
    handle.scroll_into_view_if_needed()
    before = chart.evaluate('c => c._fullLayout.yaxis2.range')
    length = chart.evaluate('c => c._fullLayout.yaxis2._length')
    box = handle.bounding_box()
    x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    page.keyboard.down('Shift')
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y - 45, steps=10)
    page.mouse.up()
    page.keyboard.up('Shift')
    expected_span = (before[1] - before[0]) * math.exp(-45 / length * math.log(4))
    page.wait_for_function('''span => {
        const r = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')._fullLayout.yaxis2.range;
        return Math.abs((r[1] - r[0]) / span - 1) < 1e-6;
    }''', arg=expected_span)
    assert sum(chart.evaluate('c => c._fullLayout.yaxis2.range')) == pytest.approx(sum(before))
    page.locator('.st-key-bg_display [role="combobox"]').click()
    page.get_by_role('option', name='Linear', exact=True).click()
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullLayout?.yaxis2.title.text === 'Signed residual' &&
            c._fullLayout.xaxis.range[0] === 10 && c._fullLayout.xaxis2.range[1] === 120;
    }''')
    np.testing.assert_array_equal(chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), linear_data)
    assert chart.evaluate('c => c.layout.shapes.some(s => s.type === "line" && s.y0 === 0 && s.y1 === 0)')


@pytest.mark.parametrize("method", ["SNIP", "arPLS"])
def test_preview_interval_edit_replaces_old_boundary(spectrum_page, tmp_path, method):
    page = spectrum_page
    # Replace the fixture's temporary scan with enough bins for both intervals.
    energy = np.arange(600.)
    y = 2 + .001 * energy + .05 * np.sin(energy / 3) + np.exp(-((energy - 520) / 5) ** 2)
    np.save(tmp_path / 'scan_T300K.npy', y[:, None, None])
    page.get_by_role('button', name='Refresh files', exact=True).click()
    page.get_by_role('tab', name='Background', exact=True).click()
    if method == "SNIP":
        page.locator('.st-key-bg_method [role="combobox"]').click()
        page.get_by_role('option', name='SNIP', exact=True).click()
    else:
        log_lambda = page.get_by_role('spinbutton', name='log10(λ), numeric', exact=True)
        log_lambda.fill('2.1')
        log_lambda.press('Enter')
    minimum = page.get_by_role('spinbutton', name='Fit energy minimum (meV)', exact=True)
    maximum = page.get_by_role('spinbutton', name='Fit energy maximum (meV)', exact=True)
    minimum.fill('190')
    minimum.press('Enter')
    maximum.fill('250')
    maximum.press('Enter')
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('Requested: 190–250 meV', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis2.range': [160, 270]})")
    # Match editing an input and clicking Preview directly, without pressing Enter.
    minimum.fill('150')
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('Requested: 150–250 meV', exact=False).wait_for()
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    try:
        page.wait_for_function('''() => {
            const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
            const rects = c?._fullLayout?.shapes.filter(s => s.type === 'rect');
            return rects?.length === 2 && rects.every(s => s.x0 >= 150 && s.x0 < 151 && s.x1 <= 250);
        }''', timeout=5000)
    except PlaywrightTimeoutError:
        details = chart.evaluate('''c => ({
            shapes: c._fullLayout.shapes.map(s => [s.type, s.x0, s.x1]),
            inputShapes: c.layout.shapes.map(s => [s.type, s.x0, s.x1]),
            starts: c._fullData.map(t => t.x[Array.from(t.y).findIndex(Number.isFinite)]),
            ranges: [c._fullLayout.xaxis.range, c._fullLayout.xaxis2.range],
            meta: c._fullLayout.meta
        })''')
        pytest.fail(f'Plot did not reflect the new interval: {details}')
    starts = chart.evaluate('''c => c._fullData.slice(1).map(t => {
        const j = Array.from(t.y).findIndex(Number.isFinite);
        return t.x[j];
    })''')
    assert len(starts) == 2 and all(150 <= value < 151 for value in starts)
    assert chart.evaluate('c => c._fullLayout.xaxis.range') == pytest.approx([160, 270])


def test_focus_interval_rescales_intensity_without_refitting(spectrum_page, tmp_path):
    page = spectrum_page
    y = 2 + 0.1 * np.sin(np.arange(600.) / 4)
    y[300] = 1e6  # dominant zero-loss bin outside the selected viewing interval
    np.save(tmp_path / 'scan_T300K.npy', y[:, None, None])
    page.get_by_role('button', name='Refresh files', exact=True).click()
    page.get_by_role('tab', name='Background', exact=True).click()
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    for label, value in [('Fit energy minimum (meV)', '150'), ('Fit energy maximum (meV)', '250')]:
        field = page.get_by_role('spinbutton', name=label, exact=True)
        field.fill(value)
        field.press('Enter')
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')

    def wait_focus():
        page.wait_for_function('''() => {
            const l = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._fullLayout;
            return l?.xaxis.range[0] === 150 && l.xaxis2.range[1] === 250 &&
                l.yaxis.range[1] < 1e-4 && l.yaxis2.range[0] < 0 && l.yaxis2.range[1] > 0;
        }''')

    wait_focus()
    data = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    page.wait_for_function('''() => document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._ev?.listenerCount('plotly_relayout') > 0''')
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis2.range': [190, 230], 'yaxis.range': [0, 0.1]})")
    page.get_by_role('button', name='Focus fit interval', exact=True).click()
    wait_focus()
    np.testing.assert_array_equal(chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), data)
    page.get_by_role('button', name='Show full spectrum', exact=True).click()
    page.wait_for_function('''() => {
        const l = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._fullLayout;
        return l?.xaxis.range[0] < -270 && l.xaxis2.range[1] > 270 && l.yaxis.range[1] > 0.9;
    }''')
    np.testing.assert_array_equal(chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), data)
    page.get_by_role('button', name='Focus fit interval', exact=True).click()
    wait_focus()
    assert page.get_by_role('spinbutton', name='Fit energy minimum (meV)', exact=True).input_value() == '150.000000'
    assert page.get_by_role('spinbutton', name='Fit energy maximum (meV)', exact=True).input_value() == '250.000000'
