import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from pybaselines import Baseline

from background_core import (BackgroundConfig, BackgroundState, corrected_display,
    fit_background, initial_bounds, input_fingerprints, PROCESSING_ORDER)
from background_exports import background_csv, background_npz
from background_view import preview_figure
from eels_core import (curve_identity_key, display_intensity, energy_loss_axis_mev,
                       extract_spectrum, gaussian_broaden_spectrum, inspect_scan)

ROOT = Path(__file__).resolve().parents[1]


def spectrum(n=501, offset=2):
    x = np.linspace(-50, 200, n)
    baseline = offset + 0.002 * (x + 50) + 0.000005 * (x + 50) ** 2
    y = baseline + 3 * np.exp(-0.5 * ((x - 70) / 3) ** 2)
    y += np.random.default_rng(42).normal(0, 0.015, n)
    return x, y, baseline


def curve(n=501, offset=2):
    x, y, _ = spectrum(n, offset)
    return dict(path=f"scan_{n}_{offset}.npy", dummy=0, probe_x=0, probe_y=0,
                label=f"Curve {offset}", energy=x, intensity=y,
                style=dict(color="#137c78", width=2.0, line_style="Solid"))


def state_for(curves):
    state = BackgroundState()
    state.sync_inputs(input_fingerprints(curves, {}, {c["path"]: (1, 2) for c in curves}))
    return state


@pytest.mark.parametrize("domain", ["full", "selected"])
def test_smooth_baseline_signed_immutable(domain):
    x, y, expected = spectrum()
    before_x, before_y = x.copy(), y.copy()
    r = fit_background(x, y, BackgroundConfig(domain=domain, energy_min=5.2, energy_max=170.3))
    assert r.diagnostics.valid and r.diagnostics.status == "converged"
    assert r.diagnostics.fitted_bins >= 8
    np.testing.assert_array_equal(x, before_x)
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_allclose(r.corrected[r.validity_mask], y[r.validity_mask] - r.baseline[r.validity_mask])
    assert (r.corrected[r.validity_mask] < 0).any()
    assert np.sqrt(np.mean((r.baseline[r.validity_mask] - expected[r.validity_mask]) ** 2)) < 0.03
    assert np.isnan(r.baseline[~r.validity_mask]).all()
    assert np.isnan(r.corrected[~r.validity_mask]).all()
    assert not np.shares_memory(r.input, y)
    if domain == "selected":
        assert r.diagnostics.requested_bounds == (5.2, 170.3)
        assert r.diagnostics.actual_bounds == (5.5, 170.0)


@pytest.mark.parametrize("value", [0, 2, -3])
def test_exact_constants(value):
    x = np.arange(20.)
    r = fit_background(x, np.full(20, value), BackgroundConfig(domain="full"))
    assert r.diagnostics.valid and r.diagnostics.status == "exact constant input"
    np.testing.assert_array_equal(r.baseline, value)
    np.testing.assert_array_equal(r.corrected, 0)


def test_almost_constant_is_not_shortcut():
    y = np.ones(100)
    y[50] += 1e-10
    r = fit_background(np.arange(100.), y, BackgroundConfig(domain="full"))
    assert r.diagnostics.status != "exact constant input"


@pytest.mark.parametrize("kwargs", [dict(method="unknown"), dict(domain="masked"),
    dict(log10_lambda=np.nan), dict(log10_lambda=11), dict(tolerance=0), dict(tolerance=np.inf),
    dict(max_iterations=0), dict(max_iterations=1.5), dict(half_window_mev=0),
    dict(domain="selected", energy_min=np.nan, energy_max=100),
    dict(domain="selected", energy_min=100, energy_max=50),
    dict(domain="selected", energy_min=-51, energy_max=100),
    dict(domain="selected", energy_min=1, energy_max=2)])
def test_invalid_parameters(kwargs):
    x, y, _ = spectrum()
    with pytest.raises(ValueError):
        fit_background(x, y, replace(BackgroundConfig(domain="full"), **kwargs))


@pytest.mark.parametrize("kind", ["shape", "short", "nan", "inf", "descending", "nonuniform", "complex"])
def test_invalid_data(kind):
    x, y, _ = spectrum()
    if kind == "shape": y = y[:, None]
    if kind == "short": x, y = x[:7], y[:7]
    if kind == "nan": y[20] = np.nan
    if kind == "inf": x[10] = np.inf
    if kind == "descending": x = x[::-1]
    if kind == "nonuniform": x[20] += 0.01
    if kind == "complex": y = y.astype(complex)
    with pytest.raises(ValueError):
        fit_background(x, y, BackgroundConfig(domain="full"))


def test_nonconvergence_and_warnings(monkeypatch):
    x, y, _ = spectrum()
    r = fit_background(x, y, BackgroundConfig(domain="full", max_iterations=1, tolerance=1e-12))
    assert not r.diagnostics.valid and r.diagnostics.status == "not converged"
    assert not r.validity_mask.any()
    assert np.isnan(r.corrected).all()
    def warned(self, values, **kwargs):
        warnings.warn("numerical issue", UserWarning)
        return values.copy(), {"tol_history": np.array([0.])}
    monkeypatch.setattr(Baseline, "arpls", warned)
    r = fit_background(x, y, BackgroundConfig(domain="full"))
    assert not r.diagnostics.valid and r.diagnostics.warnings == ("numerical issue",)
    def invalid(self, values, **kwargs):
        return np.full_like(values, np.nan), {"tol_history": np.array([0.])}
    monkeypatch.setattr(Baseline, "arpls", invalid)
    assert not fit_background(x, y, BackgroundConfig(domain="full")).diagnostics.valid


@pytest.mark.parametrize("spacing, bins", [(0.5, 7), (1., 4), (2., 2)])
def test_snip_window_and_linear_input(spacing, bins):
    x = np.arange(101) * spacing
    y = 2 + np.exp(-((x - 20) / 2) ** 2)
    r = fit_background(x, y, BackgroundConfig(method="SNIP", domain="full", half_window_mev=3.5))
    assert r.diagnostics.valid and r.diagnostics.status == "completed"
    assert r.diagnostics.tol_history == ()
    assert r.diagnostics.half_window_bins == bins
    assert r.diagnostics.effective_half_window_mev == bins * spacing
    expected, _ = Baseline(x_data=x).snip(y, max_half_window=bins, decreasing=True, filter_order=2, smooth_half_window=None)
    np.testing.assert_array_equal(r.baseline, expected)


@pytest.mark.parametrize("window", [0.1, 10, 1000])
def test_snip_no_silent_clamping(window):
    with pytest.raises(ValueError, match="half-window"):
        fit_background(np.arange(20.), np.ones(20), BackgroundConfig(method="SNIP", domain="selected",
                       energy_min=2, energy_max=12, half_window_mev=window))


def test_independent_atomic_apply_and_reset():
    curves = [curve(), curve(301, 7)]
    state = state_for(curves)
    cfg = BackgroundConfig(domain="full")
    assert not state.apply(curves, cfg)
    assert state.signal == "Corrected"
    results = list(state.applied_results.values())
    assert results[1].baseline.mean() > results[0].baseline.mean() + 4
    previous = state.applied_results
    bad = replace(cfg, domain="selected", energy_min=0, energy_max=300)
    assert len(state.apply(curves, bad)) == 2
    assert state.applied_results is previous and state.applied_config == cfg
    state.draft = bad
    state.reset()
    assert state.signal == "Input" and not state.applied_results and state.preview is None
    assert state.draft == bad


def test_fingerprint_staleness_and_style_independence():
    curves = [curve()]
    settings = dict(gaussian_sigma_mev=0, normalize_3d=True)
    revisions = {curves[0]["path"]: (1, 2)}
    original = input_fingerprints(curves, settings, revisions)
    curves[0]["label"], curves[0]["style"]["color"] = "Renamed", "#112233"
    assert input_fingerprints(curves, dict(settings, plot={"x_limits": [0, 20]}), revisions) == original
    state = state_for(curves)
    state.sync_inputs(original)
    state.apply(curves, BackgroundConfig(domain="full"))
    assert not state.sync_inputs(original)
    assert state.applied_results
    for changed in (dict(settings, gaussian_sigma_mev=1), dict(settings, normalize_3d=False),
                    dict(settings, detector_radius_px=3), dict(settings, timestep_fs=5)):
        assert input_fingerprints(curves, changed, revisions) != original
    assert input_fingerprints(curves, settings, {curves[0]["path"]: (2, 2)}) != original
    more = curves + [curve(301, 7)]
    assert state.sync_inputs(input_fingerprints(more, settings, {c["path"]: (1, 2) for c in more}))
    assert not state.applied_results and state.signal == "Input"


def test_initial_bounds_and_log_display():
    curves = [curve(), curve(301, 7)]
    low, high = initial_bounds(curves)
    assert low > 0 and high == 200
    low, high = initial_bounds([dict(energy=np.arange(-10., 0))])
    assert (low, high) == (-10, -1)
    values = np.array([-2., 0, 10, np.nan])
    before = values.copy()
    shown = corrected_display(values, "log10")
    np.testing.assert_array_equal(shown, [np.nan, np.nan, 1., np.nan])
    np.testing.assert_array_equal(values, before)
    np.testing.assert_array_equal(corrected_display(values, "linear"), before)
    assert np.isfinite(display_intensity(values[:3], "log10")).all()  # legacy clipping unchanged


@pytest.mark.parametrize("signal", ["Input", "Corrected"])
def test_exports_roundtrip_unequal_lengths(signal):
    curves = [curve(), curve(301, 7)]
    state = state_for(curves)
    assert not state.apply(curves, BackgroundConfig(energy_min=2, energy_max=180))
    with np.load(io.BytesIO(background_npz(curves, {}, state, signal)), allow_pickle=False) as z:
        assert int(z["schema_version"]) == 1
        metadata = json.loads(str(z["metadata_json"]))
        assert metadata["exported_signal"] == signal
        assert metadata["settings"]["processing_order"] == PROCESSING_ORDER
        assert metadata["background"]["package_version"] == "1.2.1"
        for i, c in enumerate(curves):
            key = f"curve_{i:03d}"
            assert z[key].shape == (len(c["energy"]), 2)
            np.testing.assert_array_equal(z[key][:, 1], z[key + ("_corrected" if signal == "Corrected" else "_input")])
            mask = z[key + "_validity_mask"]
            assert mask.dtype == bool
            assert np.isnan(z[key + "_baseline"][~mask]).all()
            np.testing.assert_array_equal(z[key + "_input"], c["intensity"])
    rows = list(csv.DictReader(io.StringIO(background_csv(curves, {}, state, signal).decode())))
    assert len(rows) == 802
    assert rows[0]["baseline"] == rows[0]["corrected_intensity"] == ""
    assert rows[0]["input_intensity"] != ""
    assert json.loads(rows[0]["metadata_json"])["exported_signal"] == signal


def test_preview_plot_linked_axes_and_masking():
    c = curve()
    r = fit_background(c["energy"], c["intensity"], BackgroundConfig(energy_min=5, energy_max=180))
    fig = preview_figure(c, r)
    assert fig.layout.xaxis.matches == "x2"
    assert fig.data[1].line.dash == "dash"
    assert not fig.data[2].connectgaps
    assert fig.data[2].line.color == c["style"]["color"]
    assert any(shape.y0 == shape.y1 == 0 for shape in fig.layout.shapes)
    intervals = [shape for shape in fig.layout.shapes if shape.type == "rect"]
    assert {shape.xref for shape in intervals} == {"x", "x2"}
    assert all((shape.x0, shape.x1) == r.diagnostics.actual_bounds for shape in intervals)


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def test_app_background_workflow_processing_and_maps(tmp_path):
    from streamlit.testing.v1 import AppTest
    data = np.ones((101, 3, 3))
    data[:, :, :] *= (3 + np.exp(-((np.arange(101) - 65) / 3) ** 2))[:, None, None]
    np.save(tmp_path / "a.npy", data)
    np.save(tmp_path / "b.npy", np.broadcast_to(data[None, :, None, None], (1, 101, 2, 1, 3, 3)).copy())
    with patch("background_core.fit_background", wraps=fit_background) as fitted:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        widget(app, "text_input", "Data folder").set_value(str(tmp_path)).run()
        assert not app.exception and not app.error
        assert [t.label for t in app.tabs][:2] == ["Spectra", "Background"]
        assert not any(w.key == "bg_signal" for w in app.radio)  # hidden until a fit is applied
        assert fitted.call_count == 0  # hidden-tab rendering never fits
        widget(app, "selectbox", "Input energy ordering").select("Unshifted FFT").run()
        widget(app, "checkbox", "Broaden EELS spectrum").check().run()
        widget(app, "number_input", "Gaussian σ (meV)").set_value(6.).run()
        app.multiselect(key="probe_positions").set_value([(0, 0), (1, 0)]).run()
        app.selectbox(key="bg_domain").select("Full recorded spectrum").run()
        app.selectbox(key="bg_method").select("SNIP").run()
        app.button(key="bg_preview").click().run()
        state = app.session_state["background_state"]
        assert not app.exception and not app.error
        assert state.preview and not state.applied_results and state.signal == "Input"
        actual_input = state.preview.input.copy()
        expected = gaussian_broaden_spectrum(energy_loss_axis_mev(101),
                    np.fft.fftshift(extract_spectrum(inspect_scan(tmp_path / "a.npy"))), 6.)
        np.testing.assert_array_equal(actual_input, expected)
        count = fitted.call_count
        widget(app, "number_input", "Energy min (meV)").set_value(0.).run()
        assert fitted.call_count == count and app.session_state["background_state"].preview is not None
        app.number_input(key="bg_window").set_value(20.).run()
        assert app.session_state["background_state"].preview is None
        app.button(key="bg_apply").click().run()
        state = app.session_state["background_state"]
        assert not app.exception and not app.error
        assert state.signal == "Corrected" and len(state.applied_results) == 3
        assert app.radio(key="bg_signal").value == "Corrected"
        before = state.applied_results
        widget(app, "color_picker", "Line color").set_value("#ff0000").run()
        assert app.session_state["background_state"].applied_results.keys() == before.keys()
        app.number_input(key="bg_window").set_value(25.).run()
        assert app.session_state["background_state"].applied_config.half_window_mev == 20
        assert any("Unapplied edits" in c.value for c in app.caption)
        # Angle-resolved metadata must remain independent even while corrected is selected.
        from angle_resolved import export_map
        with patch("angle_resolved.export_map", wraps=export_map) as exported:
            widget(app, "checkbox", "Enable angle-resolved map").check().run()
            assert not app.exception
            metadata = exported.call_args.args[3]
            assert "background" not in metadata["settings"]
            assert "background estimation and subtraction" not in metadata["settings"]["processing_order"]
        widget(app, "number_input", "Gaussian σ (meV)").set_value(8.).run()
        assert not app.session_state["background_state"].applied_results
        assert app.session_state["background_state"].signal == "Input"
        assert not any(w.key == "bg_signal" for w in app.radio)  # hidden again once results are cleared
        app.button(key="bg_apply").click().run()
        assert app.session_state["background_state"].applied_results
        app.button(key="bg_reset").click().run()
        assert not app.exception and not app.session_state["background_state"].applied_results
        assert widget(app, "number_input", "Energy min (meV)").value == 0
        assert widget(app, "number_input", "Gaussian σ (meV)").value == 8


def test_atomic_failure_after_a_valid_curve():
    curves = [curve(), curve()]
    curves[1] = dict(curves[1], path="short.npy", energy=np.arange(10.), intensity=np.ones(10))
    state = state_for(curves)
    cfg = BackgroundConfig(energy_min=2, energy_max=150)
    failures = state.apply(curves, cfg)
    assert list(failures) == [curve_identity_key(curves[1])]
    assert not state.applied_results and state.signal == "Input"


def test_app_partial_exports_lambda_sync_and_no_scan_reads(tmp_path):
    from streamlit.testing.v1 import AppTest
    from matplotlib.axes import Axes
    from background_exports import background_npz as export_background
    from cache_layer import cached_background_fit
    cached_background_fit.clear()
    np.save(tmp_path / "constant.npy", np.ones((101, 3, 3)))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    widget(app, "text_input", "Data folder").set_value(str(tmp_path)).run()
    with patch("eels_core._open_scan", side_effect=AssertionError("Background controls must reuse extraction caches")):
        app.slider(key="bg_lambda_slider").set_value(6.).run()
        assert app.number_input(key="bg_lambda_number").value == 6
        app.number_input(key="bg_lambda_number").set_value(4.).run()
        assert app.slider(key="bg_lambda_slider").value == 4
        app.button(key="bg_preview").click().run()
        assert not app.session_state["background_state"].applied_results
        original_plot = Axes.plot
        with patch.object(Axes, "plot", autospec=True, side_effect=original_plot) as plotted, \
                patch("background_exports.background_npz", wraps=export_background) as exported:
            app.button(key="bg_apply").click().run()
            assert not app.exception and not app.error
            assert app.radio(key="bg_signal").value == "Corrected"
            assert any("Partial corrected fits require masked NPZ" in c.value for c in app.caption)
            assert any("Nonpositive corrected samples are masked" in c.value for c in app.caption)
            assert np.isnan(plotted.call_args.args[2]).all()  # zero residual, masked for log Matplotlib
            curves, settings, state, signal = exported.call_args.args
            assert signal == "Corrected"
            assert np.isnan(state.applied_results[curve_identity_key(curves[0])].corrected[:50]).all()
            assert "background estimation and subtraction" not in settings["processing_order"]  # map-safe base settings
        # Labels and styles change without refitting or invalidation.
        identity = next(iter(app.session_state["background_state"].applied_results))
        widget(app, "text_input", "constant.npy").set_value("New label").run()
        assert identity in app.session_state["background_state"].applied_results
        app.radio(key="bg_signal").set_value("Input").run()
        assert not any("Partial corrected fits require masked NPZ" in c.value for c in app.caption)
        assert any(b.label == "Notebook .npy" for b in app.get("download_button"))


def test_preview_log_masks_nonpositive_values_and_keeps_signed_data():
    c = curve()
    r = fit_background(c['energy'], c['intensity'], BackgroundConfig(energy_min=5, energy_max=180))
    baseline = r.baseline.copy()
    baseline[np.flatnonzero(r.validity_mask)[0]] = -0.5
    r = replace(r, baseline=baseline)
    c['intensity'][0] = 0
    c['intensity'][1] = -1
    before = [a.copy() for a in (c['intensity'], r.baseline, r.corrected)]
    linear = preview_figure(c, r)
    logged = preview_figure(c, r, mode='log10')
    for trace, values in zip(logged.data, before):
        np.testing.assert_array_equal(trace.y, corrected_display(values, 'log10'))
        assert trace.connectgaps is False
    assert (r.corrected[r.validity_mask] < 0).any()
    assert not any(s.type == 'line' and s.y0 == s.y1 == 0 for s in logged.layout.shapes)
    assert any(s.type == 'line' and s.y0 == s.y1 == 0 for s in linear.layout.shapes)
    assert logged.layout.yaxis2.title.text == 'log10(positive residual)'
    assert logged.layout.xaxis.matches == 'x2'
    assert logged.layout.xaxis.uirevision == linear.layout.xaxis.uirevision
    assert logged.layout.yaxis.uirevision != linear.layout.yaxis.uirevision
    for actual, expected in zip((c['intensity'], r.baseline, r.corrected), before):
        np.testing.assert_array_equal(actual, expected)


def test_preview_display_does_not_refit_or_change_applied_exports(tmp_path):
    from streamlit.testing.v1 import AppTest
    from cache_layer import cached_background_fit
    cached_background_fit.clear()
    np.save(tmp_path / 'zero_residual.npy', np.ones((101, 3, 3)))
    with patch('background_core.fit_background', wraps=fit_background) as fitted:
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
        widget(app, 'text_input', 'Data folder').set_value(str(tmp_path)).run()
        assert app.selectbox(key='bg_display').value == 'Linear'
        app.button(key='bg_preview').click().run()
        app.button(key='bg_apply').click().run()
        state = app.session_state['background_state']
        preview, preview_key = state.preview, state.preview_key
        c = dict(curve(), path=str(tmp_path / 'zero_residual.npy'))
        before = background_npz([c], {}, state, 'Corrected')
        calls = fitted.call_count
        for display in ('log10', 'Linear'):
            with patch('eels_core._open_scan', side_effect=AssertionError('Display must not reread scans')):
                app.selectbox(key='bg_display').select(display).run()
            assert not app.exception and not app.error
            state = app.session_state['background_state']
            assert state.preview_key == preview_key
            np.testing.assert_array_equal(state.preview.corrected, preview.corrected)
            assert fitted.call_count == calls
            assert background_npz([c], {}, state, 'Corrected') == before
            if display == 'log10':
                assert any('No positive corrected samples' in text.value for text in app.info)
                assert any('zero reference is hidden' in text.value for text in app.caption)
        for key in ('bg_full_view', 'bg_focus_view'):
            with patch('eels_core._open_scan', side_effect=AssertionError('View must not reread scans')):
                app.button(key=key).click().run()
            assert not app.exception and not app.error
            state = app.session_state['background_state']
            assert state.preview_key == preview_key
            assert fitted.call_count == calls
            assert background_npz([c], {}, state, 'Corrected') == before


def test_focused_preview_excludes_offscreen_zero_loss_from_intensity_limits():
    c = curve()
    c['intensity'][np.argmin(abs(c['energy']))] = 1e6
    r = fit_background(c['energy'], c['intensity'], BackgroundConfig(energy_min=50, energy_max=100))
    fig = preview_figure(c, r, view_bounds=(50, 100))
    assert fig.layout.xaxis.range == fig.layout.xaxis2.range == (50, 100)
    assert fig.layout.yaxis.range[1] < 10  # off-screen spike must not set the scale
    assert fig.layout.yaxis2.range[0] < 0 < fig.layout.yaxis2.range[1]
    np.testing.assert_array_equal(fig.data[0].y, c['intensity'])  # no cropping of data
    np.testing.assert_array_equal(fig.data[2].y, r.corrected)
    full = preview_figure(c, r, view_bounds=(c['energy'][0], c['energy'][-1]), view_revision=1)
    assert full.layout.yaxis.range[1] > 1e6
    assert full.layout.xaxis.uirevision != fig.layout.xaxis.uirevision
    logged = preview_figure(c, r, 'log10', view_bounds=(50, 100))
    assert np.isfinite(logged.layout.yaxis.range).all()
    assert logged.layout.yaxis.range[1] < 1
