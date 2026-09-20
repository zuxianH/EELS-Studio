import csv
from dataclasses import replace
import io
import json

import numpy as np
import pytest

from background_core import (ANALYTIC_MODELS, BACKGROUND_MODELS, BackgroundConfig, BackgroundState,
    auto_segments_from_peaks, config_caption, default_analytic_segments, fit_background, input_fingerprints)
from background_exports import background_csv, background_npz
from background_view import preview_figure

ENERGY_FACTOR = 40.0


def analytic_curve(model_name, n=400, seed=0):
    spec = BACKGROUND_MODELS[model_name]
    x = np.linspace(2.0, 130.0, n)
    true_params = np.clip(np.array(spec.start) * 1.1 + 1e-4, spec.bounds[0], spec.bounds[1])
    background = spec.func(x / ENERGY_FACTOR, *true_params)
    peak = 0.02 * np.exp(-0.5 * ((x - 60.0) / 5.0) ** 2)
    y = background + peak + np.random.default_rng(seed).normal(0, 1e-6, n)
    return x, y, true_params


def analytic_config(model_name, segments=((5.0, 25.0), (95.0, 125.0))):
    spec = BACKGROUND_MODELS[model_name]
    return BackgroundConfig(method=model_name, segments=segments, model_start=spec.start,
                            model_lower=spec.bounds[0], model_upper=spec.bounds[1], energy_factor=ENERGY_FACTOR)


def curve(model_name, n=400, seed=0):
    x, y, true_params = analytic_curve(model_name, n, seed)
    return dict(path=f"scan_{model_name}.npy", sample=0, probe_x=0, probe_y=0,
                label=f"Curve {model_name}", energy=x, intensity=y,
                style=dict(color="#137c78", width=2.0, line_style="Solid")), true_params


def state_for(curves):
    state = BackgroundState()
    state.sync_inputs(input_fingerprints(curves, {}, {c["path"]: (1, 2) for c in curves}))
    return state


@pytest.mark.parametrize("model_name", sorted(ANALYTIC_MODELS))
def test_analytic_model_recovers_parameters_and_reveals_peak(model_name):
    c, true_params = curve(model_name)
    config = analytic_config(model_name)
    result = fit_background(c["energy"], c["intensity"], config)
    d = result.diagnostics
    assert d.valid and d.status == "converged"
    assert d.fit_segments == config.segments
    if model_name != "power0":  # a2 has zero gradient in power0 and is not identifiable
        np.testing.assert_allclose(d.model_coeffs, true_params, atol=5e-3, rtol=5e-2)
    span_mask = (c["energy"] >= 5.0) & (c["energy"] <= 125.0)
    np.testing.assert_array_equal(result.validity_mask, span_mask)
    assert np.isnan(result.baseline[~span_mask]).all()
    assert np.isnan(result.corrected[~span_mask]).all()
    near_peak_center = (c["energy"] > 55.0) & (c["energy"] < 65.0)
    assert result.corrected[near_peak_center].mean() > 0.01  # the injected peak survives subtraction
    assert not np.shares_memory(result.input, c["intensity"])


def test_analytic_model_immutable_and_energy_grid_preserved():
    c, _ = curve("power0")
    before_x, before_y = c["energy"].copy(), c["intensity"].copy()
    fit_background(c["energy"], c["intensity"], analytic_config("power0"))
    np.testing.assert_array_equal(c["energy"], before_x)
    np.testing.assert_array_equal(c["intensity"], before_y)


@pytest.mark.parametrize("n_segments", [2, 3, 4])
def test_default_analytic_segments_stay_within_range_and_ordered(n_segments):
    segments = default_analytic_segments(10.0, 200.0, n_segments)
    assert len(segments) == n_segments
    for lo, hi in segments:
        assert 10.0 <= lo < hi <= 200.0


def _power_law_with_peak(peak_center=60.0, peak_sigma=5.0, peak_amplitude=0.02, extra_peak=None, noise=1e-5, seed=0):
    x = np.linspace(2.0, 130.0, 400)
    background = 0.05 * (x / 40.0) ** -1.5
    y = background + peak_amplitude * np.exp(-0.5 * ((x - peak_center) / peak_sigma) ** 2)
    if extra_peak is not None:
        center, sigma, amplitude = extra_peak
        y = y + amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    y = y + np.random.default_rng(seed).normal(0, noise, x.size)
    return x, y


@pytest.mark.parametrize("n_segments", [2, 3, 4])
def test_auto_segments_from_peaks_excludes_the_peak(n_segments):
    x, y = _power_law_with_peak()
    segments = auto_segments_from_peaks(x, y, n_segments, 2.0, 130.0)
    assert len(segments) == n_segments
    for lo, hi in segments:
        assert 2.0 <= lo < hi <= 130.0
        assert not (lo < 60.0 < hi)  # the peak center is never inside a returned segment


def test_auto_segments_from_peaks_finds_both_peaks_and_leaves_a_middle_segment():
    x, y = _power_law_with_peak(extra_peak=(100.0, 3.0, 0.015))
    segments = auto_segments_from_peaks(x, y, 3, 2.0, 130.0)
    assert len(segments) == 3
    assert any(60.0 < lo and hi < 100.0 for lo, hi in segments)  # a segment sits between the two peaks


def test_auto_segments_from_peaks_recovers_fit_parameters():
    x, y = _power_law_with_peak()
    segments = auto_segments_from_peaks(x, y, 2, 2.0, 130.0)
    spec = BACKGROUND_MODELS["power0"]
    config = BackgroundConfig(method="power0", segments=tuple(segments), model_start=spec.start,
                              model_lower=spec.bounds[0], model_upper=spec.bounds[1], energy_factor=40.0)
    result = fit_background(x, y, config)
    assert result.diagnostics.valid
    np.testing.assert_allclose(result.diagnostics.model_coeffs[:2], (0.05, 1.5), atol=5e-3, rtol=5e-2)


@pytest.mark.parametrize("noise", [1e-5, 1e-4])
def test_auto_segments_from_peaks_rejects_flat_spectrum(noise):
    x = np.linspace(2.0, 130.0, 400)
    background = 0.05 * (x / 40.0) ** -1.5
    y = background + np.random.default_rng(1).normal(0, noise, x.size)
    with pytest.raises(ValueError):
        auto_segments_from_peaks(x, y, 2, 2.0, 130.0)


def test_auto_segments_from_peaks_ignores_a_tiny_relative_bump():
    x, y = _power_law_with_peak(peak_amplitude=0.001)  # ~4% local rise, below the 15% default
    with pytest.raises(ValueError):
        auto_segments_from_peaks(x, y, 2, 2.0, 130.0)


@pytest.mark.parametrize("bad_args", [
    dict(n_segments=1), dict(n_segments=5),          # n_segments out of range
    dict(low=100.0, high=10.0),                       # inverted domain
    dict(relative_prominence=0.0),                    # non-positive prominence
])
def test_auto_segments_from_peaks_rejects_bad_arguments(bad_args):
    x, y = _power_law_with_peak()
    kwargs = dict(n_segments=2, low=2.0, high=130.0)
    kwargs.update(bad_args)
    with pytest.raises(ValueError):
        auto_segments_from_peaks(x, y, kwargs.pop("n_segments"), kwargs.pop("low"), kwargs.pop("high"), **kwargs)


def test_auto_segments_from_peaks_rejects_tiny_domain():
    x, y = _power_law_with_peak()
    with pytest.raises(ValueError):
        auto_segments_from_peaks(x, y, 2, 55.0, 65.0)


@pytest.mark.parametrize("bad_segments", [
    (), [(5.0, 25.0)], [(5.0, 25.0)] * 5,           # wrong segment count
    [(25.0, 5.0), (95.0, 125.0)],                    # inverted
    [(-10.0, 25.0), (95.0, 125.0)],                  # outside recorded domain
    [(5.0, 6.0), (7.0, 8.0)],                        # too few points total
])
def test_analytic_model_rejects_invalid_segments(bad_segments):
    c, _ = curve("power0")
    config = analytic_config("power0", segments=tuple(bad_segments))
    with pytest.raises(ValueError):
        fit_background(c["energy"], c["intensity"], config)


def test_analytic_model_rejects_mismatched_start_bounds_length():
    c, _ = curve("power")
    config = BackgroundConfig(method="power", segments=((5.0, 25.0), (95.0, 125.0)),
                              model_start=(1.0, 2.0), model_lower=(0.0, 0.0, 0.0),
                              model_upper=(2.0, 3.0, 3.0), energy_factor=ENERGY_FACTOR)
    with pytest.raises(ValueError):
        fit_background(c["energy"], c["intensity"], config)


def test_analytic_model_rejects_bad_energy_factor():
    c, _ = curve("power0")
    config = replace(analytic_config("power0"), energy_factor=0.0)
    with pytest.raises(ValueError):
        fit_background(c["energy"], c["intensity"], config)


def test_config_caption_describes_segments():
    caption = config_caption(analytic_config("power0"))
    assert "power0" in caption and "5-25" in caption and "95-125" in caption


def test_preview_figure_shades_each_segment():
    c, _ = curve("power0")
    result = fit_background(c["energy"], c["intensity"], analytic_config("power0"))
    # log10 mode: no zero-reference hline, so every shape is one of our per-segment vrects.
    figure = preview_figure(c, result, mode="log10")
    assert len(figure.layout.shapes) == 4  # two segments x two subplot rows


def test_state_apply_and_exports_for_analytic_model():
    curves_with_truth = [curve("power0", seed=1), curve("power0", seed=2)]
    curves = [c for c, _ in curves_with_truth]
    state = state_for(curves)
    config = analytic_config("power0")
    failures = state.apply(curves, config)
    assert not failures
    assert state.signal == "Corrected"

    csv_rows = list(csv.reader(io.StringIO(background_csv(curves, {}, state, "Corrected").decode("utf-8"))))
    metadata = json.loads(csv_rows[1][-1])
    npz_bytes = background_npz(curves, {}, state, "Corrected")
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as loaded:
        loaded_metadata = json.loads(str(loaded["metadata_json"]))
        assert loaded_metadata["background"]["algorithm_parameters"]["model"] == "power0"
        assert loaded_metadata["background"]["package"] == "scipy.optimize.curve_fit"
        for i in range(len(curves)):
            assert f"curve_{i:03d}_validity_mask" in loaded
    assert metadata == loaded_metadata
