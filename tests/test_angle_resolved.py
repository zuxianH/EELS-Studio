import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from eels_core import (extract_angle_resolved, inspect_scan, rectangle_from_plot,
                       process_angle_resolved, energy_loss_axis_mev,
                       gaussian_broaden_spectrum)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shape", [(75, 7, 11), (2, 75, 3, 2, 7, 11)])
@pytest.mark.parametrize("retain_axis", ["px", "py"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_rectangle_map_matches_numpy(tmp_path, shape, retain_axis, normalize, order):
    data = np.random.default_rng(42).uniform(0.1, 5, shape).astype(np.float32, order=order)
    path = tmp_path / "scan.npy"
    np.save(path, data)
    dummy, x, y = (1, 2, 1) if len(shape) == 6 else (0, 0, 0)
    block = data[dummy, :, x, y] if len(shape) == 6 else data
    pixels, actual = extract_angle_resolved(inspect_scan(path), (1, 4, 3, 8),
        retain_axis=retain_axis, dummy=dummy, probe_x=x, probe_y=y, normalize_3d=normalize)
    expected = block[:, 1:5, 3:9].sum(axis=1 if retain_axis == "py" else 2, dtype=np.float64)
    if normalize:
        expected /= block.sum(dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
    np.testing.assert_array_equal(pixels, np.arange(3, 9) - 5 if retain_axis == "py" else np.arange(1, 5) - 3)


def test_drawn_rectangle_coordinate_convention():
    assert rectangle_from_plot((7, 11), 2.5, 8.5, 0.5, 4.5) == (1, 4, 3, 8)
    assert rectangle_from_plot((7, 11), 8.5, 2.5, 4.5, 0.5) == (1, 4, 3, 8)
    assert rectangle_from_plot((7, 11), -20, 50, -10, 40) == (0, 6, 0, 10)
    assert rectangle_from_plot((7, 11), 3.9, 4.1, 2.9, 3.1) == (3, 3, 4, 4)


@pytest.mark.parametrize("coords", [(1.1, 1.2, 2, 3), (-10, -1, 0, 5), (0, 3, 20, 30),
                                     (np.nan, 3, 0, 4)])
def test_empty_or_invalid_drawn_rectangle(coords):
    with pytest.raises(ValueError):
        rectangle_from_plot((7, 11), *coords)


@pytest.mark.parametrize("bounds", [(4, 1, 3, 8), (0, 7, 0, 10), (-1, 4, 0, 5),
                                     (0, 1, 0), (0, 1, 0, 2.5)])
def test_invalid_map_bounds(tmp_path, bounds):
    path = tmp_path / "scan.npy"
    np.save(path, np.ones((5, 7, 11)))
    with pytest.raises(ValueError, match="bounds"):
        extract_angle_resolved(inspect_scan(path), bounds)


def test_map_normalization_and_nonfinite_data(tmp_path):
    path = tmp_path / "scan.npy"
    data = np.zeros((3, 4, 5))
    np.save(path, data)
    with pytest.raises(ValueError, match="normalize"):
        extract_angle_resolved(inspect_scan(path), (1, 2, 1, 3))
    _, raw = extract_angle_resolved(inspect_scan(path), (1, 2, 1, 3), normalize_3d=False)
    assert raw.shape == (3, 3) and not raw.any()
    data[0, 0, 0] = np.nan
    np.save(path, data)
    with pytest.raises(ValueError, match="normalize"):
        extract_angle_resolved(inspect_scan(path), (1, 2, 1, 3))
    extract_angle_resolved(inspect_scan(path), (1, 2, 1, 3), normalize_3d=False)
    with pytest.raises(ValueError, match="rectangle"):
        extract_angle_resolved(inspect_scan(path), (0, 2, 0, 3), normalize_3d=False)


def test_map_processing_preserves_pixel_axis():
    energy = energy_loss_axis_mev(75)
    raw = np.random.default_rng(11).uniform(0.1, 2, (75, 4))
    raw[:, 2] = 0  # This column must not receive intensity from adjacent pixels.
    original = raw.copy()
    result = process_angle_resolved(energy, raw, unshifted=True, sigma_mev=12)
    expected = np.column_stack([gaussian_broaden_spectrum(energy,
        np.fft.fftshift(column), 12) for column in raw.T])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(raw, original)
    assert not result[:, 2].any()
    np.testing.assert_array_equal(process_angle_resolved(energy, raw), raw)


def test_rectangle_callback_and_stale_events():
    from angle_resolved import receive_rectangle
    keys = ["r0", "r1", "c0", "c1"]
    state = dict(angle_rectangle_context=dict(preview_id="current", shape=(7, 11), bound_keys=keys),
                 angle_rectangle=dict(selected=dict(preview_id="old", x0=2.5, x1=8.5, y0=0.5, y1=4.5)))
    with patch("angle_resolved.st.session_state", state):
        receive_rectangle()
        assert "r0" not in state
        state["angle_rectangle"]["selected"]["preview_id"] = "current"
        receive_rectangle()
        assert tuple(state[k] for k in keys) == (1, 4, 3, 8)


def test_angle_resolved_app_and_exports(tmp_path):
    from streamlit.testing.v1 import AppTest
    from angle_resolved import export_map
    path = tmp_path / "scan_T300K.npy"
    data = np.arange(1, 1 + 21 * 7 * 11, dtype=float).reshape(21, 7, 11)
    np.save(path, data)
    with patch("angle_resolved.export_map", wraps=export_map) as exported:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
        assert not app.exception and not app.error
        for label, value in [("Row min (px)", 1), ("Row max (px)", 4),
                             ("Column min (py)", 3), ("Column max (py)", 8)]:
            next(w for w in app.number_input if w.label == label).set_value(value)
        app.run()
        energy, pixels, intensity, metadata = exported.call_args.args
        expected = data[:, 1:5, 3:9].sum(axis=1) / data.sum()
        np.testing.assert_allclose(intensity, expected)
        np.testing.assert_array_equal(pixels, np.arange(3, 9) - 5)
        assert metadata["roi_bounds_inclusive"] == [1, 4, 3, 8]
        next(w for w in app.checkbox if w.label == "Broaden EELS spectrum").check().run()
        energy, pixels, intensity, metadata = exported.call_args.args
        np.testing.assert_allclose(intensity, process_angle_resolved(
            energy, expected, sigma_mev=1))
        with np.load(io.BytesIO(export_map(energy, pixels, intensity, metadata)), allow_pickle=False) as result:
            np.testing.assert_array_equal(result["intensity"], intensity)
        app.selectbox(key="angle_direction").select("Vertical (px)").run()
        energy, pixels, intensity, metadata = exported.call_args.args
        np.testing.assert_array_equal(pixels, np.arange(1, 5) - 3)
        np.testing.assert_allclose(intensity, process_angle_resolved(
            energy, data[:, 1:5, 3:9].sum(axis=2) / data.sum(), sigma_mev=1))
        assert not app.exception and not app.error
        next(w for w in app.number_input if w.label == "Row min (px)").set_value(6).run()
        assert any("minimum" in item.value for item in app.warning)
        assert not app.exception


def test_map_roi_survives_switching_different_planes(tmp_path):
    from streamlit.testing.v1 import AppTest
    np.save(tmp_path / "a_T300K.npy", np.ones((7, 7, 11)))
    np.save(tmp_path / "b_T1000K.npy", np.ones((5, 3, 5)))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
    sources = app.selectbox(key="angle_source").options
    next(w for w in app.number_input if w.label == "Row min (px)").set_value(2)
    next(w for w in app.number_input if w.label == "Column max (py)").set_value(8)
    app.run()
    app.selectbox(key="angle_source").select_index(1).run()
    assert not app.exception and not app.error
    assert next(w for w in app.number_input if w.label == "Column max (py)").value == 4
    app.selectbox(key="angle_source").select_index(0).run()
    assert len(sources) == 2 and not app.exception and not app.error
    assert next(w for w in app.number_input if w.label == "Row min (px)").value == 2
    assert next(w for w in app.number_input if w.label == "Column max (py)").value == 8


def test_map_plot_and_export_single_pixel():
    from angle_resolved import map_display, map_figures
    shown = map_display(np.array([[0.0], [1.0], [10.0]]), True)
    assert np.isnan(shown[0, 0])
    np.testing.assert_array_equal(shown[1:, 0], [0, 1])
    figures = map_figures(np.array([-1, 0, 1]), np.array([0]), shown,
                          "Test", "Pixels", "log10(I)", (-1, 1), (None, None))
    assert figures["png"].startswith(b"\x89PNG") and b"<svg" in figures["svg"]
