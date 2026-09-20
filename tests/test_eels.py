import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from eels_core import (
    circular_detector_mask, diffraction_pattern, energy_loss_axis_mev,
    detector_offsets_from_click, export_csv, export_npz, extract_spectrum,
    gaussian_broaden_spectrum, inspect_scan,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def notebook():
    notebook_path = ROOT / "STEM-EELS.ipynb"
    if not notebook_path.exists():
        pytest.skip("STEM-EELS.ipynb is not included in this repository")
    cells = json.loads(notebook_path.read_text())["cells"]
    namespace = {"np": np}
    # Execute only function definitions, not the notebook's data-loading examples.
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("def energy_loss_axis_mev") or source.startswith("def circular_detector_mask"):
            exec(source, namespace)
    return namespace


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("shape", [(11, 7, 8), (2, 11, 3, 2, 7, 8)])
def test_matches_notebook(tmp_path, notebook, shape, normalize):
    data = np.random.default_rng(42).uniform(0.1, 20, shape)
    path = tmp_path / "scan.npy"
    np.save(path, data)
    info = inspect_scan(path)
    sample, x, y = (1, 2, 1) if len(shape) == 6 else (0, 0, 0)
    canonical = data if len(shape) == 6 else data[None, :, None, None, :, :]
    actual = extract_spectrum(info, sample=sample, probe_x=x, probe_y=y,
                              radius=2.5, offset_px=-1, offset_py=2, normalize_3d=normalize)
    expected = notebook["extract_probe_spectrum"](
        canonical, probe_x=x, probe_y=y, sample=sample, radius=2.5,
        detector_center=(shape[-2] // 2 - 1, shape[-1] // 2 + 2), normalize_3d=normalize)
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    np.testing.assert_array_equal(diffraction_pattern(info, 4, sample=sample, probe_x=x, probe_y=y),
                                  canonical[sample, 4, x, y])


@pytest.mark.parametrize("length", [1, 5, 600])
def test_energy_axis(notebook, length):
    np.testing.assert_array_equal(energy_loss_axis_mev(length, 5.0, 3),
                                  notebook["energy_loss_axis_mev"](length, 5.0, 3))


def test_detector_boundary_and_axes():
    mask = circular_detector_mask((7, 9), center=(2, 6), radius=1)
    assert mask.sum() == 5
    assert mask[2, 6] and mask[1, 6] and mask[2, 7]
    assert not mask[3, 7]


def test_click_coordinates_and_mixed_planes():
    # Plotly x is py, Plotly y is px; use asymmetric coordinates to catch swaps.
    assert detector_offsets_from_click((267, 266), 173, 93, [(267, 266)]) == (40, -40)
    assert detector_offsets_from_click((267, 266), 133, 133, [(267, 266), (9, 8)]) == (0, 0)
    with pytest.raises(ValueError, match="another selected scan"):
        detector_offsets_from_click((267, 266), 173, 93, [(267, 266), (9, 8)])
    with pytest.raises(ValueError, match="inside"):
        detector_offsets_from_click((267, 266), -1, 93, [(267, 266)])


def test_gaussian_broadening_shape_and_units():
    energy = np.arange(-50, 50.25, 0.25)
    original = np.zeros(len(energy))
    original[len(energy) // 2] = 1
    broadened = gaussian_broaden_spectrum(energy, original, sigma_mev=3)
    assert broadened.argmax() == original.argmax()
    np.testing.assert_allclose(broadened, broadened[::-1])
    assert broadened.sum() == pytest.approx(1)
    # A delta peak acquires the requested variance, allowing for 4-sigma truncation.
    assert np.sqrt(np.sum(broadened * energy ** 2)) == pytest.approx(3, rel=0.002)
    assert broadened.max() < original.max()
    np.testing.assert_array_equal(original, gaussian_broaden_spectrum(energy, original, 0))
    assert original.sum() == 1 and original.max() == 1


def test_gaussian_boundaries():
    energy = np.arange(60, dtype=float)
    np.testing.assert_allclose(gaussian_broaden_spectrum(energy, np.ones(60), 4), 1)
    edge = np.zeros(60)
    edge[0] = 1
    result = gaussian_broaden_spectrum(energy, edge, 4)
    assert result.sum() == pytest.approx(1)
    assert result[-1] == 0  # Energy endpoints must not wrap around.


@pytest.mark.parametrize("energy, values, sigma", [
    ([0, 1, 3], [1, 2, 3], 1), ([0], [1], 1),
    ([0, 1], [1, 2], -1), ([0, 1], [1, 2], np.nan),
    ([0, 1], [1, np.nan], 1), ([0, 1], [1, 2], 10),
])
def test_bad_broadening(energy, values, sigma):
    with pytest.raises(ValueError):
        gaussian_broaden_spectrum(energy, values, sigma)


@pytest.mark.parametrize("kwargs", [{"radius": 0}, {"radius": np.nan}, {"center": (-1, 0)},
                                    {"center": (7, 0)}, {"center": (np.inf, 0)}])
def test_bad_detector(kwargs):
    with pytest.raises(ValueError):
        circular_detector_mask((7, 9), **kwargs)


@pytest.mark.parametrize("shape", [(10, 2), (1, 2, 3, 4), (0, 2, 3)])
def test_bad_shape(tmp_path, shape):
    path = tmp_path / "bad.npy"
    np.save(path, np.ones(shape))
    with pytest.raises(ValueError):
        inspect_scan(path)


def test_invalid_data_and_indices(tmp_path):
    path = tmp_path / "scan.npy"
    np.save(path, np.zeros((5, 7, 8)))
    info = inspect_scan(path)
    with pytest.raises(ValueError, match="Cannot normalize"):
        extract_spectrum(info)
    np.testing.assert_array_equal(extract_spectrum(info, normalize_3d=False), np.zeros(5))
    with pytest.raises(ValueError, match="3D"):
        extract_spectrum(info, probe_x=1)
    data = np.ones((5, 7, 8))
    data[0, 3, 4] = np.nan
    np.save(path, data)
    # Make the metadata change deterministic even on coarse filesystem clocks.
    os.utime(path, ns=(info.mtime_ns + 1_000_000_000, info.mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="File changed"):
        extract_spectrum(info)
    with pytest.raises(ValueError, match="NaN"):
        extract_spectrum(inspect_scan(path))


@pytest.mark.parametrize("dtype", [complex, object, str])
def test_reject_nonreal_arrays(tmp_path, dtype):
    path = tmp_path / "bad.npy"
    np.save(path, np.ones((2, 3, 4)).astype(dtype))
    with pytest.raises(ValueError):
        inspect_scan(path)


def test_multiblock_normalization(tmp_path):
    path = tmp_path / "scan.npy"
    data = np.random.default_rng(3).random((75, 9, 6)).astype(np.float32)
    np.save(path, data)
    mask = circular_detector_mask((9, 6), radius=2)
    expected = data[:, mask].sum(axis=1, dtype=np.float64) / data.sum(dtype=np.float64)
    np.testing.assert_allclose(extract_spectrum(inspect_scan(path), radius=2), expected, rtol=1e-14)


def test_exports_different_energy_lengths():
    curves = [dict(label=f"Test, {n}", path="scan.npy", sample=0, probe_x=0, probe_y=0,
                   energy=energy_loss_axis_mev(n), intensity=np.arange(n, dtype=float)) for n in (5, 6)]
    with np.load(io.BytesIO(export_npz(curves, {"stride": 3})), allow_pickle=False) as result:
        assert result["curve_000"].shape == (5, 2)
        assert result["curve_001"].shape == (6, 2)
        metadata = json.loads(str(result["metadata_json"]))
        assert metadata["settings"]["stride"] == 3
        assert metadata["curves"][0]["label"] == "Test, 5"
        np.testing.assert_array_equal(result["curve_001"][:, 1], curves[1]["intensity"])
    import csv
    rows = list(csv.DictReader(io.StringIO(export_csv(curves).decode())))
    assert len(rows) == 11
    assert rows[0]["label"] == "Test, 5"


def test_app_scan_selection(tmp_path):
    from streamlit.testing.v1 import AppTest
    # Keep UI coverage independent of untracked simulation data on disk.
    for name in ("scan_T300K", "scan_T1000K"):
        np.save(tmp_path / f"{name}.npy", np.ones((600, 9, 8)))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "2"
    assert app.metric[2].value == "600"
    assert app.metric[3].value == "0.919"
    radius = next(w for w in app.number_input if w.label == "radius")
    radius.set_value(12.0).run()
    assert not app.exception
    assert not app.error
    files = next(w for w in app.multiselect if w.label == "Files to compare")
    files.set_value([files.value[0]]).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    next(w for w in app.selectbox if w.label == "Intensity display").select("Linear").run()
    assert not app.exception
    next(w for w in app.multiselect if w.label == "Files to compare").set_value([]).run()
    assert not app.exception
    assert any("Select at least one scan" in message.value for message in app.info)


def test_app_six_dimensional_files(tmp_path):
    from streamlit.testing.v1 import AppTest
    from unittest.mock import patch

    for name, shape in [("a", (2, 7, 3, 2, 9, 8)), ("b", (2, 6, 2, 2, 9, 8))]:
        np.save(tmp_path / f"{name}.npy", np.random.default_rng(2).uniform(1, 2, shape))
    with patch("eels_core.export_npz", wraps=export_npz) as exported:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
        assert not app.exception and not app.error
        positions = [(0, 1), (1, 0)]
        app.multiselect(key="probe_positions").set_value(positions).run()
        assert not app.exception and not app.error
        assert app.metric[1].value == "4"
        assert all(w.label not in ("Sample index", "Probe y index") for w in app.number_input)
        curves, settings = exported.call_args.args
        assert settings["probe_positions_xy"] == positions
        assert [(c["probe_x"], c["probe_y"]) for c in curves] == positions * 2
        for curve in curves:
            expected = extract_spectrum(inspect_scan(curve["path"]),
                                        probe_x=curve["probe_x"], probe_y=curve["probe_y"])
            np.testing.assert_allclose(curve["intensity"], expected)
            assert f'x={curve["probe_x"]}, y={curve["probe_y"]}' in curve["label"]
        next(w for w in app.selectbox if w.label == "Preview spectrum").select(1).run()
        assert not app.exception and not app.error
        next(w for w in app.selectbox if w.label == "Input energy ordering").select("Unshifted FFT").run()
        assert not app.exception and not app.error
        app.multiselect(key="probe_positions").set_value([]).run()
        assert any("Select at least one probe position" in message.value for message in app.info)
        assert not app.exception
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(ROOT)).run()
        assert not app.exception and not app.error


def test_app_detector_preview_toggle_shows_and_hides_beside_spectrum(tmp_path):
    from streamlit.testing.v1 import AppTest

    np.save(tmp_path / "scan.npy", np.random.default_rng(3).uniform(1, 2, (2, 7, 3, 3, 8, 8)))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
    assert not app.exception and not app.error
    # The sidebar's "Detector preview" picker stays available regardless of the toggle.
    assert any(w.label == "Preview spectrum" for w in app.selectbox)
    shown_charts = len(app.get("plotly_chart"))

    app.toggle(key="show_detector_preview").set_value(False).run()
    assert not app.exception and not app.error
    assert any(w.label == "Preview spectrum" for w in app.selectbox)
    assert len(app.get("plotly_chart")) == shown_charts - 1

    app.toggle(key="show_detector_preview").set_value(True).run()
    assert not app.exception and not app.error
    assert any(w.label == "Preview spectrum" for w in app.selectbox)
    assert len(app.get("plotly_chart")) == shown_charts


def test_app_gaussian_broadening_processing_and_exports(tmp_path):
    from streamlit.testing.v1 import AppTest
    from unittest.mock import patch

    np.save(tmp_path / "scan.npy", np.arange(1, 22, dtype=float).reshape(1, 21, 1, 1, 1, 1))
    with patch("eels_core.export_npz", wraps=export_npz) as exported:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
        assert not app.exception and not app.error
        original, settings = exported.call_args.args
        next(w for w in app.selectbox if w.label == "Input energy ordering").select("Unshifted FFT").run()
        next(w for w in app.checkbox if w.label == "Broaden EELS spectrum").check().run()
        next(w for w in app.number_input if w.label == "Gaussian σ (meV)").set_value(15.0).run()
        assert not app.exception and not app.error
        processed, settings = exported.call_args.args
        for before, after in zip(original, processed):
            expected = gaussian_broaden_spectrum(
                after["energy"], np.fft.fftshift(before["intensity"]), 15.0)
            np.testing.assert_allclose(after["intensity"], expected)
        with np.load(io.BytesIO(export_npz(processed, settings)), allow_pickle=False) as result:
            np.testing.assert_array_equal(result["curve_000"][:, 1], processed[0]["intensity"])
