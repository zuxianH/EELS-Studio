import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from eels_core import (circular_detector_mask, detector_scan_map, diffraction_pattern,
                       energy_loss_axis_mev, extract_spectrum, inspect_scan)
from test_eels import ROOT


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("shape", [(37, 5, 7), (2, 37, 3, 4, 5, 7)])
@pytest.mark.parametrize("dtype", ["<f4", ">f8"])
def test_direct_reads_without_numpy_load(tmp_path, order, shape, dtype):
    data = np.array(np.random.default_rng(5).uniform(1, 3, shape), dtype=dtype, order=order)
    path = tmp_path / "scan.npy"
    np.save(path, data)
    canonical = data if data.ndim == 6 else data[None, :, None, None]
    dummy, x, y = (1, 2, 3) if data.ndim == 6 else (0, 0, 0)
    mask = circular_detector_mask(shape[-2:], center=(1, 4), radius=1.5)
    block = canonical[dummy, :, x, y]
    expected = block[:, mask].sum(axis=1, dtype=np.float64)
    with patch("numpy.load", side_effect=AssertionError("Full-file load/map attempted")):
        info = inspect_scan(path)
        for normalize in (False, True):
            actual = extract_spectrum(info, dummy=dummy, probe_x=x, probe_y=y,
                                      radius=1.5, offset_px=-1, offset_py=1, normalize_3d=normalize)
            np.testing.assert_allclose(actual, expected / block.sum(dtype=np.float64) if normalize else expected)
        np.testing.assert_array_equal(diffraction_pattern(info, 20, dummy=dummy, probe_x=x, probe_y=y), block[20])
        if data.ndim == 6:
            result = detector_scan_map(info, 20, dummy=dummy, radius=1.5, offset_px=-1, offset_py=1)
            np.testing.assert_allclose(result, canonical[dummy, 20, :, :, mask].sum(axis=0, dtype=np.float64))


def test_map_reads_one_row_at_selected_energy(tmp_path):
    data = np.arange(2 * 7 * 3 * 4 * 5 * 6, dtype=float).reshape(2, 7, 3, 4, 5, 6)
    path = tmp_path / "scan.npy"
    np.save(path, data)
    with patch("numpy.fromfile", wraps=np.fromfile) as reads:
        info = inspect_scan(path)
        assert not reads.called  # Inspection reads only the header.
        actual = detector_scan_map(info, 5, dummy=1, radius=1)
        assert reads.call_count == 3
        assert all(c.kwargs["count"] == 4 * 5 * 6 for c in reads.call_args_list)
    mask = circular_detector_mask((5, 6), radius=1)
    np.testing.assert_array_equal(actual, np.sum(data[1, 5], axis=(-2, -1), where=mask))


def test_zero_invalid_and_changed_map(tmp_path):
    path = tmp_path / "scan.npy"
    np.save(path, np.zeros((1, 3, 2, 4, 5, 6)))
    info = inspect_scan(path)
    np.testing.assert_array_equal(detector_scan_map(info, 1), np.zeros((2, 4)))
    for kwargs in ({"energy_index": 3}, {"energy_index": 0, "dummy": 1}, {"energy_index": 0, "radius": 0}):
        with pytest.raises(ValueError):
            detector_scan_map(info, **kwargs)
    data = np.zeros(info.shape)
    data[0, 1, 0, 0, 2, 3] = np.nan
    np.save(path, data)
    # Make the metadata change deterministic even on coarse filesystem clocks.
    os.utime(path, ns=(info.mtime_ns + 1_000_000_000, info.mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="File changed"):
        detector_scan_map(info, 1)
    with pytest.raises(ValueError, match="NaN"):
        detector_scan_map(inspect_scan(path), 1)
    with path.open("r+b") as stream:
        stream.truncate(path.stat().st_size - 1)
    with pytest.raises(ValueError, match="Truncated"):
        inspect_scan(path)


def test_map_app_defaults_and_no_spectrum_extraction(tmp_path):
    from streamlit.testing.v1 import AppTest
    path = tmp_path / "zero_scan.npy"
    np.save(path, np.zeros((2, 7, 3, 4, 5, 6)))
    with patch("eels_core.extract_spectrum", side_effect=AssertionError("Map should not extract spectra")):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        app.radio(key="visualization").set_value("2D scan map").run()
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
        assert not app.exception and not app.error
        assert app.text_input(key="map_energies").value == "0"
        assert app.number_input(key="map_timestep").value == 5.0
        assert app.number_input(key="map_stride").value == 3
        assert app.number_input(key="map_radius").value == 21.0
        assert app.metric[1].value == "12"
        assert any("all zero" in item.value for item in app.info)
        # Single-map PNG export lives in the chart toolbar; workspace saving
        # must also remain available in this visualization mode.
        assert any(w.key == "workspace_download" for w in app.get("download_button"))
        chart_config = json.loads(app.get("plotly_chart")[0].proto.config)
        assert chart_config["toImageButtonOptions"]["format"] == "png"
        assert all(w.label != "Dummy index" for w in app.number_input)
        app.selectbox(key="map_ordering").select("Unshifted FFT").run()
        assert not app.exception and not app.error
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        assert spec["layout"]["xaxis"]["title"]["text"] == "Probe x (index)"
        assert spec["layout"]["yaxis"]["title"]["text"] == "Probe y (index)"
        assert spec["data"][0]["z"]["shape"] == "4, 3"


def test_map_app_3d_explanation(tmp_path):
    from streamlit.testing.v1 import AppTest
    np.save(tmp_path / "single.npy", np.ones((7, 3, 4)))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
    app.radio(key="visualization").set_value("2D scan map").run()
    next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
    assert not app.exception and not app.error
    assert any("Select a 6D scan" in item.value for item in app.info)


def test_large_map_row_is_split_into_bounded_reads(tmp_path):
    data = np.ones((1, 1, 1, 60, 199, 199), dtype=np.float64)
    path = tmp_path / "wide_row.npy"
    np.save(path, data)
    with patch("numpy.fromfile", wraps=np.fromfile) as reads:
        result = detector_scan_map(inspect_scan(path), 0)
        assert reads.call_count == 2
        assert all(c.kwargs["count"] * data.itemsize <= 16 * 2**20 for c in reads.call_args_list)
        assert sum(c.kwargs["count"] for c in reads.call_args_list) == data.size
    np.testing.assert_array_equal(result, np.full((1, 60), circular_detector_mask((199, 199), radius=21).sum()))


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_supported_npy_header_versions(tmp_path, version):
    path = tmp_path / "scan.npy"
    data = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    with path.open("wb") as stream:
        np.lib.format.write_array(stream, data, version=version, allow_pickle=False)
    np.testing.assert_array_equal(diffraction_pattern(inspect_scan(path), 1), data[1])


@pytest.mark.parametrize("text", ["", " , ; ", "20, bad", "nan", "10, inf", "20-10"])
def test_invalid_map_energy_list(text):
    from scan_map_view import parse_map_energies
    with pytest.raises(ValueError):
        parse_map_energies(text)


def test_map_energy_list_parses_points_and_ranges():
    from scan_map_view import parse_map_energies
    requests = parse_map_energies("20, -40; 6e1\n60.01 10-20")
    assert requests == [(20, 20), (-40, -40), (60, 60), (60.01, 60.01), (10, 20)]


def test_map_energy_list_and_duplicate_bins():
    from scan_map_view import parse_map_energies, selected_map_bins
    requests = parse_map_energies("20, -40; 6e1\n60.01")
    axis = np.array([-60., -40., -20., 0., 20., 40., 60.])
    for ordering in ("FFT-shifted (notebook default)", "Unshifted FFT"):
        bins = selected_map_bins(axis, requests, ordering)
        assert [b["bin_energies_mev"] for b in bins] == [[20], [-40], [60]]
        assert bins[2]["requested"] == [[60, 60], [60.01, 60.01]]
        expected = [4, 1, 6] if ordering.startswith("FFT-shifted") else [1, 5, 3]
        assert [b["energy_indices"] for b in bins] == [[i] for i in expected]


def test_map_energy_range_sums_every_enclosed_bin():
    from scan_map_view import bin_title, selected_map_bins
    axis = np.array([-60., -40., -20., 0., 20., 40., 60.])
    bins = selected_map_bins(axis, [(-25, 25)], "FFT-shifted (notebook default)")
    assert len(bins) == 1
    entry = bins[0]
    assert entry["axis_indices"] == [2, 3, 4]
    assert entry["bin_energies_mev"] == [-20, 0, 20]
    assert entry["energy_indices"] == [2, 3, 4]
    assert bin_title(entry) == "-20–20 meV (3 bins)"
    with pytest.raises(ValueError, match="No energy bins"):
        selected_map_bins(axis, [(21, 25)], "FFT-shifted (notebook default)")


def test_app_multiple_energy_maps_and_exports(tmp_path):
    from streamlit.testing.v1 import AppTest
    from scan_map_view import cached_scan_map

    data = np.arange(1, 1 + 7 * 3 * 4 * 5 * 6, dtype=float).reshape(1, 7, 3, 4, 5, 6)
    np.save(tmp_path / "multi_energy.npy", data)
    cached_scan_map.clear()
    with patch("scan_map_view.np.savez_compressed", wraps=np.savez_compressed) as exports, \
         patch("scan_map_view.detector_scan_map", wraps=detector_scan_map) as reads:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        app.radio(key="visualization").set_value("2D scan map").run()
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
        app.text_input(key="map_energies").set_value("-100, 0, 60, 60.01").run()
        assert not app.exception and not app.error
        assert app.metric[0].value == "3"
        assert len(app.get("plotly_chart")) == 3
        assert any("recorded bins" in message.value for message in app.info)
        axis = energy_loss_axis_mev(7, 5, 3)
        indices = np.array([np.argmin(abs(axis - energy)) for energy in (-100, 0, 60)])
        expected = np.stack([data[0, i].sum(axis=(-2, -1)) for i in indices])
        result = exports.call_args.kwargs
        np.testing.assert_array_equal(result["scan_maps"], expected)
        np.testing.assert_array_equal(result["selected_energies_mev"], axis[indices])
        np.testing.assert_array_equal(result["energy_lo_mev"], axis[indices])
        np.testing.assert_array_equal(result["energy_hi_mev"], axis[indices])
        np.testing.assert_array_equal(result["bin_count"], [1, 1, 1])
        metadata = json.loads(str(result["metadata_json"]))
        assert metadata["axes"] == ["energy_map", "probe_x", "probe_y"]
        assert metadata["maps"][-1]["requested"] == [[60, 60], [60.01, 60.01]]
        assert metadata["maps"][-1]["energy_indices"] == [int(indices[-1])]
        for chart in app.get("plotly_chart"):
            trace = json.loads(chart.proto.spec)["data"][0]
            assert trace["zmin"] == expected.min() and trace["zmax"] == expected.max()
        before = reads.call_count
        app.checkbox(key="map_shared_scale").uncheck().run()
        assert not app.exception and not app.error
        assert reads.call_count == before
        for chart in app.get("plotly_chart"):
            trace = json.loads(chart.proto.spec)["data"][0]
            assert "zmin" not in trace and "zmax" not in trace
        app.selectbox(key="map_ordering").select("Unshifted FFT").run()
        assert not app.exception and not app.error
        raw_indices = np.fft.fftshift(np.arange(7))[indices]
        metadata = json.loads(str(exports.call_args.kwargs["metadata_json"]))
        np.testing.assert_array_equal([m["energy_indices"][0] for m in metadata["maps"]], raw_indices)
        np.testing.assert_array_equal(exports.call_args.kwargs["scan_maps"],
                                      np.stack([data[0, i].sum(axis=(-2, -1)) for i in raw_indices]))
        before = reads.call_count
        app.text_input(key="map_energies").set_value("60, bad").run()
        assert not app.exception
        assert any("numeric map energies" in message.value for message in app.error)
        assert reads.call_count == before


def test_app_energy_range_sums_every_enclosed_bin(tmp_path):
    from streamlit.testing.v1 import AppTest
    from scan_map_view import cached_scan_map

    data = np.arange(1, 1 + 7 * 3 * 4 * 5 * 6, dtype=float).reshape(1, 7, 3, 4, 5, 6)
    np.save(tmp_path / "range_energy.npy", data)
    cached_scan_map.clear()
    with patch("scan_map_view.np.savez_compressed", wraps=np.savez_compressed) as exports:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        app.radio(key="visualization").set_value("2D scan map").run()
        next(w for w in app.text_input if w.label == "Data folder").set_value(str(tmp_path)).run()
        app.text_input(key="map_energies").set_value("0, -120--70").run()
        assert not app.exception and not app.error
        assert app.metric[0].value == "2"
        assert len(app.get("plotly_chart")) == 2
        axis = energy_loss_axis_mev(7, 5, 3)
        point_index = int(np.argmin(abs(axis - 0)))
        range_indices = np.flatnonzero((axis >= -120) & (axis <= -70))
        assert len(range_indices) > 1  # the fixture only exercises a real sum when >1 bin falls inside it
        expected = np.stack([
            data[0, point_index].sum(axis=(-2, -1)),
            data[0, range_indices].sum(axis=(0, -2, -1)),
        ])
        result = exports.call_args.kwargs
        np.testing.assert_array_equal(result["scan_maps"], expected)
        np.testing.assert_array_equal(result["bin_count"], [1, len(range_indices)])
        np.testing.assert_allclose(result["energy_lo_mev"], [axis[point_index], axis[range_indices].min()])
        np.testing.assert_allclose(result["energy_hi_mev"], [axis[point_index], axis[range_indices].max()])
        metadata = json.loads(str(result["metadata_json"]))
        assert metadata["maps"][-1]["requested"] == [[-120, -70]]
        assert metadata["maps"][-1]["energy_indices"] == sorted(int(i) for i in range_indices)
