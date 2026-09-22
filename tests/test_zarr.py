"""Zarr/NumPy parity, bounded selections, cache freshness, and UI loading."""
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr

from eels_core import (detector_scan_map, diffraction_pattern, export_npz,
                       extract_angle_resolved, extract_spectrum, inspect_scan)
from scan_sources import ZarrScanReader, discover_scans, scan_revision

ROOT = Path(__file__).resolve().parents[1]


def save_zarr(path, data, chunks, version=3, **kwargs):
    array = zarr.open_array(str(path), mode="w", shape=data.shape, dtype=data.dtype,
                            chunks=chunks, zarr_format=version, **kwargs)
    array[:] = data
    return array


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("shape,chunks", [
    ((75, 7, 9), (19, 4, 5)),
    ((2, 75, 5, 3, 7, 9), (1, 19, 3, 2, 4, 5)),
])
@pytest.mark.parametrize("normalize", [False, True])
def test_zarr_matches_npy_for_all_views(tmp_path, version, shape, chunks, normalize):
    data = np.random.default_rng(3).uniform(.1, 5, shape).astype(np.float32)
    npy, store = tmp_path / "scan.npy", tmp_path / "scan.zarray"
    np.save(npy, data)
    save_zarr(store, data, chunks, version)
    expected, actual = inspect_scan(npy), inspect_scan(store)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    args = dict(dummy=1, probe_x=4, probe_y=2) if len(shape) == 6 else {}
    for info in [actual]:
        np.testing.assert_allclose(
            extract_spectrum(info, radius=2.5, offset_px=-1, normalize_3d=normalize, **args),
            extract_spectrum(expected, radius=2.5, offset_px=-1, normalize_3d=normalize, **args),
            rtol=1e-13)
        np.testing.assert_array_equal(diffraction_pattern(info, 38, **args),
                                      diffraction_pattern(expected, 38, **args))
        for axis in ["px", "py"]:
            pixels, values = extract_angle_resolved(info, (1, 6, 2, 7), retain_axis=axis,
                                                    normalize_3d=normalize, **args)
            e_pixels, e_values = extract_angle_resolved(expected, (1, 6, 2, 7), retain_axis=axis,
                                                        normalize_3d=normalize, **args)
            np.testing.assert_array_equal(pixels, e_pixels)
            np.testing.assert_allclose(values, e_values, rtol=1e-13)
        if len(shape) == 6:
            np.testing.assert_array_equal(detector_scan_map(info, 38, dummy=1, radius=2.5),
                                          detector_scan_map(expected, 38, dummy=1, radius=2.5))


@pytest.mark.parametrize("version", [2, 3])
def test_reader_selections_and_fill_values(tmp_path, version):
    path = tmp_path / "scan.zarr"
    data = np.arange(7 * 5 * 9, dtype=np.int16).reshape(7, 5, 9)
    save_zarr(path, data, (3, 2, 4), version, **({"order": "F"} if version == 2 else {}))
    with ZarrScanReader(path) as reader:
        for key in [(slice(1, 6), 3, slice(2, 8)), (6, 4, 8),
                    (slice(None), slice(None), slice(None)),
                    (slice(-3, None), 0, slice(-5, -1))]:
            np.testing.assert_array_equal(reader.read(key), data[key])
        for key in [(slice(None, None, 2), 0, 0), (7, 0, 0),
                    (slice(2, 2), 0, 0), (0, 0)]:
            with pytest.raises(ValueError):
                reader.read(key)
    empty = tmp_path / "empty.zarr"
    zarr.open_array(str(empty), mode="w", shape=(3, 4, 5), chunks=(2, 3, 4),
                    dtype="f8", fill_value=2, zarr_format=version)
    np.testing.assert_array_equal(diffraction_pattern(inspect_scan(empty), 1), np.full((4, 5), 2))


def test_large_chunk_is_reused_and_released(tmp_path, monkeypatch):
    data = np.ones((1, 75, 5, 2, 7, 9))
    path = tmp_path / "scan.zarray"
    save_zarr(path, data, (1, 75, 3, 2, 7, 9))
    # Force even these small fixtures to exercise the one-oversized-chunk policy.
    monkeypatch.setattr(ZarrScanReader, "cache_bytes", 1)
    original = zarr.Array.__getitem__
    with patch.object(zarr.Array, "__getitem__", autospec=True, side_effect=original) as reads:
        info = inspect_scan(path)
        assert reads.call_count == 0  # Inspection must not read any data.
        extract_spectrum(info, probe_x=1)
        assert reads.call_count == 1  # Three energy blocks reuse one decoded chunk.
        reads.reset_mock()
        detector_scan_map(info, 2)
        assert reads.call_count == 2  # One decode per spatial chunk, not per probe.
    with ZarrScanReader(path) as reader:
        reader.read((0, 1, 1, 0, slice(None), slice(None)))
        assert reader._cached_bytes > reader.cache_bytes
    assert reader._cached_bytes == 0 and not reader._cache


@pytest.mark.parametrize("version", [2, 3])
def test_revision_detects_nested_chunk_changes(tmp_path, version):
    path = tmp_path / "scan.zarr"
    array = save_zarr(path, np.ones((7, 5, 9)), (3, 2, 4), version)
    before = inspect_scan(path)
    root_stat = path.stat()
    array[0, :, :] = 3
    os.utime(path, ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns))
    assert scan_revision(path)[2] != before.revision
    with pytest.raises(ValueError, match="File changed"):
        diffraction_pattern(before, 0)
    np.testing.assert_array_equal(diffraction_pattern(inspect_scan(path), 0), np.full((5, 9), 3))
    revision = scan_revision(path)
    chunk = next(p for p in path.rglob('*') if p.is_file()
                 and p.name not in ('zarr.json', '.zarray', '.zattrs'))
    chunk.unlink()
    assert scan_revision(path) != revision


@pytest.mark.parametrize("shape,dtype", [((3, 4), 'f8'), ((0, 4, 5), 'f8'),
                                         ((3, 4, 5), 'c16'), ((3, 4, 5), 'bool')])
def test_reject_unsupported_zarr_arrays(tmp_path, shape, dtype):
    path = tmp_path / "bad.zarr"
    zarr.open_array(str(path), mode="w", shape=shape, dtype=dtype)
    with pytest.raises(ValueError, match="Expected"):
        inspect_scan(path)


def test_discovery_and_rejection_of_groups_and_broken_stores(tmp_path):
    npy, store, bare = tmp_path / 'a.npy', tmp_path / 'b.zarray', tmp_path / 'no_extension'
    data = np.ones((3, 4, 5))
    np.save(npy, data)
    save_zarr(store, data, (2, 3, 4))
    save_zarr(bare, data, (2, 3, 4), 2)
    (tmp_path / 'unrelated').mkdir()
    assert discover_scans(tmp_path) == [npy, store, bare]
    assert discover_scans(store) == [store]
    for name in ['group.zarr', 'broken.zarr']:
        path = tmp_path / name
        if name.startswith('group'):
            zarr.open_group(str(path), mode='w')
        else:
            path.mkdir()
            (path / 'zarr.json').write_text('{invalid')
        with pytest.raises(ValueError, match="Cannot open Zarr array"):
            inspect_scan(path)


def test_missing_zarr_has_install_message_and_npy_still_works(tmp_path):
    store, npy = tmp_path / 'scan.zarr', tmp_path / 'scan.npy'
    save_zarr(store, np.ones((3, 4, 5)), (2, 3, 4))
    np.save(npy, np.ones((3, 4, 5)))
    with patch.dict('sys.modules', {'zarr': None}):
        with pytest.raises(ValueError, match='pip install'):
            inspect_scan(store)
        assert inspect_scan(npy).shape == (3, 4, 5)


def test_app_mixed_sources_direct_store_and_refresh(tmp_path):
    from streamlit.testing.v1 import AppTest
    data = np.random.default_rng(5).uniform(1, 3, (1, 9, 2, 2, 5, 6))
    store = tmp_path / 'b.zarray'
    array = save_zarr(store, data, (1, 4, 1, 2, 3, 4))
    np.save(tmp_path / 'a.npy', data)
    with patch('eels_core.export_npz', wraps=export_npz) as exported:
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
        app.text_input(key='data_folder').set_value(str(tmp_path)).run()
        assert not app.exception and not app.error
        curves, _ = exported.call_args.args
        assert len(curves) == 2
        np.testing.assert_allclose(curves[0]['intensity'], curves[1]['intensity'])
        app.text_input(key='data_folder').set_value(str(store)).run()
        assert not app.exception and not app.error
        before = exported.call_args.args[0][0]['intensity'].copy()
        array[:, 0] = data[:, 0] * 5
        app.run()
        assert not app.exception and not app.error
        after = exported.call_args.args[0][0]['intensity']
        assert not np.allclose(before, after)
        np.testing.assert_allclose(after, extract_spectrum(inspect_scan(store)))
        app.radio(key='visualization').set_value('2D scan map').run()
        assert not app.exception and not app.error
        assert app.metric[1].value == '4'
