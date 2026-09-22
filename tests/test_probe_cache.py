"""Neighboring Zarr probes must share work without changing numerical results."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr

from cache_layer import spectrum_tile_tasks, spectrum_from_zarr_tile
from eels_core import extract_spectrum, inspect_scan

ROOT = Path(__file__).resolve().parents[1]


def create_scan(path, data, chunks, version=3):
    array = zarr.open_array(str(path), mode='w', shape=data.shape, dtype=data.dtype,
                            chunks=chunks, zarr_format=version)
    array[:] = data
    return array


@pytest.mark.parametrize('version', [2, 3])
@pytest.mark.parametrize('chunks', [(2, 75, 3, 3, 7, 9), (1, 19, 3, 3, 4, 5)])
def test_tile_cache_matches_direct_extraction(tmp_path, version, chunks):
    data = np.random.default_rng(9).uniform(.1, 3, (2, 75, 5, 5, 7, 9))
    path = tmp_path / 'scan.zarr'
    create_scan(path, data, chunks, version)
    npy = tmp_path / 'scan.npy'
    np.save(npy, data)
    info, reference = inspect_scan(path), inspect_scan(npy)
    for x, y in [(0, 0), (1, 2), (3, 4), (4, 4)]:
        for normalize in [True, False]:
            result = spectrum_from_zarr_tile(info, 1, x, y, 2.5, -1, 1, normalize)
            expected = extract_spectrum(reference, dummy=1, probe_x=x, probe_y=y,
                                         radius=2.5, offset_px=-1, offset_py=1,
                                         normalize_3d=normalize)
            np.testing.assert_allclose(result, expected, rtol=1e-13)


def test_cache_reuses_neighbors_and_invalidates_detector_and_source(tmp_path):
    path = tmp_path / 'scan.zarray'
    data = np.random.default_rng(4).uniform(.1, 3, (1, 75, 5, 5, 7, 9))
    array = create_scan(path, data, (1, 75, 3, 3, 7, 9))
    info = inspect_scan(path)
    original = zarr.Array.__getitem__
    with patch.object(zarr.Array, '__getitem__', autospec=True, side_effect=original) as reads:
        a = spectrum_from_zarr_tile(info, 0, 0, 0, 2, 0, 0, True)
        assert reads.call_count == 1
        spectrum_from_zarr_tile(info, 0, 2, 2, 2, 0, 0, True)
        spectrum_from_zarr_tile(info, 0, 1, 1, 2, 0, 0, False)
        assert reads.call_count == 1
        spectrum_from_zarr_tile(info, 0, 3, 4, 2, 0, 0, True)
        assert reads.call_count == 2
        spectrum_from_zarr_tile(info, 0, 0, 0, 3, 0, 0, True)
        assert reads.call_count == 3
        array[:, 0] = data[:, 0] * 7
        before = reads.call_count
        b = spectrum_from_zarr_tile(inspect_scan(path), 0, 0, 0, 2, 0, 0, True)
        assert reads.call_count == before + 1
        assert not np.allclose(a, b)


def test_invalid_neighbor_does_not_block_valid_probe(tmp_path):
    data = np.ones((1, 40, 2, 2, 5, 5))
    data[:, :, 0, 1] = 0
    data[:, 0, 1, 0, 0, 0] = np.nan  # Outside the small circular detector.
    path = tmp_path / 'scan.zarr'
    create_scan(path, data, data.shape)
    info = inspect_scan(path)
    valid = spectrum_from_zarr_tile(info, 0, 0, 0, 1, 0, 0, True)
    np.testing.assert_allclose(valid, np.full(40, 5 / (40 * 25)))
    for x, y in [(0, 1), (1, 0)]:
        with pytest.raises(ValueError, match='Cannot normalize'):
            spectrum_from_zarr_tile(info, 0, x, y, 1, 0, 0, True)
    np.testing.assert_array_equal(spectrum_from_zarr_tile(info, 0, 1, 0, 1, 0, 0, False),
                                  np.full(40, 5))


def test_adding_neighbor_in_app_does_not_read_zarr_again(tmp_path):
    from streamlit.testing.v1 import AppTest
    data = np.random.default_rng(2).uniform(1, 3, (1, 9, 5, 5, 5, 6))
    path = tmp_path / 'scan.zarray'
    create_scan(path, data, (1, 9, 3, 3, 5, 6))
    original = zarr.Array.__getitem__
    with patch.object(zarr.Array, '__getitem__', autospec=True, side_effect=original) as reads:
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
        app.text_input(key='data_folder').set_value(str(path)).run()
        assert not app.exception and not app.error
        initial_reads = reads.call_count
        assert initial_reads > 0
        app.multiselect(key='probe_positions').set_value([(0, 0), (1, 1), (2, 2)]).run()
        assert not app.exception and not app.error
        assert app.metric[1].value == '3'
        assert reads.call_count == initial_reads
        app.multiselect(key='probe_positions').set_value([(0, 0), (1, 1), (3, 3)]).run()
        assert not app.exception and not app.error
        assert reads.call_count == initial_reads + 1


def test_prepare_all_makes_any_probe_available_without_more_reads(tmp_path):
    from streamlit.testing.v1 import AppTest
    path = tmp_path / 'scan.zarr'
    create_scan(path, np.ones((1, 9, 5, 5, 5, 6)), (1, 9, 3, 3, 5, 6))
    original = zarr.Array.__getitem__
    with patch.object(zarr.Array, '__getitem__', autospec=True, side_effect=original) as reads:
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
        app.text_input(key='data_folder').set_value(str(path)).run()
        assert not app.exception and not app.error
        before = reads.call_count
        next(b for b in app.button if b.label == 'Prepare all Zarr probe spectra').click().run()
        assert not app.exception and not app.error
        assert any('All Zarr probe spectra prepared' in s.value for s in app.success)
        assert reads.call_count == before + 3  # First of four tiles was already ready.
        before = reads.call_count
        app.multiselect(key='probe_positions').set_value([(0, 0), (4, 4), (2, 4), (4, 1)]).run()
        assert not app.exception and not app.error
        assert app.metric[1].value == '4'
        assert reads.call_count == before


def test_prepare_tasks_fit_cache_and_ignore_numpy(tmp_path):
    from dataclasses import replace
    path = tmp_path / 'scan.zarray'
    create_scan(path, np.ones((1, 9, 5, 5, 5, 6)), (1, 9, 3, 3, 5, 6))
    info = inspect_scan(path)
    assert len(spectrum_tile_tasks([info, replace(info, chunks=None)])) == 4
    huge = replace(info, shape=(1, 9, 3000, 3000, 5, 6))
    assert spectrum_tile_tasks([huge]) == []
