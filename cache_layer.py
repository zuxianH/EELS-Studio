"""Streamlit caches shared across views.

Kept separate from eels_core.py, which stays independent of the UI. Sharing
one cached wrapper (rather than each view defining its own) also means a
diffraction plane requested from two different tabs is only read once.
"""
import numpy as np
import streamlit as st

from eels_core import diffraction_pattern, extract_spectrum_tile

PNG_DPI_OPTIONS = [150, 300, 600, 1200]


@st.cache_data(show_spinner=False, max_entries=32)
def cached_diffraction_pattern(info, energy_index, dummy, probe_x, probe_y):
    return diffraction_pattern(info, energy_index, dummy=dummy, probe_x=probe_x, probe_y=probe_y)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_background_fit(energy, intensity, config):
    # Cache only small extracted 1D arrays; never open a simulation file here.
    from background_core import fit_background
    return fit_background(energy, intensity, config)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_spectrum_tile(info, dummy, tile_x, tile_y, radius, offset_px, offset_py):
    # Cache only reduced spectra, never the large decoded source chunks.
    return extract_spectrum_tile(info, dummy=dummy, tile_x=tile_x, tile_y=tile_y,
                                 radius=radius, offset_px=offset_px, offset_py=offset_py)


def spectrum_from_zarr_tile(info, dummy, probe_x, probe_y, radius, offset_px, offset_py, normalize):
    """Adjacent probes reuse the same small detector-integrated tile."""
    for value, length in zip((dummy, probe_x, probe_y),
                             (info.shape[0], info.shape[2], info.shape[3])):
        if not isinstance(value, (int, np.integer)) or not 0 <= value < length:
            raise ValueError("Probe or dummy index is outside the scan")
    cx, cy = info.chunks[2:4]
    spectra, totals, finite = cached_spectrum_tile(
        info, dummy, probe_x // cx, probe_y // cy, radius, offset_px, offset_py)
    x, y = probe_x % cx, probe_y % cy
    result = spectra[:, x, y].copy()
    if normalize:
        total = totals[x, y]
        if not finite[x, y] or not np.isfinite(total) or total == 0:
            raise ValueError("Cannot normalize probe data with zero, NaN, infinite, or overflowed total intensity")
        result /= total
    if not np.isfinite(result).all():
        raise ValueError("Selected detector data contains NaN, infinite, or overflowed intensities")
    return result


def spectrum_tile_tasks(infos):
    """List spatial tiles for the selected 6D Zarr scans."""
    grids = [(info, (info.shape[2] + info.chunks[2] - 1) // info.chunks[2],
               (info.shape[3] + info.chunks[3] - 1) // info.chunks[3])
             for info in infos if info.chunks is not None and len(info.shape) == 6]
    if sum(nx * ny for _, nx, ny in grids) > 256:
        return []  # Avoid evicting early tiles before preparation has finished.
    return [(info, x, y) for info, nx, ny in grids for x in range(nx) for y in range(ny)]
