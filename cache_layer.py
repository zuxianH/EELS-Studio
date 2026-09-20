"""Streamlit caches shared across views.

Kept separate from eels_core.py, which stays independent of the UI. Sharing
one cached wrapper (rather than each view defining its own) also means a
diffraction plane requested from two different tabs is only read once.
"""
import streamlit as st

from eels_core import diffraction_pattern

PNG_DPI_OPTIONS = [150, 300, 600, 1200]


@st.cache_data(show_spinner=False, max_entries=32)
def cached_diffraction_pattern(info, energy_index, dummy, probe_x, probe_y):
    return diffraction_pattern(info, energy_index, dummy=dummy, probe_x=probe_x, probe_y=probe_y)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_background_fit(energy, intensity, config):
    # Cache only small extracted 1D arrays; never open a simulation file here.
    from background_core import fit_background
    return fit_background(energy, intensity, config)
