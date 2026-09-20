"""Relay native Plotly heatmap clicks to a Streamlit callback without remote assets."""
from pathlib import Path

import streamlit.components.v2 as components

# Read once at import time rather than on every Streamlit rerun.
_DETECTOR_CLICK_JS = Path(__file__).with_name("detector_click.js").read_text()


def register_detector_click_bridge():
    # Register in the active app runtime (also supports separate AppTest runs).
    return components.component("eels_detector_click", js=_DETECTOR_CLICK_JS)
