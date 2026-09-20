"""Preserve plot viewports and add modifier-key scaling without Python reruns."""
from pathlib import Path

import streamlit.components.v2 as components

# Read once at import time rather than on every Streamlit rerun.
_AXIS_SCALING_JS = Path(__file__).with_name("axis_scaling.js").read_text()


def register_axis_scaling():
    return components.component("eels_axis_scaling", js=_AXIS_SCALING_JS)
