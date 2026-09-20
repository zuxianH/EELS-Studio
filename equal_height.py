"""Match two card panels' rendered heights without a Python rerun."""
from pathlib import Path

import streamlit.components.v2 as components

# Read once at import time rather than on every Streamlit rerun.
_EQUAL_HEIGHT_JS = Path(__file__).with_name("equal_height.js").read_text()


def register_equal_height():
    return components.component("eels_equal_height", js=_EQUAL_HEIGHT_JS)
