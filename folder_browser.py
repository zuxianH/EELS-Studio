"""Local folder navigation for the Streamlit sidebar."""
from pathlib import Path
import os

import streamlit as st

from native_folder_picker import choose_directory
from scan_sources import resolve_data_directory


def _navigate(path):
    st.session_state["folder_browser_path"] = str(path)


def _initial_folder():
    try:
        path = resolve_data_directory(st.session_state["data_folder"])
    except (OSError, RuntimeError, ValueError):
        path = Path.home()
    return path


def _open_browser():
    _navigate(_initial_folder())
    st.session_state["folder_browser_open"] = True


def _open_native_browser():
    st.session_state.pop("folder_browser_error", None)
    _close_browser()
    try:
        selected = choose_directory(_initial_folder())
    except (OSError, RuntimeError):
        _open_browser()
        st.session_state["folder_browser_error"] = (
            "Could not open your system's folder window. Choose a folder below or enter its path.")
        return
    if selected is not None:
        _use_folder(selected)


def _close_browser():
    st.session_state["folder_browser_open"] = False


def _use_folder(path):
    # Callbacks run before the text input is created on the next rerun.
    try:
        with os.scandir(path):
            pass
    except OSError as exc:
        st.session_state["folder_browser_error"] = str(exc)
        return
    st.session_state["data_folder"] = str(path)
    _close_browser()


def render_folder_input(default):
    """Open a system folder chooser, with in-app navigation as a fallback."""
    st.session_state.setdefault("data_folder", str(default))
    folder = st.text_input("Data folder", key="data_folder",
                           help="Local folder containing NumPy/Zarr scans, or a Zarr array directory.")
    st.button("Browse folders", on_click=_open_native_browser, width="stretch",
              help="Open the folder-selection window on the computer running this app.")
    error = st.session_state.pop("folder_browser_error", None)
    if error:
        st.warning(error)
    if not st.session_state.get("folder_browser_open", False):
        return folder

    with st.container(border=True):
        st.caption("Browse local folders")
        path = Path(st.session_state["folder_browser_path"])
        st.code(str(path), language=None, wrap_lines=True)
        up, home = st.columns(2)
        up.button("Up", on_click=_navigate, args=(path.parent,),
                  disabled=path == path.parent, width="stretch")
        home.button("Home", on_click=_navigate, args=(Path.home(),),
                    width="stretch")
        readable = True
        try:
            children = sorted((child for child in path.iterdir() if child.is_dir()),
                              key=lambda child: (child.name.casefold(), child.name))
        except OSError as exc:
            st.warning(f"Cannot browse this folder: {exc}")
            children = []
            readable = False
        if children:
            child = st.selectbox("Subfolders", children, format_func=lambda p: p.name,
                                 key=f"folder_browser_child:{path}")
            st.button("Open subfolder", on_click=_navigate, args=(child,),
                      width="stretch")
        elif readable:
            st.caption("No subfolders here.")
        st.button("Use this folder", type="primary", on_click=_use_folder,
                  args=(path,), disabled=not readable, width="stretch")
        st.button("Cancel", on_click=_close_browser, width="stretch")
    return folder
