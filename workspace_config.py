"""Versioned JSON workspace settings; never serialize arrays or executable objects."""
from dataclasses import asdict
import json
import math
import re

import streamlit as st

from background_core import BACKGROUND_MODELS, BackgroundConfig

FORMAT = "eels-studio-workspace"
VERSION = 1
MAX_BYTES = 2 * 1024 * 1024
ORDERING = ["FFT-shifted (notebook default)", "Unshifted FFT"]
STYLES = ["Solid", "Dashed", "Dotted", "Dash-dot"]
CHOICES = {
    "visualization": ["Spectra & detector", "2D scan map"],
    "spectrum_ordering": ORDERING, "map_ordering": ORDERING,
    "intensity_display": ["log10", "Linear", "Intensity × E²"],
    "angle_direction": ["Horizontal (py)", "Vertical (px)"],
    "angle_color_scale": ["log10", "Linear"],
    "bg_display": ["Linear", "log10"],
    "bg_method": ["arPLS", "SNIP", *BACKGROUND_MODELS],
    "bg_domain": ["Selected energy interval", "Full recorded spectrum"],
    "bg_preview_view": ["fit", "full"],
    **{k: [150, 300, 600, 1200] for k in ("png_dpi_spectrum", "png_dpi_angle", "png_dpi_scan_map")},
}
BOOLS = set("normalize_probe broaden_spectrum pattern_log show_hover_details show_detector_preview "
            "angle_preview_log angle_manual_color map_shared_scale map_manual_scale map_broaden".split())
STRINGS = set("data_folder style_editor angle_source map_scan map_energies bg_preview_curve".split())
# Numeric fields: (integer, minimum, maximum). None means unbounded.
NUMBERS = {
    "detector_radius": (False, .1, None), "offset_px": (True, None, None), "offset_py": (True, None, None),
    "spectrum_timestep": (False, 1e-6, None), "spectrum_stride": (True, 1, None),
    "spectrum_sigma": (False, 0, None), "preview_index": (True, 0, None),
    "preview_energy": (False, None, None), "spectrum_x_min": (False, None, None),
    "spectrum_x_max": (False, None, None), "angle_preview_energy": (False, None, None),
    "angle_energy_min": (False, None, None), "angle_energy_max": (False, None, None),
    "angle_color_min": (False, None, None), "angle_color_max": (False, None, None),
    "map_color_min": (False, None, None), "map_color_max": (False, None, None),
    "map_columns": (True, 1, 12), "map_radius": (False, .1, None),
    "map_offset_px": (True, None, None), "map_offset_py": (True, None, None),
    "map_timestep": (False, 1e-6, None), "map_stride": (True, 1, None), "map_sigma": (False, 0, None),
    "bg_n_segments": (True, 2, 4), "bg_energy_factor": (False, 1e-6, None),
    "bg_min": (False, None, None), "bg_max": (False, None, None),
    "bg_lambda_slider": (False, 2, 10), "bg_lambda_number": (False, 2, 10),
    "bg_window": (False, 1e-6, None), "bg_tolerance": (False, 1e-12, .999),
    "bg_iterations": (True, 1, None),
}
for model, spec in BACKGROUND_MODELS.items():
    for parameter in spec.params:
        for field in ("start", "lower", "upper"):
            NUMBERS[f"bg_m{field}_{model}_{parameter}"] = (False, None, None)
for i in range(4):
    for side in ("lo", "hi"):
        NUMBERS[f"bg_seg_{side}_{i}"] = (False, None, None)


def is_setting(key):
    return (key in CHOICES or key in BOOLS or key in STRINGS or key in NUMBERS
            or key in {"curve_styles", "angle_saved_rois", "probe_positions"}
            or key.startswith(("files:", "label:", "line_color:", "line_style:", "line_width:", "angle_roi:")))


def _number(value, spec, name):
    integer, low, high = spec
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (integer and type(value) is not int)
            or (low is not None and value < low) or (high is not None and value > high)):
        raise ValueError(f"Invalid numeric setting: {name}")
    return int(value) if integer else float(value)


def _color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Invalid line color in config.")
    return value


def _positions(value, count, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Invalid {name} in config.")
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != count:
            raise ValueError(f"Invalid {name} in config.")
        result.append(tuple(_number(v, (True, 0, None), name) for v in row))
    return result


def _setting(key, value):
    if key in CHOICES:
        if value not in CHOICES[key] or isinstance(value, bool):
            raise ValueError(f"Unknown option for {key}.")
    elif key in BOOLS:
        if type(value) is not bool:
            raise ValueError(f"Expected true/false for {key}.")
    elif key in STRINGS or key.startswith("label:"):
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"Expected text for {key}.")
    elif key in NUMBERS:
        return _number(value, NUMBERS[key], key)
    elif key.startswith("files:"):
        if not isinstance(value, list) or any(not isinstance(p, str) or "\0" in p for p in value):
            raise ValueError("Invalid selected scan paths.")
    elif key == "probe_positions":
        return _positions(value, 2, key)
    elif key.startswith("line_color:"):
        return _color(value)
    elif key.startswith("line_style:"):
        if value not in STYLES:
            raise ValueError("Invalid line style in config.")
    elif key.startswith("line_width:"):
        return _number(value, (False, .5, 8), key)
    elif key.startswith("angle_roi:"):
        return _number(value, (True, 0, None), key)
    elif key == "curve_styles":
        if not isinstance(value, dict):
            raise ValueError("Invalid curve styles.")
        result = {}
        for identity, style in value.items():
            if not isinstance(style, dict) or set(style) != {"color", "line_style", "width"}:
                raise ValueError("Invalid curve style.")
            result[identity] = {"color": _color(style["color"]),
                                "line_style": _setting("line_style:" + identity, style["line_style"]),
                                "width": _number(style["width"], (False, .5, 8), "line width")}
        return result
    elif key == "angle_saved_rois":
        if not isinstance(value, dict):
            raise ValueError("Invalid saved map rectangles.")
        return {k: _positions([v], 4, "map rectangle")[0] for k, v in value.items()}
    return value


def _background(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"config", "signal"}:
        raise ValueError("Invalid applied background settings.")
    if value["signal"] not in ("Input", "Corrected") or not isinstance(value["config"], dict):
        raise ValueError("Invalid applied background settings.")
    raw = value["config"]
    defaults = asdict(BackgroundConfig())
    if set(raw) != set(defaults) or raw["method"] not in CHOICES["bg_method"] or raw["domain"] not in ("selected", "full"):
        raise ValueError("Invalid background configuration.")
    result = dict(raw)
    for key in ("energy_min", "energy_max", "log10_lambda", "tolerance", "half_window_mev", "energy_factor", "max_iterations"):
        if raw[key] is None and key in ("energy_min", "energy_max"):
            continue
        result[key] = _number(raw[key], (key == "max_iterations", None, None), key)
    if not (2 <= result["log10_lambda"] <= 10 and 0 < result["tolerance"] < 1
            and result["max_iterations"] >= 1 and result["half_window_mev"] > 0 and result["energy_factor"] > 0):
        raise ValueError("Invalid background fitting parameters.")
    segments = raw["segments"]
    if not isinstance(segments, list) or len(segments) > 4:
        raise ValueError("Invalid background segments.")
    result["segments"] = tuple(tuple(_number(v, (False, None, None), "segment") for v in row)
                               for row in segments if isinstance(row, list) and len(row) == 2)
    if len(result["segments"]) != len(segments):
        raise ValueError("Invalid background segments.")
    for key in ("model_start", "model_lower", "model_upper"):
        if not isinstance(raw[key], list) or len(raw[key]) > 6:
            raise ValueError("Invalid background model parameters.")
        result[key] = tuple(_number(v, (False, None, None), key) for v in raw[key])
    return {"config": result, "signal": value["signal"]}


def decode_config(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Config files must be smaller than 2 MB.")
    try:
        document = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("This is not a valid JSON config file.") from exc
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ValueError("Choose an EELS Studio workspace config.")
    if type(document.get("version")) is not int or document["version"] != VERSION:
        raise ValueError("This config version is not supported.")
    raw = document.get("settings")
    if not isinstance(raw, dict) or any(not is_setting(k) for k in raw):
        raise ValueError("The config contains unknown settings.")
    return {"settings": {k: _setting(k, v) for k, v in raw.items()},
            "background": _background(document.get("background"))}


def encode_config(state):
    settings = {k: state[k] for k in state if is_setting(k)}
    background = state.get("background_state")
    applied = ({"config": asdict(background.applied_config), "signal": background.signal}
               if background is not None and background.applied_config is not None else None)
    # While viewing 2D maps the spectrum background may still be waiting to restore.
    applied = state.get("_workspace_background", applied)
    return json.dumps({"format": FORMAT, "version": VERSION, "settings": settings,
                       "background": applied}, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def apply_config(state, decoded):
    """Replace settings only after the complete document passes validation."""
    for key in list(state):
        if is_setting(key) or key.startswith(("bg_", "folder_browser_")) or key in {
                "background_state", "detector_click", "detector_preview_context", "detector_click_error",
                "angle_rectangle", "angle_rectangle_context", "angle_rectangle_error", "_workspace_background"}:
            del state[key]
    state.update(decoded["settings"])
    if decoded["background"]:
        state["_workspace_background"] = decoded["background"]
    state["workspace_revision"] = state.get("workspace_revision", 0) + 1


def _load_upload():
    uploaded = st.session_state.get("workspace_upload")
    if uploaded is None:
        st.session_state["workspace_message"] = ("warning", "Choose a config file first.")
        return
    try:
        decoded = decode_config(uploaded.getvalue())
        apply_config(st.session_state, decoded)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        st.session_state["workspace_message"] = ("error", f"Could not load config: {exc}")
    else:
        st.session_state["workspace_message"] = ("success", "Workspace config loaded.")


def render_config_controls():
    # Detach hidden widgets from Streamlit's cleanup so switching views retains
    # settings for both spectra and maps, including styles for unselected curves.
    for key in list(st.session_state):
        if is_setting(key):
            st.session_state[key] = st.session_state[key]
    with st.sidebar.expander("Workspace config", expanded=True):
        st.caption("Save your files, parameters and appearance to continue later.")
        download_slot = st.empty()
        st.file_uploader("Config file", type=["json"], key="workspace_upload", max_upload_size=2)
        st.button("Load config", on_click=_load_upload, disabled=st.session_state.get("workspace_upload") is None,
                  width="stretch")
        message = st.session_state.pop("workspace_message", None)
        if message:
            getattr(st, message[0])(message[1])
    return download_slot


def render_config_download(slot):
    try:
        data = encode_config(st.session_state)
    except (ValueError, TypeError) as exc:
        slot.warning(f"Cannot save these settings: {exc}")
        return
    slot.download_button("Save config", data, "eels-studio-config.json", "application/json",
                         key="workspace_download", on_click="ignore", width="stretch")


def restore_background(curves, state):
    pending = st.session_state.pop("_workspace_background", None)
    if not pending:
        return
    try:
        with st.spinner("Restoring background correction…"):
            failures = state.apply(curves, BackgroundConfig(**pending["config"]))
        if failures:
            st.warning("The saved background fit could not be restored for these scans. Review the Background tab and apply it again.")
            return
        st.session_state["bg_next_signal"] = pending["signal"]
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        st.warning(f"Could not restore the saved background fit: {exc}")
