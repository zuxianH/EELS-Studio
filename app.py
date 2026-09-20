"""Run with: .venv/bin/python -m streamlit run app.py"""
from pathlib import Path
import io
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# Plotly's optional-import lookup reads sys.modules without waiting for an import
# in another Streamlit thread. Finish pandas initialization before building plots.
import pandas  # noqa: F401
import plotly.graph_objects as go
import streamlit as st

from eels_core import (
    curve_identity_key, detector_center, display_intensity,
    detector_offsets_from_click, energy_loss_axis_mev, export_csv, export_npz,
    extract_spectrum, gaussian_broaden_spectrum, inspect_scan, nearest_energy_index,
)
from cache_layer import cached_diffraction_pattern, PNG_DPI_OPTIONS
from scan_map_view import render_scan_map
from detector_click import register_detector_click_bridge
from angle_resolved import render_angle_resolved
from axis_scaling import register_axis_scaling
from equal_height import register_equal_height
from background_core import BackgroundState, config_caption, corrected_display, input_fingerprints
from background_exports import background_csv, background_npz
from background_view import render_background

ROOT = Path(__file__).resolve().parent
detector_click_bridge = register_detector_click_bridge()
axis_scaling = register_axis_scaling()
equal_height = register_equal_height()
COLORS = ["#137c78", "#dd7848", "#626cc6", "#c24c79", "#7a9845", "#428fbd"]
LINE_STYLES = {
    "Solid": ("solid", "-"),
    "Dashed": ("dash", "--"),
    "Dotted": ("dot", ":"),
    "Dash-dot": ("dashdot", "-."),
}

# Color for the Detector/Spectrum card panels, matching the app's teal theme.
PANEL_BORDER = "#dce5ec"

st.set_page_config(page_title="EELS Studio", page_icon="🔬", layout="wide")
st.markdown(f"""<style>
.block-container {{padding-top: 3.5rem; padding-bottom: 2rem;}}
h1 {{letter-spacing: -0.045em;}}
[data-testid="stMetric"] {{background: white; border: 1px solid {PANEL_BORDER};
  border-radius: 12px; padding: 14px 18px;}}
[data-testid="stSidebar"] {{border-right: 1px solid {PANEL_BORDER};}}
/* Keep the interface fully visible while an input change reruns the app. */
[data-testid="stElementContainer"], [data-testid="stExpanderDetails"] {{
  opacity: 1 !important; transition: none !important;
}}

/* Detector / Spectrum card panels */
.st-key-detector_panel, .st-key-spectrum_panel {{
  background: white; border: 1px solid {PANEL_BORDER}; border-radius: 14px;
  padding: 18px 20px; box-shadow: 0 2px 8px rgba(20,40,80,0.04);
}}
.eels-panel-title {{font-size: 19px; font-weight: 600; color: #172b3a; margin: 0 0 8px 0; text-align: center;}}

/* Compact Plotly modebar */
.modebar-container {{background: white !important; border: 1px solid {PANEL_BORDER};
  border-radius: 9px; padding: 2px;}}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=128)
def cached_spectrum(info, sample, probe_x, probe_y, radius, offset_px, offset_py, normalize):
    return extract_spectrum(info, sample=sample, probe_x=probe_x, probe_y=probe_y,
                            radius=radius, offset_px=offset_px, offset_py=offset_py,
                            normalize_3d=normalize)


@st.cache_data(show_spinner=False, max_entries=256)
def _cached_inspect(path_str, mtime_ns, size):
    return inspect_scan(path_str)


def inspect_scan_cached(path):
    """Skip re-parsing a scan's header when its mtime/size haven't changed."""
    stat = path.stat()
    return _cached_inspect(str(path), stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_export_csv(curves):
    return export_csv(curves)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_export_npz(curves, settings):
    return export_npz(curves, settings)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_notebook_npy(curves):
    buffer = io.BytesIO()
    np.save(buffer, np.stack([np.column_stack((c["energy"], c["intensity"])) for c in curves]))
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=8)
def figure_downloads(curves, mode, x_limits, normalize, dpi=300, corrected=False):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for curve in curves:
        style = curve["style"]
        ax.plot(curve["energy"], (corrected_display if corrected else display_intensity)(curve["intensity"], mode),
                label=curve["label"], color=style["color"], linewidth=style["width"],
                linestyle=LINE_STYLES[style["line_style"]][1])
    ax.set_xlabel("Energy loss (meV)")
    ax.set_ylabel(intensity_label(mode, normalize, corrected))
    ax.set_xlim(*x_limits)
    ax.grid(alpha=0.16)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    result = {}
    for extension in ("svg", "png"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=dpi, bbox_inches="tight")
        result[extension] = buffer.getvalue()
    plt.close(fig)
    return result


def intensity_label(mode, normalize, corrected=False):
    base = "Normalized detector intensity" if normalize else "Detector intensity"
    if corrected:
        base = "Background-subtracted " + base.lower()
    return f"log10({base.lower()})" if mode == "log10" else base


def suggested_label(path):
    match = re.search(r"(?:^|_)T(\d+(?:\.\d+)?)K(?:_|$)", Path(path).stem)
    return f"{match[1]} K" if match else Path(path).stem


def move_detector_from_click():
    # Component callbacks run before widgets, so updating the sidebar is safe.
    clicked = st.session_state.get("detector_click", {}).get("clicked")
    context = st.session_state.get("detector_preview_context")
    if not clicked or not context or clicked.get("preview_id") != context["preview_id"]:
        return
    try:
        dx, dy = detector_offsets_from_click(context["shape"], clicked["px"], clicked["py"],
                                              context["selected_shapes"])
        st.session_state["offset_px"] = dx
        st.session_state["offset_py"] = dy
    except (ValueError, KeyError, TypeError) as exc:
        st.session_state["detector_click_error"] = str(exc)


def reset_detector_center():
    st.session_state["offset_px"] = 0
    st.session_state["offset_py"] = 0


st.caption("STEM · SPECTROSCOPY WORKSPACE")
st.title("EELS Studio")
st.write("Choose your scans, position the detector, and explore spectra or a 2D probe map.")
view = st.radio("Visualization", ["Spectra & detector", "2D scan map"], horizontal=True, key="visualization")

with st.sidebar:
    st.header("Your scans")
    folder = st.text_input("Data folder", value=str(ROOT), help="Local folder on the computer running this app.")
    st.button("Refresh files", use_container_width=True, on_click=_cached_inspect.clear,
             help="Force every file to be re-inspected, in case one changed without its size or modified time changing.")
    try:
        directory = Path(folder).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("Enter an existing folder")
        available = sorted(directory.glob("*.npy"))
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    if not available:
        st.info("No .npy files in this folder. Enter a folder containing your scans.")
        st.stop()
    # Inspection reads only headers; arrays with unsupported dimensions are excluded.
    # Cached per (path, mtime, size), so an unchanged file's header is parsed
    # once rather than on every rerun.
    scans, rejected = {}, []
    for path in available:
        try:
            scans[str(path)] = inspect_scan_cached(path)
        except (OSError, ValueError, EOFError) as exc:
            rejected.append(f"{path.name}: {exc}")
    if rejected:
        with st.expander(f"{len(rejected)} unsupported file(s)"):
            for reason in rejected:
                st.caption(reason)
    options = list(scans)
    if not options:
        st.info("No supported 3D or 6D scan arrays found.")
        st.stop()
    selected = st.multiselect("Files to compare", options, default=options[:2],
                             format_func=lambda p: Path(p).name, key=f"files:{directory}")
    if not selected:
        st.info("Select at least one scan to get started.")
        st.stop()
    infos = [scans[p] for p in selected]
    with st.expander("Curve labels", expanded=False):
        labels = [st.text_input(Path(p).name, value=suggested_label(p), key=f"label:{p}").strip()
                  for p in selected]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        st.warning("Give each selected file a nonempty, unique curve label.")
        st.stop()

if view == "2D scan map":
    render_scan_map(infos, labels)
    st.stop()

with st.sidebar:
    st.divider()
    st.header("Extraction")
    st.caption("Detector radius and center (px, py, radius) are now set directly below the "
               "detector image in the Spectra tab.")
    min_px = min(i.shape[-2] for i in infos)
    min_py = min(i.shape[-1] for i in infos)
    for key, length in (("offset_px", min_px), ("offset_py", min_py)):
        if key in st.session_state:
            clamped = max(-(length // 2), min((length - 1) // 2, st.session_state[key]))
            if clamped != st.session_state[key]:
                st.session_state[key] = clamped
    # These reflect the editable px/py/radius fields below the detector image: Streamlit
    # already applies any pending edit to session_state before this script runs, so reading
    # it here (ahead of those widgets' own call site) still sees this render's live value.
    radius = st.session_state.get("detector_radius", 20.0)
    offset_px = st.session_state.get("offset_px", 0)
    offset_py = st.session_state.get("offset_py", 0)
    if "detector_click_error" in st.session_state:
        st.warning(st.session_state.pop("detector_click_error"))
    normalize = st.checkbox("Normalize full probe block", value=True,
                            help="Divide by the sum over all energy bins and all detector-plane pixels at this probe, before detector integration. Matches normalize_3d in the notebook.")
    six_d = [i for i in infos if len(i.shape) == 6]
    sample = 0
    if six_d:
        n_x = min(i.shape[2] for i in six_d)
        n_y = min(i.shape[3] for i in six_d)
        position_options = [(x, y) for x in range(n_x) for y in range(n_y)]
        if "probe_positions" in st.session_state:
            valid_positions = set(position_options)
            st.session_state["probe_positions"] = [
                tuple(position) for position in st.session_state["probe_positions"]
                if tuple(position) in valid_positions
            ]
        probe_positions = st.multiselect(
            "Probe positions (x, y)", position_options, default=[(0, 0)],
            format_func=lambda position: f"({position[0]}, {position[1]})",
            key="probe_positions",
            help="Choose individual probe positions. Each selected pair produces one spectrum per 6D file.")
        if not probe_positions:
            st.info("Select at least one probe position (x, y).")
            st.stop()
        if len(six_d) != len(infos):
            st.caption("3D files contribute one curve at probe (0, 0). Selected pairs apply to 6D scans.")
    else:
        probe_positions = [(0, 0)]
        st.caption("3D diffraction data · one spectrum per file.")

    with st.expander("Energy calibration", expanded=True):
        timestep = st.number_input("Simulation time step (fs)", min_value=0.000001,
                                   value=2.5, step=0.5, format="%.6f")
        stride = st.number_input("Sampling stride", min_value=1, value=3, step=1)
        st.caption("Confirm these values for your simulation. .npy arrays do not store time calibration. Applied to every selected file.")
        ordering = st.selectbox("Input energy ordering", ["FFT-shifted (notebook default)", "Unshifted FFT"])

    with st.expander("Gaussian broadening", expanded=True):
        broaden = st.checkbox("Broaden EELS spectrum", value=False)
        sigma_input = st.number_input("Gaussian σ (meV)", min_value=0.0, value=1.0, step=0.5,
                                      disabled=not broaden, help="Standard deviation of the Gaussian, applied to linear intensities after detector integration.")
        sigma_mev = sigma_input if broaden else 0.0
        if broaden:
            st.caption(f"FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV. Applies to spectra and downloads; the diffraction image stays unchanged.")

settings = dict(detector_radius_px=radius, center_offset_px=offset_px, center_offset_py=offset_py,
                normalize_3d=normalize, sample=sample, probe_positions_xy=probe_positions,
                timestep_fs=timestep, stride=stride, input_energy_ordering=ordering,
                gaussian_sigma_mev=sigma_mev, gaussian_boundary="reflect", gaussian_truncate=4.0,
                processing_order=["detector integration / optional full-probe normalization",
                                  "energy ordering", "optional Gaussian broadening",
                                  "display transform"])
curves = []
try:
    with st.spinner("Integrating detector intensities…"):
        for info, label in zip(infos, labels):
            positions = probe_positions if len(info.shape) == 6 else [(0, 0)]
            s = sample
            energy = energy_loss_axis_mev(info.canonical_shape[1], timestep, stride)
            for x, y in positions:
                intensity = cached_spectrum(info, s, x, y, radius, offset_px, offset_py, normalize)
                if ordering == "Unshifted FFT":
                    intensity = np.fft.fftshift(intensity)
                if sigma_mev > 0:
                    intensity = gaussian_broaden_spectrum(energy, intensity, sigma_mev)
                name = f"{label} · x={x}, y={y}" if len(info.shape) == 6 else label
                curves.append(dict(label=name, path=info.path, sample=s, probe_x=x, probe_y=y,
                                   energy=energy, intensity=intensity))
except (OSError, ValueError, IndexError, EOFError) as exc:
    st.error(f"Could not extract spectra: {exc}")
    st.stop()

with st.sidebar:
    with st.expander("Detector preview", expanded=True):
        preview_index = st.selectbox("Preview spectrum", list(range(len(curves))),
                                     format_func=lambda i: curves[i]["label"])
        requested_energy = st.number_input("Preview energy (meV)", value=0.0, step=1.0)
        pattern_log = st.checkbox("Log diffraction image", value=True)

metrics = st.columns(4)
metrics[0].metric("Selected scans", len(infos))
metrics[1].metric("Spectra", len(curves))
bins = sorted({len(c["energy"]) for c in curves})
metrics[2].metric("Energy bins", " / ".join(map(str, bins)))
resolutions = sorted({round(float(c["energy"][1] - c["energy"][0]), 6) for c in curves if len(c["energy"]) > 1})
metrics[3].metric("Resolution (meV)", " / ".join(f"{v:.3f}" for v in resolutions) or "—")

background_state = st.session_state.setdefault("background_state", BackgroundState())
revisions = {info.path: (info.mtime_ns, info.size) for info in infos}
background_state.source_revisions = revisions
had_background = bool(background_state.applied_results)
if background_state.sync_inputs(input_fingerprints(curves, settings, revisions)):
    st.session_state["bg_signal"] = "Input"
    if had_background:
        st.info("Background results cleared because inputs or selected curves changed. Settings are retained for refitting.")
if "bg_next_signal" in st.session_state:
    st.session_state["bg_signal"] = st.session_state.pop("bg_next_signal")
spectrum_tab, background_tab, angle_tab, details_tab = st.tabs(
    ["Spectra", "Background", "Angle-resolved EELS", "Files & method"])
with spectrum_tab:
    if background_state.applied_results:
        signal = st.radio("Spectrum signal", ["Input", "Corrected"],
                          horizontal=True, key="bg_signal", label_visibility="collapsed",
                          help="Corrected becomes available after a valid batch is applied in Background.")
        st.caption(f"{signal} · Applied: " + config_caption(background_state.applied_config))
    else:
        signal = "Input"
    background_state.signal = signal
    corrected = signal == "Corrected"
    controls = st.columns([1.3, 1, 1])
    mode_label = controls[0].selectbox("Intensity display", ["log10", "Linear"])
    mode = "log10" if mode_label == "log10" else "linear"
    x_min = controls[1].number_input("Energy min (meV)", value=-150.0, step=10.0)
    x_max = controls[2].number_input("Energy max (meV)", value=150.0, step=10.0)
    if x_min >= x_max:
        st.error("Energy min must be less than energy max.")
        st.stop()
    x_limits = (x_min, x_max)
    with st.expander("Line appearance", expanded=False):
        st.caption("Choose a spectrum to customize. Styles also apply to SVG/PNG downloads and are saved in NumPy metadata.")
        # Keep styles separate from widget state so removing/reselecting a curve
        # or editing its label does not discard its appearance during this session.
        saved_styles = st.session_state.setdefault("curve_styles", {})
        curve_keys = [curve_identity_key(c) for c in curves]
        curve_labels = dict(zip(curve_keys, [c["label"] for c in curves]))
        for i, curve_key in enumerate(curve_keys):
            saved_styles.setdefault(curve_key, dict(color=COLORS[i % len(COLORS)], line_style="Solid", width=2.0))
        if "style_editor" in st.session_state and st.session_state["style_editor"] not in curve_keys:
            st.session_state["style_editor"] = curve_keys[0]
        editing = st.selectbox("Spectrum to style", curve_keys, format_func=curve_labels.get, key="style_editor")
        current = saved_styles[editing]
        color_col, style_col, width_col = st.columns(3)
        color = color_col.color_picker("Line color", value=current["color"], key=f"line_color:{editing}")
        line_style = style_col.selectbox("Line style", list(LINE_STYLES),
                                         index=list(LINE_STYLES).index(current["line_style"]),
                                         key=f"line_style:{editing}")
        line_width = width_col.number_input("Line width", min_value=0.5, max_value=8.0,
                                            value=current["width"], step=0.5, key=f"line_width:{editing}")
        saved_styles[editing] = dict(color=color, line_style=line_style, width=line_width)
        for curve, curve_key in zip(curves, curve_keys):
            curve["style"] = saved_styles[curve_key].copy()
    toggle_cols = st.columns(2)
    show_hover = toggle_cols[0].toggle("Show hover details", value=True, key="show_hover_details",
                           help="Show or hide the energy and intensity popup when hovering over the spectra.")
    show_detector = toggle_cols[1].toggle("Show detector preview", value=True, key="show_detector_preview",
                              help="Show the click-to-position detector diffraction image beside the spectrum plot.")
    st.session_state["spectrum_render_revision"] = st.session_state.get("spectrum_render_revision", 0) + 1
    shown_curves = [dict(c, intensity=background_state.applied_results[curve_identity_key(c)].corrected)
                    for c in curves] if corrected else curves
    fig = go.Figure()
    for curve in shown_curves:
        style = curve["style"]
        fig.add_trace(go.Scatter(x=curve["energy"], y=(corrected_display if corrected else display_intensity)(curve["intensity"], mode),
                                 name=curve["label"], mode="lines", connectgaps=False,
                                 line=dict(color=style["color"], width=style["width"],
                                           dash=LINE_STYLES[style["line_style"]][0]),
                                 hovertemplate="%{y:.6g}<extra>%{fullData.name}</extra>"))
    fig.update_layout(height=440, margin=dict(l=20, r=20, t=15, b=20), template="plotly_white",
                      paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified" if show_hover else False,
                      legend=dict(orientation="h", y=-0.2, x=0), uirevision="spectrum",
                      meta=dict(eels_render_revision=st.session_state["spectrum_render_revision"]),
                      # Preserve exploration across recalculation; explicit limit edits still apply.
                      xaxis=dict(title="Energy loss (meV)", range=x_limits, zerolinecolor="#bdcbd4",
                                 gridcolor="#EAF0F7", hoverformat=".4f", uirevision=json.dumps(x_limits)),
                      yaxis=dict(title=intensity_label(mode, normalize, corrected), gridcolor="#EAF0F7",
                                 uirevision="yaxis"))
    workspace_row = st.container(key="workspace_row")
    with workspace_row:
        detector_col, plot_col = st.columns([1, 2], gap="medium") if show_detector else (None, st.container())
    with plot_col:
        with st.container(key="spectrum_panel"):
            st.markdown('<p class="eels-panel-title">EELS Spectrum</p>', unsafe_allow_html=True)
            with st.container(key="spectrum_axis_target"):
                st.plotly_chart(fig, key="spectrum_plot", use_container_width=True, config={"displaylogo": False,
                                "scrollZoom": True,
                                "toImageButtonOptions": {"format": "svg", "filename": "eels_spectra"}})
            axis_scaling(key="spectrum_axis_scaling", height=0)
            st.caption("Click a legend entry to hide that curve · double-click a legend entry to "
                       "isolate it (hide all others) · double-click again to restore all.")
            if sigma_mev > 0:
                st.caption(f"Gaussian broadening active: σ = {sigma_mev:g} meV (FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV).")
            if corrected and mode == "log10":
                st.caption("Nonpositive corrected samples are masked in log10 display (gaps are not connected). Signed linear residuals remain in data exports.")
            if not any(np.any((c["energy"] >= x_min) & (c["energy"] <= x_max)) for c in curves):
                st.warning("The selected energy window contains no data bins.")
    if detector_col is not None:
        with detector_col:
            with st.container(key="detector_panel"):
                st.markdown('<p class="eels-panel-title">Detector</p>', unsafe_allow_html=True)
                curve = curves[preview_index]
                info = scans[curve["path"]]
                axis = curve["energy"]
                _, raw_index = nearest_energy_index(axis, requested_energy,
                                                    unshifted=ordering == "Unshifted FFT")
                try:
                    pattern = cached_diffraction_pattern(info, raw_index, curve["sample"], curve["probe_x"], curve["probe_y"])
                    if not np.isfinite(pattern).all():
                        raise ValueError("Diffraction plane contains NaN or infinite intensities")
                    if pattern_log:
                        # Avoids materializing a copy of every positive value just for its minimum.
                        positive_min = np.min(pattern, where=pattern > 0, initial=np.inf)
                        if not np.isfinite(positive_min):
                            raise ValueError("This plane has no positive intensities. Turn off log diffraction image.")
                        shown = np.log10(np.clip(pattern, positive_min, None))
                    else:
                        shown = pattern
                    center = detector_center(info, offset_px, offset_py)
                    st.session_state["detector_render_revision"] = st.session_state.get("detector_render_revision", 0) + 1
                    image = go.Figure(go.Heatmap(z=shown, colorscale="Viridis", showscale=False,
                                                 hovertemplate="py=%{x}<br>px=%{y}<br>%{z:.5g}<extra></extra>"))
                    image.add_shape(type="circle", x0=center[1] - radius, x1=center[1] + radius,
                                    y0=center[0] - radius, y1=center[0] + radius,
                                    line=dict(color="#ff765e", width=2))
                    # Draw the center as shapes so it cannot steal clicks from nearby pixels.
                    image.add_shape(type="line", x0=center[1] - 2, x1=center[1] + 2,
                                    y0=center[0], y1=center[0], line=dict(color="#ff765e", width=2))
                    image.add_shape(type="line", x0=center[1], x1=center[1],
                                    y0=center[0] - 2, y1=center[0] + 2, line=dict(color="#ff765e", width=2))
                    # A fixed pixel size keeps the square detector image a predictable size in
                    # this narrow column, matching the layout scan_map_view.py already uses.
                    # No axis chrome to reserve room for, so the image can fill nearly the full box.
                    margin = dict(l=10, r=10, t=10, b=10)
                    content_side = 400
                    # A shared revision keyed on the plane's shape preserves exploration across
                    # recalculation but resets it if a differently-sized detector plane loads.
                    axis_revision = f"detector:{pattern.shape}"
                    image.update_layout(width=content_side + margin["l"] + margin["r"],
                                        height=content_side + margin["t"] + margin["b"],
                                        template="plotly_white", margin=margin, paper_bgcolor="white",
                                        plot_bgcolor="white", uirevision="detector",
                                        meta=dict(eels_render_revision=st.session_state["detector_render_revision"]),
                                        xaxis=dict(range=[-0.5, pattern.shape[1] - 0.5], uirevision=axis_revision,
                                                  visible=False),
                                        yaxis=dict(range=[-0.5, pattern.shape[0] - 0.5], visible=False,
                                                  scaleanchor="x", scaleratio=1, uirevision=axis_revision))
                    with st.container(key="detector_click_target"):
                        st.plotly_chart(image, key="detector_preview", use_container_width=False,
                                        config={"displaylogo": False, "scrollZoom": True,
                                                "toImageButtonOptions": {"format": "png", "filename": "eels_detector"}})
                    axis_scaling(key="detector_axis_scaling", height=0, data=dict(
                        selector=".st-key-detector_click_target .js-plotly-plot",
                        viewport_key="eels.detector.viewport"))
                    preview_id = json.dumps([info.path, info.mtime_ns, curve["sample"], curve["probe_x"], curve["probe_y"], raw_index])
                    st.session_state["detector_preview_context"] = dict(preview_id=preview_id, shape=pattern.shape,
                                                                       selected_shapes=[i.shape[-2:] for i in infos])
                    # Renew the bridge after each Streamlit redraw, including manual edits.
                    st.session_state["detector_click_revision"] = st.session_state.get("detector_click_revision", 0) + 1
                    detector_click_bridge(key="detector_click", data={"preview_id": preview_id,
                                          "revision": st.session_state["detector_click_revision"]},
                                          on_clicked_change=move_detector_from_click, height=0)
                    if any(c - radius < 0 or c + radius > n - 1 for c, n in zip(center, pattern.shape)):
                        st.warning("The detector extends beyond this array. Only pixels inside the recorded plane are integrated.")
                except (OSError, ValueError, IndexError, EOFError) as exc:
                    st.warning(f"Preview unavailable: {exc}")
                # Editable detector center/radius, always available even if the preview above failed.
                edit_cols = st.columns(3)
                edit_cols[0].number_input("px", min_value=-(min_px // 2), max_value=(min_px - 1) // 2,
                                          value=offset_px, step=1, key="offset_px")
                edit_cols[1].number_input("py", min_value=-(min_py // 2), max_value=(min_py - 1) // 2,
                                          value=offset_py, step=1, key="offset_py")
                edit_cols[2].number_input("radius", min_value=0.1, value=radius, step=1.0, key="detector_radius")
                st.button("Reset detector center", on_click=reset_detector_center, use_container_width=True)
    if detector_col is not None:
        equal_height(key="workspace_equal_height", height=0)
    settings["plot"] = dict(mode=mode, x_limits=x_limits, show_hover_details=show_hover)
    with st.expander("Export spectra & figures", expanded=True):
        st.caption("Data exports contain the full energy range and linear intensities. Figures use the display controls above; browser-only zoom and legend changes are not applied.")
        if sigma_mev > 0:
            st.caption(f"All downloads include Gaussian broadening (σ = {sigma_mev:g} meV). Turn broadening off to export unbroadened spectra; NPZ records the processing settings.")
        png_dpi = st.selectbox("PNG export resolution", PNG_DPI_OPTIONS, index=PNG_DPI_OPTIONS.index(300),
                               format_func=lambda d: f"{d} DPI", key="png_dpi_spectrum",
                               help="Resolution used for the PNG figure download below.")
        if background_state.applied_results:
            st.caption(f"Exported signal: {signal}. Unfitted bins are missing (NaN), never zero-filled.")
        if background_state.applied_results:
            csv_data = background_csv(curves, settings, background_state, signal, all_arrays=False)
            npz_data = background_npz(curves, settings, background_state, signal)
            export_stem = f"eels_spectra_{signal.lower()}"
        else:
            csv_data, npz_data = cached_export_csv(curves), cached_export_npz(curves, settings)
            export_stem = "eels_spectra"
        exports = st.columns(5)
        exports[0].download_button("CSV data", csv_data, export_stem + ".csv", "text/csv", use_container_width=True)
        exports[1].download_button("NumPy + settings", npz_data, export_stem + ".npz",
                                   "application/octet-stream", use_container_width=True)
        partial_corrected = corrected and any(not r.validity_mask.all() for r in background_state.applied_results.values())
        if partial_corrected:
            exports[2].caption("Partial corrected fits require masked NPZ. Select Input for the original notebook .npy export.")
        elif len(bins) == 1:
            exports[2].download_button("Notebook .npy", cached_notebook_npy(shown_curves), "eels_spectra_corrected.npy" if corrected else "eels_spectra.npy", "application/octet-stream", use_container_width=True)
        else:
            exports[2].caption("Use NPZ for unequal energy lengths.")
        figures = figure_downloads(shown_curves, mode, x_limits, normalize, png_dpi, corrected)
        exports[3].download_button("SVG figure", figures["svg"], export_stem + ".svg", "image/svg+xml", use_container_width=True)
        exports[4].download_button("PNG figure", figures["png"], export_stem + ".png", "image/png", use_container_width=True)

        if background_state.applied_results:
            background_downloads = st.columns(2)
            background_downloads[0].download_button("Background results CSV", background_csv(curves, settings, background_state, signal),
                "eels_background.csv", "text/csv", use_container_width=True)
            background_downloads[1].download_button("Background results NPZ", npz_data,
                "eels_background.npz", "application/octet-stream", use_container_width=True)

with background_tab:
    render_background(curves, background_state)

with angle_tab:
    render_angle_resolved(curves, scans, settings)

with details_tab:
    st.dataframe([{"Label": label, "File": Path(info.path).name, "Shape": str(info.shape),
                   "Type": info.dtype, "Size (MB)": round(info.size / 1e6, 1)}
                  for info, label in zip(infos, labels)], use_container_width=True, hide_index=True)
    st.markdown("""
**Array conventions**

- 3D: `(energy, px, py)`, one probe block per file.
- 6D: `(sample, energy, probe_x, probe_y, px, py)`, first sample and selected probe positions.
- The circular detector uses `(px − center_px)² + (py − center_py)² ≤ radius²`.
- Full-probe normalization divides by the sum over **all energy bins and all pixels** at that probe.
- Energy is `fftshift(fftfreq(n_energy, timestep_fs × stride / 1000)) × 4.13566769692386` in meV.
- Input intensities are assumed FFT-shifted by default, as in the notebook. Choose unshifted only if your simulation output uses raw FFT ordering.
- Optional Gaussian broadening operates on linear EELS intensity after detector integration and before log display. σ is entered in meV and converted to bins using each file's energy spacing. The kernel extends to 4σ; reflecting boundaries preserve the recorded sum without wrapping between energy endpoints. Edge features can be affected by this boundary assumption.

File headers are inspected without memory mapping. Selected data is read directly in small blocks, avoiding a full-file virtual-memory reservation. Changing the plot or preview reuses cached spectra.
Different energy lengths are supported; shared time step and stride must be appropriate for every selected scan.
""")
    st.code(json.dumps(settings, indent=2), language="json")
