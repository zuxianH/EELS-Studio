"""Box-selected energy-versus-detector-pixel maps for the Streamlit app."""
from pathlib import Path
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v2 as components

from eels_core import (
    curve_identity_key, extract_angle_resolved, nearest_energy_index, process_angle_resolved,
    rectangle_from_plot,
)
from cache_layer import cached_diffraction_pattern, PNG_DPI_OPTIONS

# Read once at import time rather than on every Streamlit rerun.
_RECTANGLE_SELECT_JS = Path(__file__).with_name("rectangle_select.js").read_text()


@st.cache_data(show_spinner=False, max_entries=16)
def cached_map(info, bounds, retain_axis, dummy, probe_x, probe_y, normalize):
    return extract_angle_resolved(info, bounds, retain_axis=retain_axis, dummy=dummy,
                                  probe_x=probe_x, probe_y=probe_y, normalize_3d=normalize)


def receive_rectangle():
    event = st.session_state.get("angle_rectangle", {}).get("selected")
    context = st.session_state.get("angle_rectangle_context")
    if not event or not context or event.get("preview_id") != context["preview_id"]:
        return
    try:
        bounds = rectangle_from_plot(context["shape"], event["x0"], event["x1"],
                                     event["y0"], event["y1"])
        for key, value in zip(context["bound_keys"], bounds):
            st.session_state[key] = value
    except (KeyError, TypeError, ValueError) as exc:
        st.session_state["angle_rectangle_error"] = str(exc)


@st.cache_data(show_spinner=False, max_entries=16)
def export_map(energy, pixels, intensity, metadata):
    output = io.BytesIO()
    np.savez_compressed(output, energy_mev=energy, pixel_offset=pixels, intensity=intensity,
                        metadata_json=np.array(json.dumps(metadata)))
    return output.getvalue()


def map_display(intensity, logarithmic):
    if not logarithmic:
        return intensity
    # Mask nonpositive bins so zeros do not stretch the log color range to -308.
    result = np.full(intensity.shape, np.nan)
    positive = intensity > 0
    result[positive] = np.log10(intensity[positive])
    return result


@st.cache_data(show_spinner=False, max_entries=4)
def map_figures(energy, pixels, shown, title, xlabel, color_label, energy_limits, color_limits, dpi=300):
    figure, axes = plt.subplots(figsize=(8, 6), layout="constrained")
    # Explicit pixel edges also render a strip only one pixel wide correctly.
    pixel_edges = np.r_[pixels - 0.5, pixels[-1] + 0.5]
    spacing = energy[1] - energy[0] if len(energy) > 1 else 1.0
    energy_edges = np.r_[energy - spacing / 2, energy[-1] + spacing / 2]
    artist = axes.pcolormesh(pixel_edges, energy_edges, np.ma.masked_invalid(shown), shading="flat",
                             cmap="viridis", vmin=color_limits[0], vmax=color_limits[1],
                             rasterized=True)
    axes.set(xlabel=xlabel, ylabel="Energy loss (meV)", title=title, ylim=energy_limits,
             xlim=(pixels[0] - 0.5, pixels[-1] + 0.5))
    figure.colorbar(artist, ax=axes, label=color_label)
    result = {}
    for extension in ("png", "svg"):
        output = io.BytesIO()
        figure.savefig(output, format=extension, dpi=dpi)
        result[extension] = output.getvalue()
    plt.close(figure)
    return result


def render_angle_resolved(curves, scans, settings):
    st.caption("Draw a rectangle on the diffraction image to form an energy-versus-pixel map. The selected strip is summed across one detector direction.")
    # Stable identifiers prevent a scan reorder from silently selecting another probe.
    choices = {curve_identity_key(c): c for c in curves}
    if st.session_state.get("angle_source") not in choices:
        st.session_state["angle_source"] = next(iter(choices))
    selected = st.selectbox("Map spectrum", list(choices), key="angle_source",
                            format_func=lambda k: choices[k]["label"])
    curve = choices[selected]
    info = scans[curve["path"]]
    shape = info.shape[-2:]
    energy = curve["energy"]
    controls = st.columns(3)
    direction = controls[0].selectbox("Retain detector direction", ["Horizontal (py)", "Vertical (px)"],
                                      key="angle_direction")
    retain_axis = "py" if direction == "Horizontal (py)" else "px"
    requested_energy = controls[1].number_input("Box preview energy (meV)", value=0.0, step=1.0,
                                               key="angle_preview_energy")
    preview_log = controls[2].checkbox("Log box diffraction image", value=True, key="angle_preview_log")
    # Bounds are stored per detector plane, shared by probes within that file.
    namespace = json.dumps([info.path, shape])
    bound_keys = [f"angle_roi:{namespace}:{name}" for name in ("r0", "r1", "c0", "c1")]
    defaults = (max(0, shape[0] // 2 - 10), min(shape[0] - 1, shape[0] // 2 + 9),
                0, shape[1] - 1)
    saved_rois = st.session_state.setdefault("angle_saved_rois", {})
    defaults = saved_rois.get(namespace, defaults)
    bound_columns = st.columns(4)
    bounds = []
    for col, key, label, default, count in zip(bound_columns, bound_keys,
            ("Row min (px)", "Row max (px)", "Column min (py)", "Column max (py)"),
            defaults, (shape[0], shape[0], shape[1], shape[1])):
        if key in st.session_state:
            st.session_state[key] = min(count - 1, max(0, st.session_state[key]))
        default = min(count - 1, max(0, default))
        bounds.append(int(col.number_input(label, min_value=0, max_value=count - 1,
                                           value=default, step=1, key=key, disabled=count == 1)))
    if "angle_rectangle_error" in st.session_state:
        st.warning(st.session_state.pop("angle_rectangle_error"))
    r0, r1, c0, c1 = bounds
    if r0 > r1 or c0 > c1:
        st.warning("Each rectangle minimum must be at most its maximum.")
        return
    saved_rois[namespace] = tuple(bounds)
    st.caption(f"Inclusive bounds · crop [:, {r0}:{r1 + 1}, {c0}:{c1 + 1}] · "
               f"sum over {'rows (px)' if retain_axis == 'py' else 'columns (py)'}. "
               "The box is independent of the circular spectrum detector.")
    display_controls = st.columns(3)
    logarithmic = display_controls[0].selectbox("Map color scale", ["log10", "Linear"], key="angle_color_scale") == "log10"
    energy_min = display_controls[1].number_input("Map energy min (meV)", value=-150.0, step=10.0, key="angle_energy_min")
    energy_max = display_controls[2].number_input("Map energy max (meV)", value=150.0, step=10.0, key="angle_energy_max")
    if energy_min >= energy_max:
        st.warning("Map energy min must be less than map energy max.")
        return
    color_limits = (None, None)
    if st.checkbox("Set map color limits", key="angle_manual_color"):
        low, high = st.columns(2)
        suffix = "log10 intensity" if logarithmic else "linear intensity"
        color_limits = (low.number_input(f"Map color min ({suffix})", value=-10.0 if logarithmic else 0.0, key="angle_color_min"),
                        high.number_input(f"Map color max ({suffix})", value=-5.0 if logarithmic else 1.0, key="angle_color_max"))
        if color_limits[0] >= color_limits[1]:
            st.warning("Map color min must be less than map color max.")
            return

    unshifted = settings["input_energy_ordering"] == "Unshifted FFT"
    index, raw_index = nearest_energy_index(energy, requested_energy, unshifted=unshifted)
    try:
        plane = cached_diffraction_pattern(info, raw_index, curve["dummy"], curve["probe_x"], curve["probe_y"])
        if not np.isfinite(plane).all():
            raise ValueError("Diffraction preview contains nonfinite values")
        pixels, raw_map = cached_map(info, tuple(bounds), retain_axis, curve["dummy"],
                                     curve["probe_x"], curve["probe_y"], settings["normalize_3d"])
        intensity = process_angle_resolved(energy, raw_map, unshifted=unshifted,
                                           sigma_mev=settings["gaussian_sigma_mev"])
    except (OSError, ValueError, IndexError) as exc:
        st.error(f"Could not create angle-resolved map: {exc}")
        return

    st.caption("Drag to draw a new box, or move/resize its outline. Use the toolbar for zoom or pan. Numeric bounds update when you release the mouse.")
    preview_column, map_column = st.columns(2)
    with preview_column:
        preview = go.Figure(go.Heatmap(z=map_display(plane, preview_log), colorscale="Viridis",
                            colorbar=dict(title="log10(I)" if preview_log else "Intensity"),
                            hovertemplate="Column py=%{x}<br>Row px=%{y}<extra></extra>"))
        preview.add_shape(type="rect", x0=c0 - 0.5, x1=c1 + 0.5, y0=r0 - 0.5, y1=r1 + 0.5,
                           line=dict(color="#ff765e", width=2), fillcolor="rgba(255,118,94,0.08)",
                           editable=True)
        preview.update_layout(height=520, title=f"Draw a box · E = {energy[index]:.3f} meV",
                               template="plotly_white", dragmode="drawrect",
                               newshape=dict(line_color="#ff765e", fillcolor="rgba(255,118,94,0.08)"),
                               margin=dict(l=10, r=10, t=50, b=30),
                               xaxis=dict(title="Column py (pixel)", range=[-0.5, shape[1] - 0.5]),
                               yaxis=dict(title="Row px (pixel)", range=[-0.5, shape[0] - 0.5],
                                          scaleanchor="x", scaleratio=1))
        with st.container(key="angle_rectangle_target"):
            st.plotly_chart(preview, key="angle_diffraction", use_container_width=True,
                            config={"displaylogo": False, "modeBarButtonsToAdd": ["drawrect"],
                                    "edits": {"shapePosition": True}})
        preview_id = json.dumps([selected, info.mtime_ns, info.revision, shape, raw_index])
        st.session_state["angle_rectangle_context"] = dict(preview_id=preview_id, shape=shape,
                                                           bound_keys=bound_keys)
        bridge = components.component("eels_angle_rectangle", js=_RECTANGLE_SELECT_JS)
        revision = st.session_state.get("angle_rectangle_revision", 0) + 1
        st.session_state["angle_rectangle_revision"] = revision
        bridge(key="angle_rectangle", data=dict(preview_id=preview_id, revision=revision),
               on_selected_change=receive_rectangle, height=0)
        if preview_log and not np.any(plane > 0):
            st.caption("This preview plane has no positive intensity. Choose another energy or turn off its log scale.")

    xlabel = f"{retain_axis} offset from array center (pixels)"
    base_label = "Normalized strip intensity" if settings["normalize_3d"] else "Strip intensity"
    color_label = f"log10({base_label.lower()})" if logarithmic else base_label
    shown = map_display(intensity, logarithmic)
    with map_column:
        plot = go.Figure(go.Heatmap(x=pixels, y=energy, z=shown, customdata=intensity,
                          colorscale="Viridis", zmin=color_limits[0], zmax=color_limits[1],
                          colorbar=dict(title=dict(text=color_label, side="right"), thickness=15),
                          hovertemplate="Pixel offset=%{x}<br>Energy=%{y:.4f} meV<br>I=%{customdata:.6g}<extra></extra>"))
        plot.update_layout(height=520, title=curve["label"], template="plotly_white",
                             margin=dict(l=10, r=10, t=50, b=30),
                             xaxis=dict(title=xlabel, range=[pixels[0] - 0.5, pixels[-1] + 0.5]),
                             yaxis=dict(title="Energy loss (meV)", range=[energy_min, energy_max]))
        st.plotly_chart(plot, key="angle_map", use_container_width=True,
                        config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
    st.caption(f"Map shape: {len(energy)} energy bins × {len(pixels)} pixels. "
               "Pixel zero is the array's integer center; no angular calibration is assumed. "
               "Sidebar normalization, energy calibration, and Gaussian broadening apply. Broadening acts only along energy.")
    if logarithmic and np.any(intensity <= 0):
        st.caption("Nonpositive bins are blank on the log map; exported linear data retains them.")
    if not np.any((energy >= energy_min) & (energy <= energy_max)):
        st.warning("The selected map energy window contains no data bins.")
    metadata = dict(source={k: curve[k] for k in ("label", "path", "dummy", "probe_x", "probe_y")},
                     settings=settings, roi_bounds_inclusive=bounds,
                     retained_axis=retain_axis, pixel_origin="array integer center",
                     intensity_axes=["energy", "pixel"], color_scale="log10" if logarithmic else "linear",
                     energy_limits=[energy_min, energy_max], color_limits=color_limits)
    png_dpi = st.selectbox("PNG export resolution", PNG_DPI_OPTIONS, index=PNG_DPI_OPTIONS.index(300),
                           format_func=lambda d: f"{d} DPI", key="png_dpi_angle",
                           help="Resolution used for the Map PNG download below.")
    downloads = st.columns(3)
    downloads[0].download_button("Map NumPy + settings", export_map(energy, pixels, intensity, metadata),
                                  "angle_resolved_eels.npz", "application/octet-stream")
    figures = map_figures(energy, pixels, shown, curve["label"], xlabel, color_label,
                           (energy_min, energy_max), color_limits, png_dpi)
    for col, extension in zip(downloads[1:], ("png", "svg")):
        col.download_button(f"Map {extension.upper()}", figures[extension],
                              f"angle_resolved_eels.{extension}", f"image/{'svg+xml' if extension == 'svg' else 'png'}")
    st.caption("NPZ contains the full energy range, pixel offsets, linear intensity, and processing settings. Figure downloads use the map limits above.")
