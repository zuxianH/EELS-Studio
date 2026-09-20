"""Interactive raw detector map at selected energies or energy ranges of a 6D probe scan."""
import io
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from eels_core import (circular_detector_mask, detector_center, detector_scan_map,
                       energy_loss_axis_mev, gaussian_broaden_spectrum)
from cache_layer import PNG_DPI_OPTIONS


def reset_map_detector_center():
    st.session_state["map_offset_px"] = 0
    st.session_state["map_offset_py"] = 0


@st.cache_data(show_spinner=False, max_entries=64)
def cached_scan_map(info, raw_index, dummy, radius, offset_px, offset_py):
    return detector_scan_map(info, raw_index, dummy=dummy, radius=radius,
                             offset_px=offset_px, offset_py=offset_py)


_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_RANGE_PATTERN = re.compile(rf"^(?P<lo>{_NUMBER})-(?P<hi>{_NUMBER})$")


def parse_map_energies(text):
    """Parse map energy requests separated by commas, semicolons, or whitespace.

    A plain value like "20" requests the single nearest stored bin. A range like
    "10-20" requests every bin whose energy falls within that window, inclusive.
    Returns a list of (lo, hi) tuples, with lo == hi for plain values.
    """
    tokens = [token for token in re.split(r"[,;\s]+", text.strip()) if token]
    if not tokens:
        raise ValueError("Enter at least one map energy or range, for example: 20, 40, 10-20")
    requests = []
    for token in tokens:
        match = _RANGE_PATTERN.match(token)
        if match:
            lo, hi = float(match["lo"]), float(match["hi"])
        else:
            try:
                lo = hi = float(token)
            except ValueError:
                raise ValueError(
                    "Enter numeric map energies or ranges separated by commas, for example: 20, 40, 10-20"
                ) from None
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError("Map energies must be finite numbers")
        if lo > hi:
            raise ValueError(f"Range '{token}' must have its lower bound first, for example: 10-20")
        requests.append((lo, hi))
    return requests


def selected_map_bins(axis, requests, ordering):
    """Group requests mapping to the same set of stored bins, preserving input order.

    A point request (lo == hi) picks the single nearest bin. A range request sums
    every bin whose energy falls within [lo, hi], inclusive.
    """
    raw_indices = np.arange(len(axis))
    if ordering == "Unshifted FFT":
        raw_indices = np.fft.fftshift(raw_indices)
    bins = {}
    order = []
    for lo, hi in requests:
        if lo == hi:
            axis_indices = (int(np.argmin(np.abs(axis - lo))),)
        else:
            axis_indices = tuple(i for i in range(len(axis)) if lo <= axis[i] <= hi)
            if not axis_indices:
                raise ValueError(f"No energy bins fall within {lo:g}-{hi:g} meV")
        if axis_indices not in bins:
            bins[axis_indices] = dict(
                axis_indices=list(axis_indices),
                energy_indices=[int(raw_indices[i]) for i in axis_indices],
                bin_energies_mev=[float(axis[i]) for i in axis_indices],
                requested=[])
            order.append(axis_indices)
        bins[axis_indices]["requested"].append([lo, hi])
    return [bins[key] for key in order]


def bin_title(entry):
    """Panel title for a bin entry: a single energy, or a summed range with its bin count."""
    energies = entry["bin_energies_mev"]
    if len(energies) == 1:
        return f"{energies[0]:.6g} meV"
    return f"{energies[0]:.6g}–{energies[-1]:.6g} meV ({len(energies)} bins)"


def map_broadening_weights(axis, axis_index, sigma_mev):
    """Gaussian kernel weights, by axis position, for broadening one map across nearby energy bins.

    Reuses gaussian_broaden_spectrum on a one-hot vector so the reflect-boundary
    and truncation behavior exactly matches spectrum broadening.
    """
    if sigma_mev <= 0:
        return {axis_index: 1.0}
    onehot = np.zeros(len(axis))
    onehot[axis_index] = 1.0
    weights = gaussian_broaden_spectrum(axis, onehot, sigma_mev)
    return {int(i): float(weights[i]) for i in np.flatnonzero(weights)}


@st.cache_data(show_spinner=False, max_entries=8)
def maps_png(scan_maps, titles, columns=3, color_limits=None, dpi=300):
    columns = min(columns, len(scan_maps))
    rows = (len(scan_maps) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 6 * rows),
                             layout="constrained", squeeze=False)
    try:
        used_axes = list(axes.flat)[:len(scan_maps)]
        image = None
        for ax, scan_map, title in zip(used_axes, scan_maps, titles):
            image = ax.imshow(scan_map.T, origin="lower", cmap="viridis", interpolation="nearest",
                               vmin=color_limits[0] if color_limits else None,
                               vmax=color_limits[1] if color_limits else None)
            ax.set(xlabel="Probe x (index)", ylabel="Probe y (index)", title=title)
            # Sharing one scale makes a colorbar per panel redundant; show one for all.
            if color_limits is None:
                fig.colorbar(image, ax=ax, label="Detector-summed intensity (a.u.)")
        for ax in list(axes.flat)[len(scan_maps):]:
            ax.set_visible(False)
        if color_limits is not None:
            fig.colorbar(image, ax=used_axes, label="Detector-summed intensity (a.u.)", shrink=0.8)
        output = io.BytesIO()
        fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(fig)


@st.cache_data(show_spinner=False, max_entries=8)
def all_maps_npz_bytes(scan_maps, bins, base_metadata):
    metadata = dict(base_metadata, axes=["energy_map", "probe_x", "probe_y"], maps=bins)
    energies = [entry["bin_energies_mev"] for entry in bins]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, scan_maps=np.stack(scan_maps),
                        selected_energies_mev=np.array([float(np.mean(e)) for e in energies]),
                        energy_lo_mev=np.array([e[0] for e in energies]),
                        energy_hi_mev=np.array([e[-1] for e in energies]),
                        bin_count=np.array([len(e) for e in energies]),
                        metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()



def render_scan_map(infos, labels):
    six_d = {info.path: (info, label) for info, label in zip(infos, labels) if len(info.shape) == 6}
    if not six_d:
        st.info("Select a 6D scan to visualize a 2D probe map. 3D arrays contain only one probe position.")
        return
    with st.sidebar:
        st.divider()
        st.header("2D scan map")
        if st.session_state.get("map_scan") not in six_d:
            st.session_state["map_scan"] = next(iter(six_d))
        path = st.selectbox("Map scan", list(six_d), format_func=lambda p: six_d[p][1], key="map_scan")
        info, _ = six_d[path]
        dummy = 0
        energy_text = st.text_input("Map energies (meV)", value="0", key="map_energies",
                                    help="Enter one or more energies separated by commas, for example: 20, 40, 60, 80. "
                                         "Use lo-hi for a range summed over every bin inside it, for example: 10-20.")
        shared_scale = st.checkbox("Shared color scale", value=True, key="map_shared_scale",
                                   help="Use the same intensity range across all maps. Turn off to show each map's spatial contrast on its own scale.")
        manual_scale = st.checkbox("Set intensity scale manually", value=False, key="map_manual_scale",
                                   help="Override the automatic color range with fixed min/max values, applied to all maps.")
        manual_limits = None
        if manual_scale:
            lo_col, hi_col = st.columns(2)
            manual_limits = (lo_col.number_input("Intensity min (a.u.)", value=0.0, format="%.6g", key="map_color_min"),
                             hi_col.number_input("Intensity max (a.u.)", value=1.0, format="%.6g", key="map_color_max"))
        columns_per_row = st.number_input("Maps per row", min_value=1, max_value=12, value=3, step=1,
                                          key="map_columns",
                                          help="Grid layout for the panels below and the combined PNG export, e.g. 2 for a 2-wide grid, 1 to stack maps in a single column.")
        radius = st.number_input("Detector radius (pixels)", min_value=0.1, value=21.0,
                                 step=1.0, key="map_radius")
        offset_cols = st.columns(2)
        offsets = []
        for col, axis, size in zip(offset_cols, ("px", "py"), info.shape[-2:]):
            key = f"map_offset_{axis}"
            lo, hi = -(size // 2), (size - 1) // 2
            if key in st.session_state:
                st.session_state[key] = max(lo, min(hi, st.session_state[key]))
            offsets.append(col.number_input(f"${axis[0]}_{axis[1]}$ offset", min_value=lo, max_value=hi,
                                            value=0, step=1, key=key))
        st.button("Reset detector center", on_click=reset_map_detector_center,
                 use_container_width=True, key="map_reset_center")
        with st.expander("Energy calibration", expanded=True):
            timestep = st.number_input("Simulation time step (fs)", min_value=0.000001,
                                       value=5.0, step=0.5, format="%.6f", key="map_timestep")
            stride = st.number_input("Sampling stride", min_value=1, value=3, step=1, key="map_stride")
            st.caption("Confirm the time calibration for this simulation. The map covers every probe x and y.")
            ordering = st.selectbox("Input energy ordering", ["FFT-shifted (notebook default)", "Unshifted FFT"],
                                    key="map_ordering")
        with st.expander("Gaussian broadening", expanded=True):
            broaden = st.checkbox("Broaden EELS scan map", value=False, key="map_broaden")
            sigma_input = st.number_input("Gaussian σ (meV)", min_value=0.0, value=1.0, step=0.5,
                                          disabled=not broaden, key="map_sigma",
                                          help="Standard deviation of the Gaussian, applied across nearby energy "
                                               "bins before summing each map.")
            sigma_mev = sigma_input if broaden else 0.0
            if broaden:
                st.caption(f"FWHM = {sigma_mev * np.sqrt(8 * np.log(2)):.3f} meV. Each map becomes a "
                          "Gaussian-weighted sum over nearby energy bins.")
    st.subheader("2D scan maps")
    st.caption("Raw detector-summed intensity at each selected energy bin, shown on a linear color scale. "
               "A range (e.g. 10-20) sums every bin within that window into one map. "
               "Map controls are independent of spectrum controls; normalization does not apply to these maps. "
               "Optional Gaussian broadening (sidebar) sums each map over nearby energy bins.")
    try:
        requests = parse_map_energies(energy_text)
        axis = energy_loss_axis_mev(info.shape[1], timestep, stride)
        bins = selected_map_bins(axis, requests, ordering)
        raw_indices = np.arange(len(axis))
        if ordering == "Unshifted FFT":
            raw_indices = np.fft.fftshift(raw_indices)
        center = detector_center(info, *offsets)
        mask = circular_detector_mask(info.shape[-2:], center, radius)

        def broadened_scan_map(entry):
            total = None
            for axis_index in entry["axis_indices"]:
                for index, weight in map_broadening_weights(axis, axis_index, sigma_mev).items():
                    contribution = weight * cached_scan_map(info, int(raw_indices[index]), dummy, radius, *offsets)
                    total = contribution if total is None else total + contribution
            return total

        with st.spinner("Summing the detector across probe rows at selected energies…"):
            # Each read reduces one energy slice to a small map before the next
            # energy is read. Only these reduced maps are kept in memory.
            scan_maps = [broadened_scan_map(entry) for entry in bins]
        if manual_scale:
            if manual_limits[0] >= manual_limits[1]:
                st.error("Intensity min must be less than intensity max.")
                return
            limits = manual_limits
        else:
            limits = ((min(float(m.min()) for m in scan_maps), max(float(m.max()) for m in scan_maps))
                      if shared_scale else None)
        metrics = st.columns(3)
        metrics[0].metric("Energy maps", len(bins))
        metrics[1].metric("Probe positions", str(scan_maps[0].size))
        metrics[2].metric("Detector pixels", str(int(mask.sum())))
        if len(bins) < len(requests):
            st.info("Some requested energies or ranges select the same set of recorded bins; "
                    "each distinct selection is shown once.")
        if manual_scale:
            range_caption = f"Manual intensity range [{limits[0]:.6g}, {limits[1]:.6g}] applied to all maps."
        elif shared_scale:
            range_caption = "Shared intensity range across all maps."
        else:
            range_caption = "Each map uses its own intensity range."
        st.caption(f"Detector center (px, py) = {center}. Horizontal = probe x; vertical = probe y. " + range_caption)
        base_metadata = dict(source_file=info.path, dummy=dummy,
                             timestep_fs=timestep, stride=stride, input_energy_ordering=ordering,
                             detector_radius_px=radius, detector_center_px_py=center,
                             axes=["probe_x", "probe_y"], intensity="raw detector sum",
                             gaussian_sigma_mev=sigma_mev,
                             shared_color_scale=shared_scale, manual_color_scale=manual_scale,
                             color_limits=limits)
        titles = [bin_title(entry) for entry in bins]
        stem = f"{Path(info.path).stem}_scan_map"
        n_columns = min(columns_per_row, len(bins))
        # Reserve fixed pixel margins for the title, axis labels, and colorbar, then size
        # the panel so the plotting area itself (width/height minus those margins) is
        # already square. Plotly's colorbar length tracks the *declared* axis domain, not
        # the smaller box "constrain: domain" draws when it has to correct a mismatch, so
        # building a square domain up front (rather than relying on that correction) is
        # what keeps the colorbar the same height as the map.
        margin = dict(l=55, r=90, t=45, b=55)
        content_side = 560 if n_columns == 1 else max(220, min(480, 1400 // n_columns))
        panel_width = content_side + margin["l"] + margin["r"]
        panel_height = content_side + margin["t"] + margin["b"]
        for start in range(0, len(bins), n_columns):
            panels = st.columns(n_columns)
            for panel, i in zip(panels, range(start, min(start + n_columns, len(bins)))):
                with panel:
                    entry, scan_map, title = bins[i], scan_maps[i], titles[i]
                    panel_key = "-".join(str(index) for index in entry["energy_indices"])
                    if np.all(scan_map == 0):
                        st.info("This detector-integrated map is all zero; no spatial contrast is present in this slice.")
                    figure = go.Figure(go.Heatmap(
                        x=np.arange(scan_map.shape[0]), y=np.arange(scan_map.shape[1]), z=scan_map.T,
                        colorscale="Viridis", zsmooth=False,
                        colorbar=dict(title="Intensity (a.u.)", thickness=18),
                        zmin=limits[0] if limits else None, zmax=limits[1] if limits else None,
                        hovertemplate="probe x=%{x}<br>probe y=%{y}<br>Intensity=%{z:.6g}<extra></extra>"))
                    figure.update_layout(
                        title=title, width=panel_width, height=panel_height,
                        template="plotly_white", margin=margin,
                        xaxis=dict(title="Probe x (index)", range=[-0.5, scan_map.shape[0] - 0.5]),
                        yaxis=dict(title="Probe y (index)", range=[-0.5, scan_map.shape[1] - 0.5],
                                  scaleanchor="x", scaleratio=1))
                    panel_stem = f"{stem}_E{panel_key}"
                    st.plotly_chart(figure, key=f"scan_map_plot:{panel_key}", use_container_width=False,
                                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": panel_stem}})
        if len(bins) > 1:
            st.subheader("Download all maps")
            png_dpi = st.selectbox("PNG export resolution", PNG_DPI_OPTIONS, index=PNG_DPI_OPTIONS.index(300),
                                   format_func=lambda d: f"{d} DPI", key="png_dpi_scan_map",
                                   help="Resolution used for the PNG download below.")
            exports = st.columns(2)
            exports[0].download_button("All maps + settings .npz", all_maps_npz_bytes(scan_maps, bins, base_metadata),
                                       stem + "s.npz", "application/octet-stream", key="all_maps_npz")
            exports[1].download_button("All maps PNG", maps_png(scan_maps, titles, n_columns, limits, png_dpi),
                                       stem + "s.png", "image/png", key="all_maps_png")
    except (OSError, ValueError, IndexError, EOFError) as exc:
        st.error(f"Could not create scan maps: {exc}")
