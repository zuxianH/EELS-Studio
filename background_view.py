"""Background tab for the existing EELS Studio interface."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from background_core import (ANALYTIC_MODELS, BACKGROUND_MODELS, BackgroundConfig, auto_segments_from_peaks,
                             config_caption, corrected_display, default_analytic_segments, initial_bounds)
from cache_layer import cached_background_fit
from eels_core import curve_identity_key
from axis_scaling import register_axis_scaling


def _sync_lambda(source, target):
    st.session_state[target] = st.session_state[source]


def _apply_auto_segments(curve, n_segments, low, high):
    # Runs before the script reruns, so the segment number_inputs (instantiated
    # below with the same keys) pick up these values instead of their defaults.
    try:
        detected = auto_segments_from_peaks(curve["energy"], curve["intensity"], n_segments, low, high)
    except ValueError as exc:
        st.session_state["bg_auto_segments_error"] = str(exc)
        return
    st.session_state.pop("bg_auto_segments_error", None)
    for i, (lo, hi) in enumerate(detected):
        st.session_state[f"bg_seg_lo_{i}"] = lo
        st.session_state[f"bg_seg_hi_{i}"] = hi


def _intensity_view_range(arrays, *, include_zero=False):
    """Padded limits from the visible samples, without altering their values."""
    values = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.append(values, 0.0)
    if not len(values):
        return None
    low, high = float(values.min()), float(values.max())
    padding = (high - low) * 0.06 if high > low else (abs(low) * 0.06 or 1.0)
    return [low - padding, high + padding]


def preview_figure(curve, result=None, mode="linear", *, view_bounds=None, view_revision=0):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10)
    fig.add_trace(go.Scatter(x=curve["energy"], y=corrected_display(curve["intensity"], mode), name="Input (after broadening)",
                            mode="lines", connectgaps=False, line=dict(color="#88949e", width=1.5)), row=1, col=1)
    if result is not None:
        fit_segments = result.diagnostics.fit_segments
        if fit_segments:
            for lo, hi in fit_segments:
                for row in (1, 2):
                    fig.add_vrect(x0=lo, x1=hi, fillcolor="#137c78", opacity=0.10,
                                  line_width=0, row=row, col=1, exclude_empty_subplots=False)
        else:
            low, high = result.diagnostics.actual_bounds
            for row in (1, 2):
                fig.add_vrect(x0=low, x1=high, fillcolor="#137c78", opacity=0.08,
                              line_width=0, row=row, col=1, exclude_empty_subplots=False)
        if result.diagnostics.valid:
            fig.add_trace(go.Scatter(x=result.energy, y=corrected_display(result.baseline, mode), name="Estimated baseline (dashed)",
                mode="lines", connectgaps=False, line=dict(color="#485868", width=2, dash="dash")), row=1, col=1)
            style = curve["style"]
            dash = {"Solid": "solid", "Dashed": "dash", "Dotted": "dot", "Dash-dot": "dashdot"}[style["line_style"]]
            fig.add_trace(go.Scatter(x=result.energy, y=corrected_display(result.corrected, mode), name="Corrected (input − baseline)",
                mode="lines", connectgaps=False, line=dict(color=style["color"], width=style["width"], dash=dash)), row=2, col=1)
    if mode != "log10":
        fig.add_hline(y=0, line_color="#88949e", line_width=1, row=2, col=1)
    identity = "background:" + curve_identity_key(curve)
    fig.update_layout(height=590, template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=15, b=15), hovermode="x unified",
                      legend=dict(orientation="h", y=1.13, x=0),
                      uirevision=identity)
    fig.update_yaxes(title_text="log10(input intensity)" if mode == "log10" else "Input intensity", row=1, col=1)
    fig.update_yaxes(title_text="log10(positive residual)" if mode == "log10" else "Signed residual", row=2, col=1)
    # Explicit view actions reset saved ranges; fitting and ordinary reruns do not.
    view_identity = f"{identity}:view:{view_revision}"
    fig.update_yaxes(uirevision=f"{view_identity}:{mode}", zeroline=mode != "log10")
    fig.update_xaxes(uirevision=view_identity)
    if view_bounds is not None:
        low, high = view_bounds
        visible = (curve["energy"] >= low) & (curve["energy"] <= high)
        upper = [corrected_display(curve["intensity"], mode)[visible]]
        if result is not None and result.diagnostics.valid:
            upper.append(corrected_display(result.baseline, mode)[visible])
            lower = _intensity_view_range(
                [corrected_display(result.corrected, mode)[visible]], include_zero=mode != "log10")
            if lower is not None:
                fig.update_yaxes(range=lower, autorange=False, row=2, col=1)
        upper_range = _intensity_view_range(upper)
        if upper_range is not None:
            fig.update_yaxes(range=upper_range, autorange=False, row=1, col=1)
        fig.update_xaxes(range=[low, high], autorange=False)
    fig.update_xaxes(title_text="Energy loss (meV)", row=2, col=1)
    return fig


def render_background(curves, state):
    st.caption("Fit linear, energy-unweighted detector spectra after optional Gaussian broadening. "
               "Changing Gaussian σ requires a new fit. Plot zoom and display limits never change the fitting domain.")
    st.markdown("""<style>
    @media (max-width: 900px) {
      .st-key-background_layout [data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
      .st-key-background_layout [data-testid="stColumn"] {min-width: 100%;}
    }
    </style>""", unsafe_allow_html=True)
    with st.container(key="background_layout"):
        left, right = st.columns([3, 7], gap="medium")
        identities = [curve_identity_key(c) for c in curves]
        labels = dict(zip(identities, [c["label"] for c in curves]))
        with right:
            if st.session_state.get("bg_preview_curve") not in identities:
                st.session_state["bg_preview_curve"] = identities[0]
            selected = st.selectbox("Preview-spectrum selector", identities, format_func=labels.get, key="bg_preview_curve")
            display = st.selectbox("Preview intensity display", ["Linear", "log10"], key="bg_display",
                help="Display only: input, baseline, and corrected intensity. Fitting always uses linear intensity.")
            mode = "log10" if display == "log10" else "linear"
        curve = curves[identities.index(selected)]
        with left, st.container(border=True):
            method = st.selectbox("Method", ["arPLS", "SNIP", *ANALYTIC_MODELS], key="bg_method")
            low, high = initial_bounds(curves)
            if method in ANALYTIC_MODELS:
                spec = BACKGROUND_MODELS[method]
                n_segments = st.number_input("Fit segments", min_value=2, max_value=4, value=2, step=1,
                    key="bg_n_segments",
                    help="Flanking energy windows that exclude the peak; the background is fit only on "
                         "these, then evaluated and subtracted across their full span (first segment's "
                         "start to last segment's end), including the peak region between them.")
                st.button("Auto-detect segments from peak", key="bg_auto_segments", use_container_width=True,
                    help="Finds the most prominent local peak(s) in the preview spectrum (at least a ~15% "
                         "local rise) and places segments to flank them. A starting point, not a substitute "
                         "for checking the preview plot — it can miss a peak below that threshold or pick "
                         "the wrong one for a noisy or multi-featured spectrum.",
                    on_click=_apply_auto_segments, args=(curve, n_segments, low, high))
                if "bg_auto_segments_error" in st.session_state:
                    st.warning(st.session_state.pop("bg_auto_segments_error"))
                defaults = default_analytic_segments(low, high, n_segments)
                segments = []
                for i in range(n_segments):
                    # Seed via session_state (not value=) since Auto-detect also writes these
                    # keys; passing both triggers a Streamlit widget-policy warning.
                    st.session_state.setdefault(f"bg_seg_lo_{i}", defaults[i][0])
                    st.session_state.setdefault(f"bg_seg_hi_{i}", defaults[i][1])
                    cols = st.columns(2)
                    seg_lo = cols[0].number_input(f"Segment {i + 1} min (meV)", key=f"bg_seg_lo_{i}", format="%.6g")
                    seg_hi = cols[1].number_input(f"Segment {i + 1} max (meV)", key=f"bg_seg_hi_{i}", format="%.6g")
                    segments.append((seg_lo, seg_hi))
                minimum, maximum = min(lo for lo, _ in segments), max(hi for _, hi in segments)
                energy_factor = st.number_input("Energy scale factor (meV)", min_value=1e-6, value=40.0,
                    key="bg_energy_factor",
                    help="Energies are divided by this before fitting (numerical conditioning), "
                         "matching the source toolkit.")
                with st.expander("Start point & bounds", expanded=False):
                    st.caption("Defaults for power/power0 mirror the source toolkit's STO example; "
                               "exppoly/pVoigt defaults are generic starting points that need tuning "
                               "per dataset.")
                    model_start, model_lower, model_upper = [], [], []
                    for i, name in enumerate(spec.params):
                        c1, c2, c3 = st.columns(3)
                        model_start.append(c1.number_input(f"{name} start", value=spec.start[i],
                            key=f"bg_mstart_{method}_{name}", format="%.6g"))
                        model_lower.append(c2.number_input(f"{name} lower bound", value=spec.bounds[0][i],
                            key=f"bg_mlower_{method}_{name}", format="%.6g"))
                        model_upper.append(c3.number_input(f"{name} upper bound", value=spec.bounds[1][i],
                            key=f"bg_mupper_{method}_{name}", format="%.6g"))
                config = BackgroundConfig(method=method, domain="selected", energy_min=minimum, energy_max=maximum,
                    segments=tuple(segments), model_start=tuple(model_start), model_lower=tuple(model_lower),
                    model_upper=tuple(model_upper), energy_factor=energy_factor)
            else:
                domain = st.selectbox("Fit domain", ["Selected energy interval", "Full recorded spectrum"], key="bg_domain")
                minimum = st.number_input("Fit energy minimum (meV)", value=low, format="%.6f", key="bg_min",
                                          disabled=domain == "Full recorded spectrum")
                maximum = st.number_input("Fit energy maximum (meV)", value=high, format="%.6f", key="bg_max",
                                          disabled=domain == "Full recorded spectrum")
                st.caption("Initial bounds use common positive-energy coverage where possible. The first positive bin "
                           "is not automatically beyond the zero-loss tail. At least eight fitted bins are required.")
                log_lambda, tolerance, iterations, window = 5.0, 1e-3, 100, 10.0
                if method == "arPLS":
                    st.session_state.setdefault("bg_lambda_slider", 5.0)
                    st.session_state.setdefault("bg_lambda_number", 5.0)
                    st.slider("log10(λ)", 2.0, 10.0, step=0.1, key="bg_lambda_slider",
                              on_change=_sync_lambda, args=("bg_lambda_slider", "bg_lambda_number"))
                    log_lambda = st.number_input("log10(λ), numeric", min_value=2.0, max_value=10.0, step=0.1,
                        key="bg_lambda_number", on_change=_sync_lambda, args=("bg_lambda_number", "bg_lambda_slider"))
                    st.caption(f"λ = {10 ** log_lambda:g}. Larger λ gives a smoother baseline. λ = 1e5 is a starting point, not a TACAW optimum.")
                else:
                    window = st.number_input("Maximum half-window (meV)", min_value=0.000001, value=10.0,
                                             key="bg_window", format="%.6f")
                    st.caption("Rounded to the nearest bin separately on each grid (half ties up). Linear intensity; no log transform.")
                with st.expander("Advanced settings", expanded=False):
                    if method == "arPLS":
                        tolerance = st.number_input("Tolerance", min_value=1e-12, max_value=0.999,
                                                    value=1e-3, format="%.6g", key="bg_tolerance")
                        iterations = st.number_input("Maximum iterations", min_value=1, value=100, step=1, key="bg_iterations")
                        st.caption("Second-order differences; convergence is required for application.")
                    else:
                        st.caption("decreasing=True · filter_order=2 · no extra smoothing")
                config = BackgroundConfig(method=method, domain="selected" if domain.startswith("Selected") else "full",
                    energy_min=minimum if domain.startswith("Selected") else None,
                    energy_max=maximum if domain.startswith("Selected") else None,
                    log10_lambda=log_lambda, tolerance=tolerance, max_iterations=iterations, half_window_mev=window)
            state.draft = config
            preview_key = (selected, state.fingerprints[selected], config)
            if state.preview_key != preview_key:
                state.preview, state.preview_key = None, None
            if state.applied_config is not None and state.applied_config != config:
                st.caption("Unapplied edits — main spectra and exports still use the applied settings.")
            if st.button("Preview", key="bg_preview", use_container_width=True):
                # A failed explicit refresh must never leave an older preview visible.
                state.preview, state.preview_key = None, None
                try:
                    state.preview = cached_background_fit(curve["energy"], curve["intensity"], config)
                    state.preview_key = preview_key
                except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    st.error(str(exc))
            if st.button("Apply to all spectra", key="bg_apply", use_container_width=True):
                with st.spinner("Fitting each selected spectrum independently…"):
                    failures = state.apply(curves, config, cached_background_fit)
                if failures:
                    st.error("Batch not applied. Previous applied results, if any, are retained.")
                    for identity, error in failures.items():
                        st.error(f"{labels[identity]}: {error}")
                else:
                    st.session_state["bg_next_signal"] = "Corrected"
                    st.rerun()
            if st.button("Reset background", key="bg_reset", use_container_width=True):
                state.reset()
                st.session_state["bg_next_signal"] = "Input"
                st.rerun()
        with right:
            result = state.preview
            if result is None:
                if method in ANALYTIC_MODELS:
                    requested = f"{len(segments)} segments spanning {minimum:.6g}–{maximum:.6g} meV"
                else:
                    requested = (f"{minimum:.6g}–{maximum:.6g} meV" if config.domain == "selected"
                                 else "full recorded spectrum")
                st.caption(f"Draft fit interval: {requested} · No matching preview. Press Preview to fit.")
            else:
                d = result.diagnostics
                if d.fit_segments:
                    seg_text = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in d.fit_segments)
                    st.caption(f"Segments [{seg_text}] meV · {d.fitted_bins} points fit · "
                               f"span {d.actual_bounds[0]:.6g}–{d.actual_bounds[1]:.6g} meV · solver: {d.status}")
                else:
                    st.caption(f"Requested: {d.requested_bounds[0]:.6g}–{d.requested_bounds[1]:.6g} meV · "
                               f"actual bin-aligned: {d.actual_bounds[0]:.6g}–{d.actual_bounds[1]:.6g} meV · "
                               f"{d.fitted_bins} fitted bins · solver: {d.status}")
            recorded = (float(curve["energy"][0]), float(curve["energy"][-1]))
            focus = (minimum, maximum) if config.domain == "selected" else recorded
            valid_focus = (np.isfinite(focus).all() and recorded[0] <= focus[0] < focus[1] <= recorded[1]
                           and np.any((curve["energy"] >= focus[0]) & (curve["energy"] <= focus[1])))
            focus_column, full_column = st.columns(2)
            with focus_column:
                focus_clicked = st.button("Focus fit interval", key="bg_focus_view", disabled=not valid_focus,
                    use_container_width=True, help="View only: zoom both energy axes and rescale intensities within the fit interval.")
            with full_column:
                full_clicked = st.button("Show full spectrum", key="bg_full_view", use_container_width=True,
                    help="View only: show the recorded energy domain and rescale intensities.")
            if focus_clicked or full_clicked:
                st.session_state["bg_preview_view"] = "fit" if focus_clicked else "full"
                st.session_state["bg_view_revision"] = st.session_state.get("bg_view_revision", 0) + 1
            view_bounds = (focus if valid_focus and st.session_state.get("bg_preview_view", "fit") == "fit"
                           else recorded)
            figure = preview_figure(curve, result, mode, view_bounds=view_bounds,
                                    view_revision=st.session_state.get("bg_view_revision", 0))
            st.caption("The initial view focuses on the fit interval and scales intensity within it. "
                       "Use Focus fit interval to restore that view after zooming. View controls never refit the baseline.")
            revision = st.session_state.get("background_render_revision", 0) + 1
            st.session_state["background_render_revision"] = revision
            # Refresh numerical traces independently of uirevision, which preserves zoom.
            figure.update_layout(datarevision=revision, meta=dict(eels_render_revision=revision))
            st.plotly_chart(figure, key="background_preview_plot",
                            use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
            axis_scaling = register_axis_scaling()
            axis_scaling(key="background_axis_scaling", height=0, data=dict(
                selector=".st-key-background_preview_plot .js-plotly-plot",
                viewport_key="eels.background.viewport"))
            st.caption("Scroll or pinch to zoom about the cursor · drag an axis to move it · "
                       "Shift + drag an axis to scale it "
                       "(up/right zooms in; down/left zooms out). Energy axes stay linked; "
                       "each intensity axis scales independently. These gestures do not refit the baseline.")
            if mode == "log10":
                st.caption("Nonpositive input, baseline, and corrected samples are masked for log10 display; "
                           "gaps are not connected. Zero has no log10 value, so the zero reference is hidden. "
                           "Signed linear values remain unchanged in fits and exports.")
                if result is not None and result.diagnostics.valid and not np.any(result.corrected > 0):
                    st.info("No positive corrected samples to show in log10. Select Linear to inspect the signed residuals.")
            if result is None:
                st.caption("Press Preview to fit this spectrum with the draft settings. The preview is independent of applied results.")
            else:
                d = result.diagnostics
                if not d.valid:
                    st.warning("This preview is invalid and cannot be applied.")
                if d.half_window_bins is not None:
                    st.caption(f"SNIP half-window: {d.half_window_bins} bins = {d.effective_half_window_mev:.6g} meV. No convergence statistic is used.")
                elif d.tol_history:
                    st.caption(f"Final tolerance: {d.tol_history[-1]:.4g} · {len(d.tol_history)} iterations")
            bounds = [(c["energy"][0], c["energy"][-1]) for c in curves] if config.domain == "full" else [(minimum, maximum)]
            if any(a <= 0 <= b for a, b in bounds):
                st.warning("The fitting domain contains zero energy: no zero-loss exclusion has been performed.")
            if config.domain == "full":
                st.caption("Each spectrum uses its own full recorded domain; coverage may differ.")
                st.dataframe([dict(Spectrum=c["label"], Minimum_meV=float(c["energy"][0]),
                                   Maximum_meV=float(c["energy"][-1])) for c in curves], hide_index=True)
            spacings = [c["energy"][1] - c["energy"][0] for c in curves if len(c["energy"]) > 1]
            if spacings and not np.allclose(spacings, spacings[0], rtol=1e-8, atol=0):
                st.warning("Energy spacings differ. The same numerical arPLS λ does not imply identical smoothing in energy units. Original grids are preserved without interpolation.")
            st.caption("A local interval should include useful background information around the peak. "
                       "Negative residuals are retained; both energy axes are linked. No extrapolation is drawn.")
            if state.applied_results:
                st.caption("Applied: " + config_caption(state.applied_config))
                st.dataframe([dict(Spectrum=c["label"], Requested_meV=str(state.applied_results[identity].diagnostics.requested_bounds),
                    Actual_meV=str(state.applied_results[identity].diagnostics.actual_bounds),
                    Bins=state.applied_results[identity].diagnostics.fitted_bins,
                    Status=state.applied_results[identity].diagnostics.status,
                    SNIP_bins=state.applied_results[identity].diagnostics.half_window_bins,
                    SNIP_meV=state.applied_results[identity].diagnostics.effective_half_window_mev)
                    for c, identity in zip(curves, identities)], hide_index=True)
                if state.applied_config.method in ANALYTIC_MODELS:
                    param_names = state.applied_results[identities[0]].diagnostics.model_param_names
                    st.caption("Fitted model coefficients (± approx. 95% CI, normal approximation — not "
                               "bit-identical to MATLAB's t-distribution-based nonlinear-regression CI):")
                    st.dataframe([dict(Spectrum=c["label"], **{
                            name: f"{state.applied_results[identity].diagnostics.model_coeffs[i]:.4g} ± "
                                  f"{state.applied_results[identity].diagnostics.model_coeff_errors[i]:.2g}"
                            for i, name in enumerate(param_names)})
                        for c, identity in zip(curves, identities)], hide_index=True)
    st.caption("An empirical baseline may remove genuine broad vibrational intensity. Subtraction is an analysis choice, "
               "not automatic identification of nonphysical background. Check parameter and interval sensitivity; "
               "this does not correct Fourier leakage or validate multiphonon intensity.")
