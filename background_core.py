"""Pure 1D background fitting. Inputs are linear spectra after broadening."""
from dataclasses import dataclass, field
import hashlib
import json
import warnings
from typing import Callable

import numpy as np
import pybaselines
import scipy
from pybaselines import Baseline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_widths

from eels_core import curve_identity_key

MIN_FIT_BINS = 8
CONFIDENCE_Z = 1.959964  # normal-approximation 95% CI half-width multiplier
PROCESSING_ORDER = ["detector integration / optional full-probe normalization", "energy ordering",
                    "optional Gaussian broadening", "background estimation and subtraction",
                    "display transform"]


def _power(x, a0, a1, a2):
    return a0 * np.power(x, -a1) + a2


def _power0(x, a0, a1, a2):
    return a0 * np.power(x, -a1) + 0.0 * a2  # a2 is fitted but has no effect, matching the MATLAB model


def _exppoly(x, a0, a1, a2, a3, a5):
    return a0 * np.exp(-a1 * x + a2 * x**2 - a3 * x**3) + a5


def _pvoigt(x, g, a4, a2, h, w2, c0):
    return g * np.exp(a4 * x**4 - a2 * x**2) + h / (w2 + x**2) + c0


@dataclass(frozen=True)
class ModelSpec:
    func: callable
    params: tuple[str, ...]
    start: tuple[float, ...]
    bounds: tuple[tuple[float, ...], tuple[float, ...]]


# Ported from Vibrational-EELS_background_subtraction (MATLAB; Yan et al., Nature 645,
# 2025). Defaults for power/power0 mirror that toolkit's STO example (coeff_start,
# Upper_bound = -Lower_bound), which uses fit_model='power0' — listed first so it is also
# this app's default selection. The toolkit only ever demonstrates a 3-parameter model;
# exppoly/pVoigt defaults below are reasonable generic starting points, not validated for
# arbitrary datasets.
BACKGROUND_MODELS = {
    "power0": ModelSpec(_power0, ("a0", "a1", "a2"), (3e-2, 1.8, -2e-3),
                         ((-0.1, -5.0, -0.1), (0.1, 5.0, 0.1))),
    "power": ModelSpec(_power, ("a0", "a1", "a2"), (3e-2, 1.8, -2e-3),
                        ((-0.1, -5.0, -0.1), (0.1, 5.0, 0.1))),
    "exppoly": ModelSpec(_exppoly, ("a0", "a1", "a2", "a3", "a5"), (0.1, 0.5, 0.05, 0.005, 0.0),
                          ((-1.0, -5.0, -1.0, -1.0, -1.0), (1.0, 5.0, 1.0, 1.0, 1.0))),
    "pVoigt": ModelSpec(_pvoigt, ("g", "a4", "a2", "h", "w2", "c0"), (0.1, 0.01, 0.5, 0.1, 1.0, 0.0),
                         ((-1.0, -1.0, -5.0, -1.0, 1e-6, -1.0), (1.0, 1.0, 5.0, 1.0, 50.0, 1.0))),
}
ANALYTIC_MODELS = tuple(BACKGROUND_MODELS)


@dataclass(frozen=True)
class BackgroundConfig:
    method: str = "arPLS"
    domain: str = "selected"
    energy_min: float | None = None
    energy_max: float | None = None
    log10_lambda: float = 5.0
    tolerance: float = 1e-3
    max_iterations: int = 100
    half_window_mev: float = 10.0
    # ANALYTIC_MODELS only: 2-4 flanking energy segments (meV) that exclude the peak, and
    # the corresponding model's start point / bounds. domain/energy_min/energy_max are
    # still set (to the overall segment span) so the view's existing focus/zero-energy
    # logic keeps working, but fit_background ignores them for these methods.
    segments: tuple[tuple[float, float], ...] = ()
    model_start: tuple[float, ...] = ()
    model_lower: tuple[float, ...] = ()
    model_upper: tuple[float, ...] = ()
    energy_factor: float = 40.0


@dataclass(frozen=True)
class FitDiagnostics:
    requested_bounds: tuple[float, float]
    actual_bounds: tuple[float, float]
    fitted_bins: int
    spacing_mev: float
    status: str
    valid: bool
    contains_zero: bool
    tol_history: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()
    half_window_bins: int | None = None
    effective_half_window_mev: float | None = None
    package_version: str = pybaselines.__version__
    fit_segments: tuple[tuple[float, float], ...] = ()
    model_param_names: tuple[str, ...] = ()
    model_coeffs: tuple[float, ...] = ()
    model_coeff_errors: tuple[float, ...] = ()


@dataclass(frozen=True)
class BackgroundResult:
    energy: np.ndarray
    input: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    validity_mask: np.ndarray
    diagnostics: FitDiagnostics


def fit_background(energy, intensity, config: BackgroundConfig) -> BackgroundResult:
    """Fit one contiguous domain, never extrapolate or change the input grid.

    Selected limits must be covered in full. End bins are selected inward
    (energy >= minimum, energy <= maximum); at least eight bins are required.
    SNIP converts meV to the nearest integer bin, with half ties rounded up.
    """
    if np.iscomplexobj(energy) or np.iscomplexobj(intensity):
        raise ValueError("Energy and intensity must be real")
    x, y = np.array(energy, dtype=float, copy=True), np.array(intensity, dtype=float, copy=True)
    if x.ndim != 1 or y.shape != x.shape or x.size < MIN_FIT_BINS:
        raise ValueError("Require matching 1D arrays with at least eight fitted bins")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Energy and intensity must be finite")
    steps = np.diff(x)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        raise ValueError("Require a strictly increasing, uniform energy axis")
    if config.method in ANALYTIC_MODELS:
        return _fit_analytic_background(x, y, steps[0], config)
    if config.method not in ("arPLS", "SNIP") or config.domain not in ("selected", "full"):
        raise ValueError("Unknown background method or fit domain")
    if not np.isfinite(config.log10_lambda) or not 2 <= config.log10_lambda <= 10:
        raise ValueError("log10(lambda) must be between 2 and 10")
    if not np.isfinite(config.tolerance) or not 0 < config.tolerance < 1:
        raise ValueError("Tolerance must be finite and between zero and one")
    if isinstance(config.max_iterations, bool) or not isinstance(config.max_iterations, (int, np.integer)) or config.max_iterations < 1:
        raise ValueError("Maximum iterations must be a positive integer")
    if not np.isfinite(config.half_window_mev) or config.half_window_mev <= 0:
        raise ValueError("SNIP half-window must be finite and positive")
    if config.domain == "selected":
        bounds = (config.energy_min, config.energy_max)
        if any(v is None for v in bounds) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("Fit bounds must be finite and strictly ordered")
        if bounds[0] < x[0] or bounds[1] > x[-1]:
            raise ValueError(f"Requested interval {bounds} is not covered by [{x[0]:g}, {x[-1]:g}] meV")
    else:
        bounds = (float(x[0]), float(x[-1]))
    mask = (x >= bounds[0]) & (x <= bounds[1])
    if np.count_nonzero(mask) < MIN_FIT_BINS:
        raise ValueError("The fitting interval must contain at least eight fitted bins")
    fitted_x, fitted_y = x[mask], y[mask]
    half_bins = None
    if config.method == "SNIP":
        ratio = config.half_window_mev / steps[0]
        if not np.isfinite(ratio) or ratio > len(fitted_y):
            raise ValueError("SNIP half-window exceeds the fitting interval")
        half_bins = int(np.floor(ratio + 0.5))
        if not 1 <= half_bins <= (len(fitted_y) - 1) // 2:
            raise ValueError(f"SNIP half-window converts to {half_bins} bins; require 1–{(len(fitted_y) - 1) // 2}. No clamping is performed.")
    history = ()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if config.method == "arPLS" and np.all(fitted_y == fitted_y[0]):
            baseline = np.full_like(fitted_y, fitted_y[0])
            status, valid = "exact constant input", True
        elif config.method == "arPLS":
            baseline, params = Baseline(x_data=fitted_x, check_finite=True).arpls(
                fitted_y, lam=10.0 ** config.log10_lambda, diff_order=2,
                tol=config.tolerance, max_iter=config.max_iterations)
            history = tuple(float(v) for v in params["tol_history"])
            valid = bool(history and np.isfinite(history).all() and history[-1] <= config.tolerance)
            status = "converged" if valid else "not converged"
        else:
            baseline, _ = Baseline(x_data=fitted_x, check_finite=True).snip(
                fitted_y, max_half_window=half_bins, decreasing=True, filter_order=2,
                smooth_half_window=None)
            status, valid = "completed", True
        residual = fitted_y - baseline
    messages = tuple(str(w.message) for w in caught)
    if messages:
        status, valid = "library warning: " + "; ".join(messages), False
    if np.shape(baseline) != fitted_y.shape or not np.isfinite(baseline).all() or not np.isfinite(residual).all():
        status, valid = "numerically invalid output", False
    full_baseline, corrected = np.full_like(y, np.nan), np.full_like(y, np.nan)
    if valid:
        full_baseline[mask], corrected[mask] = baseline, residual
    else:
        mask[:] = False
    diagnostics = FitDiagnostics(tuple(float(v) for v in bounds),
        (float(fitted_x[0]), float(fitted_x[-1])), len(fitted_y), float(steps[0]), status, valid,
        bool(bounds[0] <= 0 <= bounds[1]), history, messages, half_bins,
        float(half_bins * steps[0]) if half_bins is not None else None)
    for array in (x, y, full_baseline, corrected, mask):
        array.setflags(write=False)
    return BackgroundResult(x, y, full_baseline, corrected, mask, diagnostics)


def _fit_analytic_background(x, y, spacing, config: BackgroundConfig) -> BackgroundResult:
    """Fit an analytic model on flanking segments; subtract it across their full span.

    Unlike the single-domain arPLS/SNIP path, the model is fit only on the union of
    `config.segments` (which should exclude the peak of interest) but evaluated and
    subtracted across the whole span from the first segment's start to the last
    segment's end — including the peak region between them, which is the point of an
    empirical-background subtraction. energy_min/energy_max/domain are not consulted.
    """
    spec = BACKGROUND_MODELS[config.method]
    n_params = len(spec.params)
    segments = config.segments
    if not 2 <= len(segments) <= 4:
        raise ValueError("Provide 2 to 4 fit segments for this model")
    channel_mask = np.zeros_like(x, dtype=bool)
    for lo, hi in segments:
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError(f"Invalid fit segment [{lo}, {hi}]")
        if lo < x[0] or hi > x[-1]:
            raise ValueError(f"Segment [{lo:g}, {hi:g}] meV is not covered by [{x[0]:g}, {x[-1]:g}] meV")
        channel_mask |= (x >= lo) & (x <= hi)
    min_points = max(MIN_FIT_BINS, n_params)
    if np.count_nonzero(channel_mask) < min_points:
        raise ValueError(f"Fit segments must cover at least {min_points} bins total")
    if len(config.model_start) != n_params or len(config.model_lower) != n_params or len(config.model_upper) != n_params:
        raise ValueError(f"start/bounds must each have {n_params} values for model {config.method!r}")
    if not np.isfinite(config.energy_factor) or config.energy_factor <= 0:
        raise ValueError("energy_factor must be finite and positive")
    span = (min(lo for lo, _ in segments), max(hi for _, hi in segments))
    span_mask = (x >= span[0]) & (x <= span[1])
    x_fit, y_fit = x[channel_mask] / config.energy_factor, y[channel_mask]
    status, valid = "converged", True
    popt = np.full(n_params, np.nan)
    errors = np.full(n_params, np.nan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            popt, pcov = curve_fit(spec.func, x_fit, y_fit, p0=config.model_start,
                                   bounds=(config.model_lower, config.model_upper),
                                   method="trf", maxfev=10000)
        except RuntimeError as exc:
            status, valid = f"solver failed: {exc}", False
    messages = tuple(str(w.message) for w in caught)
    if messages and valid:
        status, valid = "library warning: " + "; ".join(messages), False
    if valid:
        diagonal = np.diag(pcov)
        errors = (CONFIDENCE_Z * np.sqrt(diagonal) if np.isfinite(diagonal).all() and (diagonal >= 0).all()
                  else np.full(n_params, np.nan))
        baseline_span = spec.func(x[span_mask] / config.energy_factor, *popt)
        if not np.isfinite(baseline_span).all():
            status, valid = "numerically invalid output", False
    full_baseline, corrected = np.full_like(y, np.nan), np.full_like(y, np.nan)
    if valid:
        full_baseline[span_mask] = baseline_span
        corrected[span_mask] = y[span_mask] - baseline_span
    else:
        span_mask = np.zeros_like(x, dtype=bool)
    diagnostics = FitDiagnostics(
        requested_bounds=(float(span[0]), float(span[1])), actual_bounds=(float(span[0]), float(span[1])),
        fitted_bins=int(np.count_nonzero(channel_mask)), spacing_mev=float(spacing), status=status, valid=valid,
        contains_zero=bool(span[0] <= 0 <= span[1]), package_version=f"scipy {scipy.__version__}",
        fit_segments=tuple((float(lo), float(hi)) for lo, hi in segments), model_param_names=spec.params,
        model_coeffs=tuple(float(v) for v in popt), model_coeff_errors=tuple(float(v) for v in errors))
    for array in (x, y, full_baseline, corrected, span_mask):
        array.setflags(write=False)
    return BackgroundResult(x, y, full_baseline, corrected, span_mask, diagnostics)


def default_analytic_segments(low, high, n_segments):
    """Evenly spaced flanking windows (~15% of [low, high] wide) leaving a gap for the peak."""
    span = high - low
    width = span * 0.15
    fractions = {2: (0.1, 0.9), 3: (0.1, 0.5, 0.9), 4: (0.1, 0.37, 0.63, 0.9)}[n_segments]
    return [(low + f * span - width / 2, low + f * span + width / 2) for f in fractions]


def auto_segments_from_peaks(energy, intensity, n_segments, low, high, *,
                             relative_prominence=0.15, rel_height=0.9):
    """Detect the peak(s) in [low, high] and return n_segments background-only windows.

    Peaks are located on log(intensity) with scipy.signal.find_peaks, so "prominence"
    (how much a peak stands out from its immediate surroundings) is measured
    relatively rather than in absolute intensity units — the default requires at
    least a ~15% local rise, and stays equally sensitive whether the peak sits near
    the top or the bottom of a steeply decaying background. Absolute-intensity
    prominence would miss a small peak riding on a much larger overall range (or,
    just as easily, flag ordinary curvature as a "peak") because a decaying
    power-law/exponential background can span orders of magnitude across the domain
    while the peak itself is a much smaller bump. The top three peaks by prominence
    are kept. Each kept peak's footprint comes from scipy.signal.peak_widths at
    rel_height (0.9: close to its full base), padded by a quarter of its own width on
    each side, then merged where footprints overlap. Segments then fill the widest
    remaining gaps between exclusions, repeatedly splitting the currently-widest gap
    in half if fewer gaps than n_segments remain — a starting point for the peak(s)
    actually present, not a substitute for checking the preview plot.
    """
    energy, intensity = np.asarray(energy, dtype=float), np.asarray(intensity, dtype=float)
    if energy.shape != intensity.shape or energy.ndim != 1:
        raise ValueError("energy and intensity must be matching 1D arrays")
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Domain bounds must be finite and strictly ordered")
    if not 2 <= n_segments <= 4:
        raise ValueError("n_segments must be between 2 and 4")
    if not np.isfinite(relative_prominence) or relative_prominence <= 0:
        raise ValueError("relative_prominence must be finite and positive")
    mask = (energy >= low) & (energy <= high)
    x, y = energy[mask], intensity[mask]
    if x.size < 20:
        raise ValueError("Domain has too few points to auto-detect a peak; place segments manually")
    if not np.isfinite(y).all():
        raise ValueError("Domain contains non-finite intensity; place segments manually")
    log_y = np.log(np.clip(y, np.finfo(float).tiny, None))
    peaks, properties = find_peaks(log_y, prominence=np.log1p(relative_prominence))
    if peaks.size == 0:
        raise ValueError(f"No peak found with at least a {relative_prominence:.0%} local rise; "
                         "place segments manually")
    top = np.argsort(properties["prominences"])[::-1][:3]
    peaks = peaks[top]
    widths, _, left_ips, right_ips = peak_widths(log_y, peaks, rel_height=rel_height)
    indices = np.arange(x.size)
    excluded = []
    for left_ip, right_ip in zip(left_ips, right_ips):
        lo, hi = float(np.interp(left_ip, indices, x)), float(np.interp(right_ip, indices, x))
        pad = 0.25 * max(hi - lo, np.finfo(float).eps)
        excluded.append((max(low, lo - pad), min(high, hi + pad)))
    excluded.sort()
    merged = [excluded[0]]
    for lo, hi in excluded[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    gaps, cursor = [], low
    for lo, hi in merged:
        if lo > cursor:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if high > cursor:
        gaps.append((cursor, high))
    if not gaps:
        raise ValueError("The detected peak(s) cover the whole domain; place segments manually")
    while len(gaps) < n_segments:
        gaps.sort(key=lambda g: g[1] - g[0])
        lo, hi = gaps.pop()
        mid = (lo + hi) / 2
        gaps.extend([(lo, mid), (mid, hi)])
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    return sorted(gaps[:n_segments])


def corrected_display(values, mode):
    """Keep signed residuals in linear mode; leave gaps for nonpositive log data."""
    values = np.asarray(values, dtype=float)
    if mode != "log10":
        return values.copy()
    output = np.full_like(values, np.nan)
    positive = np.isfinite(values) & (values > 0)
    output[positive] = np.log10(values[positive])
    return output


def initial_bounds(curves):
    """Prefer common positive coverage with eight bins in every curve."""
    low = max(float(c["energy"][0]) for c in curves)
    high = min(float(c["energy"][-1]) for c in curves)
    positives = [c["energy"][c["energy"] > max(0, low)] for c in curves]
    if all(len(p) for p in positives):
        positive_low = max(float(p[0]) for p in positives)
        if all(np.count_nonzero((c["energy"] >= positive_low) & (c["energy"] <= high)) >= MIN_FIT_BINS for c in curves):
            return positive_low, high
    return low, high  # If no valid common interval exists, explicit fitting validation explains it.


def input_fingerprints(curves, settings, revisions):
    """Labels, styles and view limits are deliberately excluded."""
    processing = {k: v for k, v in settings.items() if k not in ("plot", "probe_positions_xy")}
    result = {}
    for curve in curves:
        identity = curve_identity_key(curve)
        digest = hashlib.sha256(json.dumps([identity, processing, revisions[curve["path"]]], sort_keys=True).encode())
        for name in ("energy", "intensity"):
            digest.update(np.asarray(curve[name], dtype="<f8").tobytes())
        result[identity] = digest.hexdigest()
    return result


@dataclass
class BackgroundState:
    draft: BackgroundConfig | None = None
    preview: BackgroundResult | None = None
    preview_key: tuple | None = None
    applied_config: BackgroundConfig | None = None
    applied_results: dict[str, BackgroundResult] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    source_revisions: dict[str, tuple[int, int]] = field(default_factory=dict)
    signal: str = "Input"

    def sync_inputs(self, fingerprints):
        changed = self.fingerprints != fingerprints
        if changed:
            self.reset()
            self.fingerprints = fingerprints.copy()
        return changed

    def reset(self):
        self.preview = None
        self.preview_key = None
        self.applied_config = None
        self.applied_results = {}
        self.signal = "Input"

    def apply(self, curves, config, fitter: Callable = fit_background):
        results, failures = {}, {}
        for curve in curves:
            identity = curve_identity_key(curve)
            try:
                result = fitter(curve["energy"], curve["intensity"], config)
                if not result.diagnostics.valid:
                    failures[identity] = result.diagnostics.status
                else:
                    results[identity] = result
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures[identity] = str(exc)
        if not failures:
            self.applied_config, self.applied_results = config, results
            self.signal = "Corrected"
        return failures


def config_caption(config):
    if config.method in ANALYTIC_MODELS:
        segments = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in config.segments)
        return (f"{config.method} · segments [{segments}] meV (x/{config.energy_factor:g}) · "
               "fitted after optional Gaussian broadening")
    parameters = (f"log10(λ)={config.log10_lambda:g}, tol={config.tolerance:g}, max iterations={config.max_iterations}"
                  if config.method == "arPLS" else f"half-window={config.half_window_mev:g} meV")
    domain = "each full recorded domain" if config.domain == "full" else f"requested {config.energy_min:g}–{config.energy_max:g} meV"
    return f"{config.method} · {parameters} · {domain} · fitted after optional Gaussian broadening"
