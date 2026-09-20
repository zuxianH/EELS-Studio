"""Versioned exports keep background arrays separate from JSON curve metadata."""
from dataclasses import asdict
import csv
import io
import json

import numpy as np
import pybaselines
import scipy

from background_core import ANALYTIC_MODELS, BACKGROUND_MODELS, PROCESSING_ORDER
from eels_core import curve_identity_key

SCHEMA_VERSION = 1


def background_metadata(curves, settings, state, signal):
    config = state.applied_config
    if config.method in ANALYTIC_MODELS:
        parameters = dict(model=config.method, param_names=BACKGROUND_MODELS[config.method].params,
                          start=config.model_start, lower_bounds=config.model_lower,
                          upper_bounds=config.model_upper, segments_meV=config.segments,
                          energy_factor=config.energy_factor)
        package, package_version = "scipy.optimize.curve_fit", scipy.__version__
    else:
        parameters = (dict(lam=10.0 ** config.log10_lambda, diff_order=2, tol=config.tolerance,
                           max_iter=config.max_iterations) if config.method == "arPLS" else
                      dict(max_half_window_mev=config.half_window_mev, decreasing=True,
                           filter_order=2, smooth_half_window=None, transform="linear"))
        package, package_version = "pybaselines", pybaselines.__version__
    return dict(schema_version=SCHEMA_VERSION, exported_signal=signal,
        settings=dict(settings, processing_order=PROCESSING_ORDER),
        background=dict(configuration=asdict(config), algorithm_parameters=parameters, package=package,
                        package_version=package_version,
                        fitted_input_stage="linear, energy-unweighted intensity after optional Gaussian broadening"),
        curves=[dict(**{k: c[k] for k in ("label", "path", "sample", "probe_x", "probe_y", "style") if k in c},
                     source_revision=state.source_revisions.get(c["path"]),
                     input_fingerprint=state.fingerprints[curve_identity_key(c)],
                     diagnostics=asdict(state.applied_results[curve_identity_key(c)].diagnostics)) for c in curves])


def background_npz(curves, settings, state, signal):
    arrays = dict(schema_version=np.array(SCHEMA_VERSION),
                  metadata_json=np.array(json.dumps(background_metadata(curves, settings, state, signal))))
    for i, curve in enumerate(curves):
        key = f"curve_{i:03d}"
        result = state.applied_results[curve_identity_key(curve)]
        values = result.corrected if signal == "Corrected" else result.input
        arrays[key] = np.column_stack((result.energy, values))
        for name in ("energy", "input", "baseline", "corrected", "validity_mask"):
            arrays[f"{key}_{name}"] = getattr(result, name)
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def background_csv(curves, settings, state, signal, *, all_arrays=True):
    output = io.StringIO()
    writer = csv.writer(output)
    columns = ["input_intensity", "baseline", "corrected_intensity", "validity_mask"] if all_arrays else ["intensity"]
    writer.writerow(["curve_id", "label", "source_file", "sample", "probe_x", "probe_y", "energy_meV", "signal", *columns, "metadata_json"])
    metadata = json.dumps(background_metadata(curves, settings, state, signal))
    for i, c in enumerate(curves):
        r = state.applied_results[curve_identity_key(c)]
        for j, energy in enumerate(r.energy):
            values = ([r.input[j], r.baseline[j], r.corrected[j], bool(r.validity_mask[j])] if all_arrays
                      else [r.corrected[j] if signal == "Corrected" else r.input[j]])
            values = [v if np.isfinite(v) else "" for v in values]
            writer.writerow([f"curve_{i:03d}", c["label"], c["path"], c["sample"], c["probe_x"], c["probe_y"],
                             energy, signal, *values, metadata if i == j == 0 else ""])
    return output.getvalue().encode("utf-8")
