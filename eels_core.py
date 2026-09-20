"""Notebook-compatible EELS extraction, independent of the user interface."""
from dataclasses import dataclass
from pathlib import Path
import json
import io
import csv

from scan_reader import ScanReader

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class ScanInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    mtime_ns: int
    size: int

    @property
    def canonical_shape(self):
        if len(self.shape) == 3:
            energy, px, py = self.shape
            return (1, energy, 1, 1, px, py)
        return self.shape


def inspect_scan(path):
    path = Path(path).expanduser().resolve()
    scan = ScanReader(path)
    stat = path.stat()
    return ScanInfo(str(path), scan.shape, str(scan.dtype), stat.st_mtime_ns, stat.st_size)


def energy_loss_axis_mev(chunk, timestep_fs=2.5, stride=3):
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("Energy bin count must be a positive integer")
    if not np.isfinite(timestep_fs) or timestep_fs <= 0:
        raise ValueError("Time step must be finite and positive")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("Stride must be a positive integer")
    frequency_thz = np.fft.fftshift(np.fft.fftfreq(chunk, timestep_fs * stride / 1000.0))
    return frequency_thz * 4.13566769692386


def nearest_energy_index(axis, requested_energy, *, unshifted=False):
    """Return (display index, raw storage index) of the axis bin nearest requested_energy.

    The display index locates the bin in the FFT-shifted axis as presented to
    the user; the raw index accounts for on-disk storage order when the
    underlying array uses unshifted FFT ordering.
    """
    axis = np.asarray(axis, dtype=float)
    index = int(np.argmin(np.abs(axis - requested_energy)))
    raw_index = int(np.fft.fftshift(np.arange(len(axis)))[index]) if unshifted else index
    return index, raw_index


def circular_detector_mask(shape, center=None, radius=20):
    if len(shape) != 2 or any(n < 1 for n in shape):
        raise ValueError("Detector plane must have two nonempty dimensions")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Detector radius must be finite and positive")
    center = center if center is not None else (shape[0] // 2, shape[1] // 2)
    if len(center) != 2 or not all(np.isfinite(v) and 0 <= v < n for v, n in zip(center, shape)):
        raise ValueError(f"Detector center {center} is outside shape {shape}")
    px, py = np.ogrid[:shape[0], :shape[1]]
    mask = (px - center[0]) ** 2 + (py - center[1]) ** 2 <= radius ** 2
    if not mask.any():
        raise ValueError("Detector contains no pixel centers; increase its radius")
    return mask


def detector_center(info, offset_px=0, offset_py=0):
    px, py = info.shape[-2:]
    return px // 2 + offset_px, py // 2 + offset_py


def detector_offsets_from_click(shape, px, py, selected_shapes):
    """Convert a preview pixel to shared offsets, checking every selected plane."""
    if not all(np.isfinite(v) and 0 <= v < n for v, n in zip((px, py), shape)):
        raise ValueError("Click inside the diffraction plane")
    offsets = tuple(int(np.floor(v + 0.5)) - n // 2 for v, n in zip((px, py), shape))
    if any(not 0 <= n // 2 + offset < n for plane in selected_shapes for n, offset in zip(plane, offsets)):
        raise ValueError("This position is outside another selected scan. Choose a point closer to the center or select scans with compatible detector planes.")
    return offsets


def _gaussian_sigma_bins(energy, sigma_mev):
    """Validate a uniform, increasing energy axis and convert sigma to bins."""
    steps = np.diff(energy)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-8, atol=0):
        raise ValueError("Gaussian broadening requires a uniformly spaced, increasing energy axis")
    if sigma_mev > energy[-1] - energy[0]:
        raise ValueError("Gaussian sigma must not exceed the recorded energy span")
    return sigma_mev / steps[0]


def gaussian_broaden_spectrum(energy_mev, intensity, sigma_mev=0.0):
    """Gaussian convolution of linear intensity on a uniform energy grid.

    Sigma is the standard deviation in meV. Reflecting boundaries preserve
    the sum of the recorded samples and do not wrap energy gain into loss.
    The kernel is truncated at four standard deviations.
    """
    energy = np.asarray(energy_mev, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if energy.ndim != 1 or values.shape != energy.shape or not energy.size:
        raise ValueError("Broadening requires matching nonempty 1D energy and intensity arrays")
    if not np.isfinite(energy).all() or not np.isfinite(values).all():
        raise ValueError("Broadening requires finite energy and intensity values")
    if not np.isfinite(sigma_mev) or sigma_mev < 0:
        raise ValueError("Gaussian sigma must be finite and nonnegative")
    if sigma_mev == 0:
        return values.copy()
    if len(energy) < 2:
        raise ValueError("Gaussian broadening requires at least two energy bins")
    return gaussian_filter1d(values, sigma=_gaussian_sigma_bins(energy, sigma_mev),
                             mode="reflect", truncate=4.0)


def _open_scan(info):
    stat = Path(info.path).stat()
    if (stat.st_mtime_ns, stat.st_size) != (info.mtime_ns, info.size):
        raise ValueError("File changed on disk; refresh the file list")
    return ScanReader(info.path)


def _validate_index(name, value, length):
    if not isinstance(value, (int, np.integer)) or not 0 <= value < length:
        raise ValueError(f"{name} index {value} is outside [0, {length})")


def _probe_key(info, sample, probe_x, probe_y):
    if len(info.shape) == 3:
        if (sample, probe_x, probe_y) != (0, 0, 0):
            raise ValueError("3D arrays contain only sample 0 and probe (0, 0)")
        return [slice(None)] * 3
    for name, value, length in zip(
        ("sample", "probe x", "probe y"), (sample, probe_x, probe_y),
        (info.shape[0], info.shape[2], info.shape[3]),
    ):
        _validate_index(name, value, length)
    return [sample, slice(None), probe_x, probe_y, slice(None), slice(None)]


def extract_spectrum(info, *, sample=0, probe_x=0, probe_y=0, radius=20,
                     offset_px=0, offset_py=0, normalize_3d=True):
    """Integrate bounded direct reads with float64 accumulation.

    Normalization divides by the sum of the entire selected (energy, px, py)
    block, exactly the normalization defined in STEM-EELS.ipynb.
    """
    mask = circular_detector_mask(info.shape[-2:], detector_center(info, offset_px, offset_py), radius)
    key = _probe_key(info, sample, probe_x, probe_y)
    energy_axis = 0 if len(info.shape) == 3 else 1
    n_energy = info.canonical_shape[1]
    spectrum = np.empty(n_energy, dtype=np.float64)
    total = 0.0
    plane_bytes = int(np.prod(info.shape[-2:])) * np.dtype(info.dtype).itemsize
    block_size = max(1, min(32, 16 * 2**20 // plane_bytes))
    with _open_scan(info) as reader:
        for start in range(0, n_energy, block_size):
            key[energy_axis] = slice(start, min(start + block_size, n_energy))
            part = reader.read(key)
            selected = part[:, mask]
            if normalize_3d:
                # This full-block check already covers the masked subset below.
                if not np.isfinite(part).all():
                    raise ValueError("Cannot normalize: probe data contains NaN or infinite intensities")
                total += part.sum(dtype=np.float64)
            elif not np.isfinite(selected).all():
                raise ValueError("Selected detector data contains NaN or infinite intensities")
            spectrum[start:start + len(part)] = selected.sum(axis=1, dtype=np.float64)
    if normalize_3d:
        if not np.isfinite(total) or total == 0:
            raise ValueError(f"Cannot normalize probe data with total intensity {total}")
        spectrum /= total
    if not np.isfinite(spectrum).all():
        raise ValueError("Integrated intensity overflowed; check the input data")
    return spectrum


def diffraction_pattern(info, energy_index, *, sample=0, probe_x=0, probe_y=0):
    _validate_index("Energy", energy_index, info.canonical_shape[1])
    key = _probe_key(info, sample, probe_x, probe_y)
    key[0 if len(info.shape) == 3 else 1] = energy_index
    with _open_scan(info) as reader:
        return reader.read(key).astype(float)


def detector_scan_map(info, energy_index, *, sample=0, radius=21,
                      offset_px=0, offset_py=0):
    """Return raw detector sums indexed by (probe_x, probe_y) at one energy.

    Read at most one probe-x row at a time, splitting larger rows into blocks
    of at most 16 MiB (or one detector plane if a plane exceeds that size).
    No spectrum normalization or broadening is applied.
    """
    if len(info.shape) != 6:
        raise ValueError("A 2D scan map requires a 6D scan")
    _validate_index("Sample", sample, info.shape[0])
    _validate_index("Energy", energy_index, info.shape[1])
    mask = circular_detector_mask(info.shape[-2:], detector_center(info, offset_px, offset_py), radius)
    n_x, n_y = info.shape[2:4]
    scan_map = np.empty((n_x, n_y), dtype=np.float64)
    plane_bytes = int(np.prod(info.shape[-2:])) * np.dtype(info.dtype).itemsize
    row_step = max(1, 16 * 2**20 // plane_bytes)
    with _open_scan(info) as reader:
        for x in range(n_x):
            for y in range(0, n_y, row_step):
                stop = min(y + row_step, n_y)
                row = reader.read((sample, energy_index, x, slice(y, stop), slice(None), slice(None)))
                values = np.sum(row, axis=(-2, -1), where=mask, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Selected detector data contains NaN, infinite, or overflowed intensities")
                scan_map[x, y:stop] = values
    return scan_map


def rectangle_from_plot(shape, x0, x1, y0, y1):
    """Convert Plotly coordinates to inclusive (px_min, px_max, py_min, py_max).

    Select pixel centers inside the drawn rectangle, clipped to the image.
    Plotly x is the array's py (column); Plotly y is px (row).
    """
    if not np.isfinite([x0, x1, y0, y1]).all():
        raise ValueError("Rectangle coordinates must be finite")
    bounds = []
    for a, b, count in ((y0, y1, shape[0]), (x0, x1, shape[1])):
        low = max(0, int(np.ceil(min(a, b))))
        high = min(count - 1, int(np.floor(max(a, b))))
        if low > high:
            raise ValueError("Draw a box containing at least one pixel center in each direction")
        bounds.extend((low, high))
    return tuple(bounds)


def extract_angle_resolved(info, bounds, *, retain_axis="py", sample=0,
                           probe_x=0, probe_y=0, normalize_3d=True):
    """Sum a rectangular strip, returning (pixel offsets, energy-by-pixel map).

    Bounds are inclusive (px_min, px_max, py_min, py_max). Retaining py sums
    rows (px), matching cropped.sum(axis=1) for an (energy, row, column) cube.
    Full-probe normalization uses the same denominator as extract_spectrum.
    """
    if retain_axis not in ("px", "py"):
        raise ValueError("Retained detector axis must be px or py")
    if len(bounds) != 4 or any(not isinstance(v, (int, np.integer)) for v in bounds):
        raise ValueError("Rectangle bounds must be four integer pixel indices")
    r0, r1, c0, c1 = bounds
    if not (0 <= r0 <= r1 < info.shape[-2] and 0 <= c0 <= c1 < info.shape[-1]):
        raise ValueError("Rectangle bounds must be ordered and inside the diffraction plane")
    retained = np.arange(c0, c1 + 1) if retain_axis == "py" else np.arange(r0, r1 + 1)
    pixels = retained - info.shape[-1 if retain_axis == "py" else -2] // 2
    key = _probe_key(info, sample, probe_x, probe_y)
    energy_axis = 0 if len(info.shape) == 3 else 1
    n_energy = info.canonical_shape[1]
    result = np.empty((n_energy, len(pixels)), dtype=np.float64)
    total = 0.0
    plane_bytes = int(np.prod(info.shape[-2:])) * np.dtype(info.dtype).itemsize
    block_size = max(1, min(32, 16 * 2**20 // plane_bytes))
    with _open_scan(info) as reader:
        for start in range(0, n_energy, block_size):
            key[energy_axis] = slice(start, min(start + block_size, n_energy))
            part = reader.read(key)
            crop = part[:, r0:r1 + 1, c0:c1 + 1]
            if normalize_3d:
                # This full-block check already covers the cropped rectangle below.
                if not np.isfinite(part).all():
                    raise ValueError("Cannot normalize: probe data contains NaN or infinite intensities")
                total += part.sum(dtype=np.float64)
            elif not np.isfinite(crop).all():
                raise ValueError("Selected rectangle contains NaN or infinite intensities")
            result[start:start + len(part)] = crop.sum(axis=1 if retain_axis == "py" else 2,
                                                      dtype=np.float64)
    if normalize_3d:
        if not np.isfinite(total) or total == 0:
            raise ValueError(f"Cannot normalize probe data with total intensity {total}")
        result /= total
    if not np.isfinite(result).all():
        raise ValueError("Integrated map overflowed; check the input data")
    return pixels, result


def process_angle_resolved(energy, intensity, *, unshifted=False, sigma_mev=0.0):
    """Order and correct a map, broadening only along energy, never along pixels."""
    energy = np.asarray(energy, dtype=float)
    values = np.array(intensity, dtype=float, copy=True)
    if energy.ndim != 1 or values.ndim != 2 or values.shape[0] != energy.size or not values.size:
        raise ValueError("Expected a nonempty (energy, pixel) map matching the energy axis")
    if not np.isfinite(values).all() or not np.isfinite(energy).all():
        raise ValueError("Map and energy values must be finite")
    if not np.isfinite(sigma_mev) or sigma_mev < 0:
        raise ValueError("Gaussian sigma must be finite and nonnegative")
    if unshifted:
        values = np.fft.fftshift(values, axes=0)
    if sigma_mev > 0:
        if len(energy) < 2:
            raise ValueError("Gaussian broadening requires at least two energy bins")
        # One vectorized call along the energy axis, instead of looping per pixel column.
        values = gaussian_filter1d(values, sigma=_gaussian_sigma_bins(energy, sigma_mev),
                                   axis=0, mode="reflect", truncate=4.0)
    if not np.isfinite(values).all():
        raise ValueError("Corrected map overflowed; check the input intensity")
    return values


def display_intensity(values, mode):
    if mode == "log10":
        return np.log10(np.clip(values, np.finfo(float).tiny, None))
    return values


def curve_identity_key(curve):
    """Stable identifier for a curve's source, independent of its display label."""
    return json.dumps([curve["path"], curve["sample"], curve["probe_x"], curve["probe_y"]])


def export_csv(curves):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["label", "source_file", "sample", "probe_x", "probe_y", "energy_meV", "intensity"])
    for curve in curves:
        for energy, intensity in zip(curve["energy"], curve["intensity"]):
            writer.writerow([curve["label"], curve["path"], curve["sample"], curve["probe_x"],
                             curve["probe_y"], energy, intensity])
    return output.getvalue().encode("utf-8")


def export_npz(curves, settings):
    arrays = {f"curve_{i:03d}": np.column_stack((c["energy"], c["intensity"])) for i, c in enumerate(curves)}
    metadata = {"settings": settings, "curves": [
        {k: v for k, v in c.items() if k not in ("energy", "intensity")} for c in curves
    ]}
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()
