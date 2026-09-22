# EELS Studio

A local browser app for exploring vibrational EELS: select NumPy (`.npy`) or local Zarr diffraction scans, position a circular detector, compare spectra, subtract backgrounds, and build 2D/angle-resolved maps. Calculations follow the lab's STEM-EELS notebook workflow, reimplemented in `eels_core.py`.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501 (binds to localhost only). Once `.venv` exists, `./run_app.sh` starts it from anywhere.

## Usage

- **Load scans** — click **Browse folders** in the sidebar or type a data-folder path, then select files. The folder-selection window opens on the computer running the app; an in-app browser appears if a desktop window is unavailable. Cancelling keeps the current folder. Supports 3D `(energy, px, py)` and 6D `(dummy, energy, probe_x, probe_y, px, py)` arrays.
- **Detector** — set `px`, `py`, and `radius` under the detector image, or click the image to position the center. For 6D scans, pick probe positions in the sidebar.
- **Spectrum** — choose **Linear**, **log10**, or **Intensity × E²**, set axis limits, customize line styles, and export as CSV/NumPy/SVG/PNG. The E² option uses energy in meV and applies to plots and figure downloads; data exports retain unscaled linear intensities. Changing the display mode resets the vertical range while retaining the horizontal view. Drag to pan, Shift+drag an axis to scale it, scroll to zoom, and use the legend to hide/isolate curves.
- **Background subtraction** — in the **Background** tab, fit with arPLS, SNIP, or one of four analytic models, preview, then apply. Switch the Spectra view between Input/Corrected.
- **2D scan maps** — under **Visualization → 2D scan map**, request one or more energies to see detector-summed intensity across probe positions.
- **Angle-resolved EELS** — draw a rectangle on the diffraction image to sum a strip across one detector direction, producing an energy-vs-pixel map.

## NumPy and Zarr input

In **Data folder**, choose the folder containing your `.npy` files and Zarr array
directories (`.zarr` or `.zarray`). Both formats can be selected together under
**Files to compare**. Pasted paths may include surrounding quotes, whitespace,
or escaped underscores/spaces.
You can also paste a Zarr array directory directly, such as
`/scratch/project_465002371/zuxian/torched_TACAW/BTO_Ba-O_bussi/tacaw_results_stemeels/scan_gpu_tacaw.zarray`.

Local Zarr v2/v3 arrays support the same real numeric 3D/6D shapes, spectra,
diffraction previews, angle-resolved maps, background subtraction, and exports
as NumPy. For a Zarr group, select the array directory inside it. ZIP stores and
remote URLs are not supported. Existing environments need
`.venv/bin/python -m pip install -r requirements.txt` once, then an app restart.
Time step, sampling stride, and energy ordering still use the app controls;
Zarr attributes do not automatically override them.

NumPy reads remain bounded to small blocks. Zarr reads only intersecting chunks,
but each compressed chunk must be decoded in full. A per-operation cache retains
up to 64 MiB of decoded data, or one larger chunk, and is released when that
operation ends. Codec buffers need additional memory. Large chunks can therefore
be much slower and require more RAM than NumPy; the app flags chunks above
256 MiB. The example BTO array has 3.54 GiB uncompressed chunks, so use its `.npy`
copy or rechunk a separate Zarr copy for faster interactive viewing.

For 6D Zarr scans, selecting one probe also prepares the spectra of neighboring
probes in the same spatial chunk. Only the small integrated spectra and
normalization totals are cached; decoded chunks are released. Changing the
normalization, energy calibration, or broadening reuses these spectra. Changing
the detector geometry or source data requires fresh integration.

To browse arbitrary probe positions repeatedly, click **Prepare all Zarr probe
spectra** after choosing the detector. It reads every selected 6D Zarr scan once
and caches the reduced spectra. The initial pass can take several minutes for
large scans. Preparation is retained in the running app's memory, not across
restarts. This option is shown when the selected scans fit within the cache's
256 spatial-tile entries. For the BTO 2D scan, each tile covers 3 × 3 probes;
100 tiles cover all 900 probe positions. Diffraction previews and angle-resolved
maps still read raw detector data; the prepared cache accelerates spectrum
extraction when adding probe positions.

Inspection reads metadata and checks file stats without loading array values.
Changes to metadata or nested chunk files invalidate cached results. Use completed
simulation outputs: reading while a simulation writes can combine different
stages of its output. No input data is modified.

## Tests

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest -q
```

Browser-gesture tests are optional and need Playwright plus a browser:

```bash
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install firefox
EELS_BROWSER=firefox .venv/bin/python -m pytest -q tests/test_axis_browser.py tests/test_angle_browser.py tests/test_background_browser.py
```

## License

MIT — see [LICENSE](LICENSE).
