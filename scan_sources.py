"""Discover and read local NumPy files and Zarr arrays."""
from collections import OrderedDict
import hashlib
from itertools import product
import math
import os
import re
import socket
import stat
from pathlib import Path

import numpy as np

from scan_reader import ScanReader


def resolve_data_directory(value):
    """Accept plain paths and common quoted/escaped clipboard representations.

    Prefer a literal existing path, including names containing backslashes or
    whitespace. Only try cleaned alternatives when the literal path is missing.
    """
    raw = str(value)
    cleaned = raw.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'`":
        cleaned = cleaned[1:-1]
    # Markdown escapes underscores/dots; shell pastes often escape spaces.
    unescaped = re.sub(r"\\([_. ()-])", r"\1", cleaned)
    candidates = list(dict.fromkeys([raw, cleaned, unescaped]))
    for candidate in candidates:
        if not candidate.strip():
            continue
        path = Path(candidate).expanduser()
        try:
            mode = path.stat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            continue
        except PermissionError as exc:
            raise ValueError(f"Cannot access folder on {socket.gethostname()}: "
                             f"{str(path)!r}. Permission denied.") from exc
        if not stat.S_ISDIR(mode):
            raise ValueError(f"Expected a folder or Zarr array directory, got a file: {str(path)!r}")
        return path.resolve()
    shown = unescaped if unescaped.strip() else raw
    raise ValueError(f"Folder not found on {socket.gethostname()}: {shown!r}. "
                     "Enter a path accessible on the computer running EELS Studio.")


def is_zarr_store(path):
    path = Path(path)
    return path.is_dir() and any((path / name).is_file()
                                for name in ("zarr.json", ".zarray", ".zgroup"))


def discover_scans(directory):
    """Accept a containing folder or a Zarr array directory itself."""
    directory = Path(directory)
    if is_zarr_store(directory):
        return [directory]
    return sorted(path for path in directory.iterdir()
                  if (path.is_file() and path.suffix == ".npy")
                  or (path.is_dir() and (path.suffix in (".zarr", ".zarray")
                                        or is_zarr_store(path))))


def scan_revision(path):
    """Fingerprint metadata and chunk stats without reading array contents.

    A directory's own mtime does not change when an existing chunk is rewritten.
    Include every entry so rewrites, additions, and removals invalidate results.
    """
    path = Path(path)
    stat = path.stat()
    if not path.is_dir():
        return stat.st_mtime_ns, stat.st_size, ""
    digest = hashlib.sha256()
    newest, size = stat.st_mtime_ns, 0

    def fail(error):
        raise error

    for parent, dirs, files in os.walk(path, onerror=fail):
        dirs.sort()
        for name in [".", *sorted(files)]:
            entry = Path(parent) / name
            stat = entry.stat()
            digest.update(repr((str(entry.relative_to(path)), stat.st_mtime_ns,
                                stat.st_ctime_ns, stat.st_size)).encode())
            newest = max(newest, stat.st_mtime_ns)
            if name != ".":
                size += stat.st_size
    return newest, size, digest.hexdigest()


def open_scan_reader(path):
    return ZarrScanReader(path) if Path(path).is_dir() else ScanReader(path)


class ZarrScanReader:
    """Reuse decoded chunks during one extraction, never between reruns.

    Retain up to 64 MiB, or one larger chunk. Large compressed chunks inherently
    need extra decode memory; evict old chunks before allocating the next one.
    """
    cache_bytes = 64 * 2**20

    def __init__(self, path):
        self.path = Path(path)
        try:
            import zarr
        except ImportError as exc:
            raise ValueError("Zarr support requires zarr. Run .venv/bin/python -m pip "
                             "install -r requirements.txt from EELS_studio.") from exc
        try:
            self.array = zarr.open_array(str(self.path), mode="r")
            self.shape = tuple(self.array.shape)
            self.dtype = np.dtype(self.array.dtype)
            self.chunks = tuple(self.array.chunks)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise ValueError(f"Cannot open Zarr array: {exc}. Select an array directory, "
                             "not a Zarr group.") from exc
        if len(self.shape) not in (3, 6) or any(n <= 0 for n in self.shape):
            raise ValueError("Expected (energy, px, py) or "
                             "(dummy, energy, probe_x, probe_y, px, py); "
                             f"got {self.shape}")
        if self.dtype.kind not in "fiu":
            raise ValueError(f"Expected real numeric intensities; got {self.dtype}")
        self._cache = OrderedDict()
        self._cached_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._cache.clear()
        self._cached_bytes = 0

    def _chunk(self, coordinates):
        if coordinates in self._cache:
            self._cache.move_to_end(coordinates)
            return self._cache[coordinates]
        selection = tuple(slice(i * chunk, min((i + 1) * chunk, size))
                          for i, chunk, size in zip(coordinates, self.chunks, self.shape))
        nbytes = math.prod(s.stop - s.start for s in selection) * self.dtype.itemsize
        while self._cache and self._cached_bytes + nbytes > self.cache_bytes:
            _, old = self._cache.popitem(last=False)
            self._cached_bytes -= old.nbytes
            del old
        try:
            block = np.asarray(self.array[selection])
        except (ValueError, TypeError, RuntimeError) as exc:
            raise ValueError(f"Could not decode Zarr data in {self.path}: {exc}") from exc
        self._cache[coordinates] = block
        self._cached_bytes += block.nbytes
        return block

    def iter_chunks(self, key):
        """Yield whole decoded chunks intersecting a tuple of unit-step slices.

        Consumers must release each block before advancing to avoid retaining a
        previous oversized chunk while the next chunk is decoded.
        """
        if len(key) != len(self.shape):
            raise ValueError("Provide one slice for each scan axis")
        grid = []
        for selection, size, chunk in zip(key, self.shape, self.chunks):
            start, stop, step = selection.indices(size)
            if step != 1 or stop <= start:
                raise ValueError("Expected nonempty, unit-step slices")
            grid.append(range(start // chunk, (stop - 1) // chunk + 1))
        for coordinates in product(*grid):
            bounds = tuple(slice(i * chunk, min((i + 1) * chunk, size))
                           for i, chunk, size in zip(coordinates, self.chunks, self.shape))
            yield bounds, self._chunk(coordinates)

    def read(self, key):
        """Read integer/unit-step selections without materializing the full scan."""
        if len(key) != len(self.shape):
            raise ValueError("Provide one index or slice for each scan axis")
        starts, stops, squeeze = [], [], []
        for axis, (index, size) in enumerate(zip(key, self.shape)):
            if isinstance(index, slice):
                start, stop, step = index.indices(size)
                if step != 1 or stop <= start:
                    raise ValueError("Expected nonempty, unit-step slices")
            else:
                if not isinstance(index, (int, np.integer)) or not 0 <= index < size:
                    raise ValueError(f"Index {index} is outside axis {axis} of size {size}")
                start, stop = int(index), int(index) + 1
                squeeze.append(axis)
            starts.append(start)
            stops.append(stop)
        result = np.empty(tuple(b - a for a, b in zip(starts, stops)), dtype=self.dtype)
        grid = (range(a // c, (b - 1) // c + 1)
                for a, b, c in zip(starts, stops, self.chunks))
        for coordinates in product(*grid):
            source, target = [], []
            for i, chunk, start, stop in zip(coordinates, self.chunks, starts, stops):
                origin = i * chunk
                lo, hi = max(start, origin), min(stop, origin + chunk)
                source.append(slice(lo - origin, hi - origin))
                target.append(slice(lo - start, hi - start))
            # Do not keep a reference to the previous block across cache eviction.
            result[tuple(target)] = self._chunk(coordinates)[tuple(source)]
        return result.squeeze(axis=tuple(squeeze))
