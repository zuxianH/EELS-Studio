"""Read bounded blocks from numeric .npy scans without a full-file memory map."""
import math
from pathlib import Path

import numpy as np


class ScanReader:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                read_header = np.lib.format.read_array_header_1_0
            elif version in ((2, 0), (3, 0)):
                # Supported scalar numeric headers are ASCII in both versions.
                read_header = np.lib.format.read_array_header_2_0
            else:
                raise ValueError(f"Unsupported .npy version: {version}")
            self.shape, self.fortran_order, self.dtype = read_header(stream)
            self.offset = stream.tell()
        if len(self.shape) not in (3, 6) or any(n <= 0 for n in self.shape):
            raise ValueError(
                "Expected (energy, px, py) or (dummy, energy, probe_x, probe_y, px, py); "
                f"got {self.shape}"
            )
        if self.dtype.kind not in "fiu":
            raise ValueError(f"Expected real numeric intensities; got {self.dtype}")
        if self.path.stat().st_size < self.offset + math.prod(self.shape) * self.dtype.itemsize:
            raise ValueError("Truncated .npy file: data is smaller than its header declares")
        self.order = "F" if self.fortran_order else "C"
        self.fast_axes = list(range(len(self.shape)))
        if not self.fortran_order:
            self.fast_axes.reverse()
        self.strides = [0] * len(self.shape)
        stride = self.dtype.itemsize
        for axis in self.fast_axes:
            self.strides[axis] = stride
            stride *= self.shape[axis]

    def __enter__(self):
        self.stream = self.path.open("rb")
        return self

    def __exit__(self, *args):
        self.stream.close()

    def read(self, key):
        """Read an internal selection of integers and unit-step slices.

        Coalesce contiguous axes into file reads. C-order detector rows take
        one read; Fortran-order inputs use the same offsets in their own order.
        Only the selected block is allocated. Callers bound the block size.
        """
        if len(key) != len(self.shape):
            raise ValueError("Provide one index or slice for each scan axis")
        starts, lengths, squeeze = [], [], []
        for axis, (index, size) in enumerate(zip(key, self.shape)):
            if isinstance(index, slice):
                start, stop, step = index.indices(size)
                if step != 1 or stop <= start:
                    raise ValueError("Expected nonempty, unit-step slices")
                length = stop - start
            else:
                if not isinstance(index, (int, np.integer)) or not 0 <= index < size:
                    raise ValueError(f"Index {index} is outside axis {axis} of size {size}")
                start, length = int(index), 1
                squeeze.append(axis)
            starts.append(start)
            lengths.append(length)
        # Include the first partially selected axis, then stop: any slower
        # selected axis would introduce gaps in the contiguous file range.
        run_axes = []
        for axis in self.fast_axes:
            run_axes.append(axis)
            if lengths[axis] != self.shape[axis]:
                break
        loop_axes = [axis for axis in range(len(key)) if axis not in run_axes]
        run_shape = tuple(lengths[axis] for axis in sorted(run_axes))
        count = math.prod(run_shape)
        result = np.empty(lengths, dtype=self.dtype, order=self.order)
        for position in np.ndindex(*(lengths[axis] for axis in loop_axes)):
            file_indices = starts.copy()
            output_key = [slice(None)] * len(key)
            for axis, value in zip(loop_axes, position):
                file_indices[axis] += value
                output_key[axis] = value
            offset = self.offset + sum(i * stride for i, stride in zip(file_indices, self.strides))
            self.stream.seek(offset)
            values = np.fromfile(self.stream, dtype=self.dtype, count=count)
            if values.size != count:
                raise EOFError(f"Incomplete scan block in {self.path}")
            result[tuple(output_key)] = values.reshape(run_shape, order=self.order)
        return result.squeeze(axis=tuple(squeeze))
