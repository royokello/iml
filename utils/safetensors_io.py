"""Read safetensors tensors without the mmap-backed storage slice.

`safe_open(...).get_tensor()` on Windows faults with an access violation when a
tensor crosses the 2 GiB offset mark in the file. This reader parses the header
itself and copies the tensor bytes into a private buffer, so large files read
the same way they do on Linux.

The header API does not import torch, so it is testable without torch.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

__all__ = ["TensorFile"]

_DTYPE_NAMES = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


class TensorFile:
    """A safetensors file whose tensors are read into plain CPU memory.

    Args:
        path: The safetensors file.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the header is malformed or a dtype is unknown.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Tensor file not found: {self.path}")

        with self.path.open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            self._header = json.loads(handle.read(header_length))

        self._metadata = self._header.pop("__metadata__", None) or {}
        self._data_start = 8 + header_length
        for name, info in self._header.items():
            if info["dtype"] not in _DTYPE_NAMES:
                raise ValueError(f"Unsupported dtype {info['dtype']!r} for tensor {name!r}.")

    def keys(self) -> list[str]:
        """Return the tensor names in file order."""
        return list(self._header)

    def metadata(self) -> dict:
        """Return the file metadata."""
        return dict(self._metadata)

    def info(self, name: str) -> dict:
        """Return the stored dtype, shape, and byte offsets of a tensor."""
        if name not in self._header:
            raise KeyError(f"Tensor not found: {name}")
        return dict(self._header[name])

    def read(self, name: str):
        """Read one tensor into a private buffer.

        Args:
            name: The tensor name.

        Returns:
            The torch tensor with the stored shape and dtype.

        Raises:
            KeyError: If the tensor is missing.
            EOFError: If the file ends before the tensor data does.
        """
        import torch

        info = self._header[name]
        dtype = getattr(torch, _DTYPE_NAMES[info["dtype"]])
        begin, end = info["data_offsets"]
        byte_count = end - begin
        if byte_count == 0:
            return torch.empty(info["shape"], dtype=dtype)

        buffer = bytearray(byte_count)
        with self.path.open("rb") as handle:
            handle.seek(self._data_start + begin)
            view = memoryview(buffer)
            while view:
                read = handle.readinto(view)
                if not read:
                    raise EOFError(f"File ended while reading {name!r} from {self.path}.")
                view = view[read:]

        raw = torch.frombuffer(buffer, dtype=torch.uint8)
        if dtype is not torch.uint8:
            raw = raw.view(dtype)
        return raw.reshape(info["shape"])
