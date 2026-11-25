import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from utils.safetensors_io import TensorFile


class TensorFileTest(unittest.TestCase):
    def test_header_lists_keys_shape_and_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(
                {
                    "a.weight": np.zeros((4, 8), dtype=np.float32),
                    "b.weight": np.arange(6, dtype=np.int32),
                },
                str(path),
                metadata={"component": "test"},
            )

            handle = TensorFile(path)

            self.assertEqual(handle.keys(), ["a.weight", "b.weight"])
            self.assertEqual(handle.metadata(), {"component": "test"})
            info = handle.info("a.weight")
            self.assertEqual(info["dtype"], "F32")
            self.assertEqual(tuple(info["shape"]), (4, 8))
            begin, end = info["data_offsets"]
            self.assertEqual(end - begin, 4 * 8 * 4)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            TensorFile("/tmp/does-not-exist.safetensors")

    def test_missing_tensor_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file({"a.weight": np.zeros((2, 2), dtype=np.float32)}, str(path))
            handle = TensorFile(path)
            with self.assertRaises(KeyError):
                handle.info("missing.weight")
