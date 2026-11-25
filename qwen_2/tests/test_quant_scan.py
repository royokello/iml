import tempfile
import unittest
from pathlib import Path

from image.generate.quant_scan import _extract_methods


class QuantScanTest(unittest.TestCase):
    def test_keeps_known_methods_and_drops_dead_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for stem in ("q8", "q6_mini", "q4_max", "sym_med_nano", "aff_high_nano", "fp16"):
                (directory / f"{stem}_quant.safetensors").touch()
            self.assertEqual(_extract_methods(directory), ["fp16", "q4-max", "q6-mini", "q8"])

    def test_missing_directory_returns_nothing(self):
        self.assertEqual(_extract_methods(Path("/nonexistent/iml-quant-scan")), [])


if __name__ == "__main__":
    unittest.main()
