import unittest

from utils.quant.name import QUANT_METHODS_BY_MAIN_BITS
from utils.quant.validators import quant_method_family, quant_method_mode


class QuantValidatorTest(unittest.TestCase):
    def test_every_method_maps_to_its_family_and_mode(self):
        expected = {
            "q2": ("affine", "low"),
            "q3": ("symmetric", "low"),
            "q4": ("affine", "med"),
            "q5": ("affine", "high"),
            "q6": ("symmetric", "med"),
            "q8": ("symmetric", "high"),
        }
        self.assertEqual(set(expected), set(QUANT_METHODS_BY_MAIN_BITS))
        for method, (family, mode) in expected.items():
            with self.subTest(method=method):
                self.assertEqual(quant_method_family(method), family)
                self.assertEqual(quant_method_mode(method), mode)

    def test_variant_names_are_rejected(self):
        for method in ("q6-mini", "q4-max", "q6-nano"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    quant_method_family(method)
                with self.assertRaises(ValueError):
                    quant_method_mode(method)

    def test_old_method_names_are_rejected(self):
        for method in ("sym-high", "sym-med", "aff-high", "sym-high-mini"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    quant_method_family(method)
                with self.assertRaises(ValueError):
                    quant_method_mode(method)


if __name__ == "__main__":
    unittest.main()
