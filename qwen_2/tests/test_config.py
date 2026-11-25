import unittest

from qwen_2.config import (
    BASE_SHIFT,
    DEFAULT_WEIGHT_DTYPE,
    MAX_SHIFT,
    TEXT_TEMPLATE_T2I,
    TURBO_SIGMAS,
    calculate_shift,
    image_token_count,
    parse_quant_method,
    resolve_model_root,
    resolve_outputs_dir,
    round_to_multiple,
)


class ConfigTest(unittest.TestCase):
    def test_text_template_takes_the_prompt_keyword(self):
        text = TEXT_TEMPLATE_T2I.format(prompt="a red fox")
        self.assertIn("a red fox", text)
        self.assertNotIn("{prompt}", text)

    def test_turbo_sigmas_are_exact(self):
        self.assertEqual(
            TURBO_SIGMAS,
            (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568),
        )

    def test_calculate_shift_endpoints(self):
        self.assertAlmostEqual(calculate_shift(256), BASE_SHIFT, places=6)
        self.assertAlmostEqual(calculate_shift(8192), MAX_SHIFT, places=6)

    def test_round_to_multiple(self):
        self.assertEqual(round_to_multiple(1000), 1024)
        self.assertEqual(round_to_multiple(1024), 1024)

    def test_image_token_count(self):
        self.assertEqual(image_token_count(1024, 1024), 4096)

    def test_model_paths(self):
        root = "/tmp/iml"
        self.assertEqual(str(resolve_model_root(root)), "/tmp/iml/qwen_2/model")
        self.assertEqual(str(resolve_outputs_dir(root)), "/tmp/iml/qwen_2/outputs")

    def test_dtype_names_select_the_unquantized_checkpoint(self):
        for name in ("fp16", "bf16", "fp32"):
            with self.subTest(name=name):
                self.assertEqual(parse_quant_method(name), (None, name))
        self.assertEqual(parse_quant_method("BF16"), (None, "bf16"))

    def test_quant_method_names_pass_through(self):
        for method in ("sym-med-nano", "aff-high-nano", "sym-high-mini"):
            with self.subTest(method=method):
                self.assertEqual(
                    parse_quant_method(method), (method, DEFAULT_WEIGHT_DTYPE)
                )

    def test_none_means_the_unquantized_checkpoint(self):
        self.assertEqual(parse_quant_method(None), (None, DEFAULT_WEIGHT_DTYPE))
