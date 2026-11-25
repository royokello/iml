import unittest

from qwen_2.config import (
    BASE_SHIFT,
    DEFAULT_REF_SIZE,
    DEFAULT_WEIGHT_DTYPE,
    MAX_SHIFT,
    TEXT_SYSTEM_PROMPT,
    TEXT_TEMPLATE_T2I,
    TURBO_SIGMAS,
    build_edit_text,
    calculate_dimensions,
    calculate_shift,
    image_token_count,
    parse_quant_method,
    resolve_model_root,
    resolve_outputs_dir,
    resolve_sizes,
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
        for method in ("q6-mini", "q5-mini", "q8-mini"):
            with self.subTest(method=method):
                self.assertEqual(
                    parse_quant_method(method), (method, DEFAULT_WEIGHT_DTYPE)
                )

    def test_none_means_the_unquantized_checkpoint(self):
        self.assertEqual(parse_quant_method(None), (None, DEFAULT_WEIGHT_DTYPE))

    def test_edit_text_repeats_one_slot_per_image(self):
        text = build_edit_text("make it red", 2)
        self.assertEqual(text.count("<|image_pad|>"), 2)
        self.assertEqual(text.count("<image1>"), 1)
        self.assertEqual(text.count("<image2>"), 1)
        self.assertIn("make it red", text)
        self.assertIn(TEXT_SYSTEM_PROMPT, text)

    def test_edit_text_requires_a_reference(self):
        with self.assertRaises(ValueError):
            build_edit_text("make it red", 0)

    def test_calculate_dimensions_uses_the_area_and_the_aspect(self):
        self.assertEqual(calculate_dimensions(512 * 512, 1.0), (512, 512))
        self.assertEqual(calculate_dimensions(512 * 512, 2.0), (736, 384))
        self.assertEqual(calculate_dimensions(512 * 512, 16 / 9), (672, 384))

    def test_resolve_sizes_without_references(self):
        sizes, output = resolve_sizes(None, None, None, DEFAULT_REF_SIZE)
        self.assertEqual(sizes, [])
        self.assertEqual(output, (1024, 1024))

    def test_resolve_sizes_uses_the_last_reference(self):
        sizes, output = resolve_sizes([(1024, 512), (512, 512)], None, None, 512)
        self.assertEqual(sizes, [(736, 384), (512, 512)])
        self.assertEqual(output, (512, 512))

    def test_resolve_sizes_keeps_an_explicit_axis(self):
        sizes, output = resolve_sizes([(1024, 512)], 640, None, 512)
        self.assertEqual(sizes, [(736, 384)])
        self.assertEqual(output, (640, 384))

    def test_none_means_the_unquantized_checkpoint(self):
        self.assertEqual(parse_quant_method(None), (None, DEFAULT_WEIGHT_DTYPE))
