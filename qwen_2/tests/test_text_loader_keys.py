import unittest

from qwen_2.text_encoder.keys import _strip_model_key, _strip_text_key


class TextLoaderKeysTest(unittest.TestCase):
    def test_strip_text_key(self):
        self.assertEqual(
            _strip_text_key("model.language_model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight",
        )

    def test_strip_model_key(self):
        self.assertEqual(
            _strip_model_key("model.language_model.layers.0.self_attn.q_proj.weight"),
            "language_model.layers.0.self_attn.q_proj.weight",
        )
        self.assertEqual(
            _strip_model_key("model.visual.blocks.0.attn.qkv.weight"),
            "visual.blocks.0.attn.qkv.weight",
        )
