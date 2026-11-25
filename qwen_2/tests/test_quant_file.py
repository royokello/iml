import os
import unittest
from pathlib import Path

from safetensors import safe_open


def quant_path(*parts: str) -> str:
    """Return a path under the model root.

    The model root is read from `QWEN2_ROOT`. It defaults to the repo folder so
    the paths match the documentation.
    """
    root = Path(os.environ.get("QWEN2_ROOT", "."))
    return str(root.joinpath("qwen_2", "model", *parts))


class QuantFileTest(unittest.TestCase):
    def test_text_encoder_quant_file(self):
        with safe_open(
            quant_path("text_encoder", "sym_high_mini_quant.safetensors"),
            framework="numpy",
        ) as handle:
            keys = set(handle.keys())
        self.assertNotIn("lm_head.weight", keys)
        self.assertIn("model.language_model.embed_tokens.sub_scales", keys)
        self.assertIn("model.language_model.layers.0.self_attn.q_proj.sub_scales", keys)
        self.assertIn("model.language_model.layers.35.mlp.down_proj.sub_scales", keys)
        self.assertEqual(sum(key.startswith("model.visual.") for key in keys), 351)

    def test_denoiser_quant_file(self):
        with safe_open(
            quant_path("denoiser", "turbo", "sym_med_nano_quant.safetensors"),
            framework="numpy",
        ) as handle:
            keys = set(handle.keys())
        self.assertIn("modulation.1.weight", keys)
        self.assertIn("transformer_blocks.0.attn.to_q.sub_scales", keys)
        self.assertIn("transformer_blocks.31.img_mlp.out.sub_scales", keys)
        self.assertNotIn("transformer_blocks.32.attn.to_q.weight", keys)
