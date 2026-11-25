import unittest

from utils.quant.name import mixed_quant_methods
from utils.quant.targets import build_mixed_target_config

from anima.denoiser.targets import _build_anima_denoiser_target_tensors
from anima.text_encoder.targets import _build_anima_text_encoder_targets
from flux2.denoiser.targets import _build_flux2_denoiser_target_tensors
from flux2.text_encoder.target import _build_target_tensors as _build_flux2_text_target_tensors
from flux3.denoiser.targets import _build_flux3_denoiser_target_tensors
from flux3.text_encoder.target import _build_flux3_text_encoder_target_tensors
from ideogram.denoiser.targets import _build_ideogram_denoiser_target_tensors
from ideogram.text_encoder.target import _build_ideogram_text_encoder_targets
from krea_2.denoiser.targets import _build_krea2_denoiser_target_tensors
from krea_2.text_encoder.target import _build_krea2_text_encoder_targets
from qwen_2.denoiser.targets import _build_qwen2_denoiser_target_tensors
from qwen_2.text_encoder.target import _build_qwen2_text_encoder_targets
from sdxl.denoiser.targets import build_sdxl_denoiser_target_tensors
from wan22.quant.targets import (
    _build_wan22_denoiser_mixed_target_tensors,
    _build_wan22_text_encoder_mixed_target_tensors,
)


def _builders() -> dict[str, dict[str, list[str]]]:
    return {
        "anima/denoiser": _build_anima_denoiser_target_tensors(),
        "anima/text_encoder": _build_anima_text_encoder_targets(),
        "flux2/denoiser/4b": _build_flux2_denoiser_target_tensors("4b"),
        "flux2/text_encoder": _build_flux2_text_target_tensors(),
        "flux3/denoiser": _build_flux3_denoiser_target_tensors(),
        "flux3/text_encoder": _build_flux3_text_encoder_target_tensors(),
        "ideogram/denoiser": _build_ideogram_denoiser_target_tensors(),
        "ideogram/text_encoder": _build_ideogram_text_encoder_targets(),
        "krea_2/denoiser": _build_krea2_denoiser_target_tensors(),
        "krea_2/text_encoder": _build_krea2_text_encoder_targets(),
        "qwen_2/denoiser": _build_qwen2_denoiser_target_tensors(),
        "qwen_2/text_encoder": _build_qwen2_text_encoder_targets(),
        "sdxl/denoiser": build_sdxl_denoiser_target_tensors(
            [
                "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
                "down_blocks.0.resnets.0.time_emb_proj.weight",
            ]
        ),
        "wan22/denoiser": _build_wan22_denoiser_mixed_target_tensors(),
        "wan22/text_encoder": _build_wan22_text_encoder_mixed_target_tensors(),
    }


class RankKeyTest(unittest.TestCase):
    def test_every_builder_uses_rank_or_method_keys(self):
        allowed = set(mixed_quant_methods()) | {"fp32", "fp16", "bf16"}
        for name, targets in _builders().items():
            with self.subTest(builder=name):
                for key in targets:
                    if key.startswith("rank_"):
                        self.assertTrue(key[5:].isdigit(), f"{name} has a malformed rank key {key!r}")
                    else:
                        self.assertIn(key, allowed, f"{name} has an unknown group key {key!r}")

    def test_every_builder_starts_at_rank_1_without_gaps(self):
        for name, targets in _builders().items():
            with self.subTest(builder=name):
                ranks = sorted(int(key[5:]) for key in targets if key.startswith("rank_"))
                self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_every_builder_accepts_every_method(self):
        for name, targets in _builders().items():
            for method in mixed_quant_methods():
                with self.subTest(builder=name, method=method):
                    config = build_mixed_target_config(method, targets)
                    for names in config.values():
                        for tensor in names:
                            self.assertIn(
                                tensor,
                                {item for group in targets.values() for item in group},
                                f"{name} produced an unknown tensor {tensor!r}",
                            )
