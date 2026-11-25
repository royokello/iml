import csv
import unittest
from pathlib import Path

from qwen_2.text_encoder.target import (
    TEXT_ENCODER_EXCLUSION_PREFIX,
    _build_qwen2_text_encoder_targets,
)


def read_names(path: str) -> set[str]:
    with Path(path).open(newline="") as handle:
        return {row["name"] for row in csv.DictReader(handle)}


class TextTargetTest(unittest.TestCase):
    def test_targets_cover_every_checkpoint_tensor(self):
        names = read_names("qwen_2/text_encoder/tensor.csv")
        targets = _build_qwen2_text_encoder_targets()
        covered = set().union(*targets.values())
        excluded = {name for name in names if name.startswith(TEXT_ENCODER_EXCLUSION_PREFIX)}
        self.assertEqual(covered | excluded, names)
        self.assertEqual(covered & excluded, set())

    def test_target_counts(self):
        targets = _build_qwen2_text_encoder_targets()
        self.assertEqual(len(targets["rank_1"]), 72)
        self.assertEqual(len(targets["rank_2"]), 144)
        self.assertEqual(len(targets["rank_3"]), 36)
        self.assertEqual(len(targets["bf16"]), 496)
        self.assertEqual(
            targets["q8"],
            ["model.language_model.embed_tokens.weight"],
        )

    def test_residual_writes_are_high(self):
        targets = _build_qwen2_text_encoder_targets()
        self.assertIn("model.language_model.layers.0.self_attn.v_proj.weight", targets["rank_1"])
        self.assertIn("model.language_model.layers.0.mlp.down_proj.weight", targets["rank_1"])

    def test_attention_and_ffn_inputs_are_medium(self):
        targets = _build_qwen2_text_encoder_targets()
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
        ):
            self.assertIn(f"model.language_model.layers.0.{suffix}", targets["rank_2"])

    def test_attention_output_is_low(self):
        targets = _build_qwen2_text_encoder_targets()
        self.assertEqual(
            targets["rank_3"],
            [
                f"model.language_model.layers.{index}.self_attn.o_proj.weight"
                for index in range(36)
            ],
        )

    def test_vision_is_bf16(self):
        targets = _build_qwen2_text_encoder_targets()
        vision = {name for name in targets["bf16"] if name.startswith("model.visual.")}
        self.assertEqual(len(vision), 351)
        self.assertFalse(any(name.startswith("model.visual.") for name in targets["rank_1"]))
        self.assertFalse(any(name.startswith("model.visual.") for name in targets["rank_2"]))
        self.assertFalse(any(name.startswith("model.visual.") for name in targets["rank_3"]))
