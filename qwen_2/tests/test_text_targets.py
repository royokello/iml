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
        self.assertEqual(len(targets["high"]), 108)
        self.assertEqual(len(targets["low"]), 144)
        self.assertEqual(len(targets["bf16"]), 496)
        self.assertEqual(targets["sym-high"], ["model.language_model.embed_tokens.weight"])

    def test_vision_is_bf16(self):
        targets = _build_qwen2_text_encoder_targets()
        vision = {name for name in targets["bf16"] if name.startswith("model.visual.")}
        self.assertEqual(len(vision), 351)
        self.assertFalse(any(name.startswith("model.visual.") for name in targets["high"]))
        self.assertFalse(any(name.startswith("model.visual.") for name in targets["low"]))
