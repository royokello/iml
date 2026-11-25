import csv
import unittest
from pathlib import Path

from qwen_2.denoiser.targets import _build_qwen2_denoiser_target_tensors


class DenoiserTargetTest(unittest.TestCase):
    def test_targets_cover_every_checkpoint_tensor(self):
        with Path("qwen_2/denoiser/tensor.csv").open(newline="") as handle:
            names = {row["name"] for row in csv.DictReader(handle)}
        targets = _build_qwen2_denoiser_target_tensors()
        self.assertEqual(set().union(*targets.values()), names)
        self.assertEqual(len(targets["high"]), 96)
        self.assertEqual(len(targets["low"]), 128)
        self.assertEqual(len(targets["bf16"]), 73)

    def test_shared_modulation_is_bf16(self):
        targets = _build_qwen2_denoiser_target_tensors()
        self.assertIn("modulation.1.weight", targets["bf16"])

    def test_every_block_is_covered(self):
        targets = _build_qwen2_denoiser_target_tensors()
        for index in range(32):
            self.assertIn(f"transformer_blocks.{index}.attn.to_q.weight", targets["low"])
            self.assertIn(f"transformer_blocks.{index}.img_mlp.out.weight", targets["high"])
