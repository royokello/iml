import csv
import unittest
from pathlib import Path

from qwen_2.denoiser.targets import _build_qwen2_denoiser_target_tensors
from utils.quant.targets import build_mixed_target_config


class DenoiserTargetTest(unittest.TestCase):
    def test_targets_cover_every_checkpoint_tensor(self):
        with Path("qwen_2/denoiser/tensor.csv").open(newline="") as handle:
            names = {row["name"] for row in csv.DictReader(handle)}
        targets = _build_qwen2_denoiser_target_tensors()
        self.assertEqual(set().union(*targets.values()), names)
        self.assertEqual(len(targets["rank_1"]), 64)
        self.assertEqual(len(targets["rank_2"]), 128)
        self.assertEqual(len(targets["rank_3"]), 32)
        self.assertEqual(len(targets["bf16"]), 73)

    def test_shared_modulation_is_bf16(self):
        targets = _build_qwen2_denoiser_target_tensors()
        self.assertIn("modulation.1.weight", targets["bf16"])
        self.assertIn("norm_out.linear.weight", targets["bf16"])

    def test_every_block_is_covered(self):
        targets = _build_qwen2_denoiser_target_tensors()
        for index in range(32):
            prefix = f"transformer_blocks.{index}."
            self.assertIn(prefix + "attn.to_q.weight", targets["rank_1"])
            self.assertIn(prefix + "attn.to_out.0.weight", targets["rank_1"])
            self.assertIn(prefix + "attn.to_v.weight", targets["rank_2"])
            self.assertIn(prefix + "img_mlp.gate_layer.weight", targets["rank_2"])
            self.assertIn(prefix + "img_mlp.out.weight", targets["rank_2"])
            self.assertIn(prefix + "attn.to_k.weight", targets["rank_3"])

    def test_quantized_split_matches_the_written_checkpoint(self):
        targets = _build_qwen2_denoiser_target_tensors()
        config = build_mixed_target_config("q6-mini", targets)
        self.assertEqual(
            {method: len(names) for method, names in config.items()},
            {"q6": 64, "q5": 160, "bf16": 73},
        )
