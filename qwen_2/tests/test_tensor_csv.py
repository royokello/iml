import csv
import unittest
from pathlib import Path


def read_rows(path: str) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


class TensorCsvTest(unittest.TestCase):
    def test_text_encoder_counts(self):
        rows = read_rows("qwen_2/text_encoder/tensor.csv")
        names = {row["name"] for row in rows}
        self.assertEqual(len(rows), 750)
        self.assertIn("model.language_model.layers.35.mlp.down_proj.weight", names)
        self.assertIn("model.visual.blocks.26.attn.qkv.weight", names)
        self.assertNotIn("model.visual.blocks.27.attn.qkv.weight", names)
        self.assertEqual(sum(name.startswith("model.visual.") for name in names), 351)

    def test_denoiser_counts(self):
        rows = read_rows("qwen_2/denoiser/tensor.csv")
        names = {row["name"] for row in rows}
        self.assertEqual(len(rows), 297)
        self.assertIn("transformer_blocks.31.img_mlp.gate_layer.weight", names)
        self.assertIn("modulation.1.weight", names)
        self.assertNotIn("transformer_blocks.32.attn.to_q.weight", names)
