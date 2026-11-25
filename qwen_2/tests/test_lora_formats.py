import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from qwen_2.lora.formats import parse_lora_keys, read_lora_file


def _write_lora(path: Path, keys: dict[str, tuple[int, int]], metadata: dict | None = None) -> None:
    tensors = {
        key: np.zeros(shape, dtype=np.float32) for key, shape in keys.items()
    }
    save_file(tensors, str(path), metadata=metadata)


class ParseLoraKeysTest(unittest.TestCase):
    def test_strips_the_diffusion_model_prefix(self):
        entries = parse_lora_keys(
            [
                "diffusion_model.transformer_blocks.7.attn.to_q.lora_A.weight",
                "diffusion_model.transformer_blocks.7.attn.to_q.lora_B.weight",
            ]
        )
        self.assertEqual(list(entries), ["transformer_blocks.7.attn.to_q"])

    def test_accepts_bare_native_keys(self):
        entries = parse_lora_keys(
            [
                "transformer_blocks.7.attn.to_q.lora_A.weight",
                "transformer_blocks.7.attn.to_q.lora_B.weight",
            ]
        )
        self.assertIn("transformer_blocks.7.attn.to_q", entries)

    def test_expands_the_fused_gate_up_into_two_modules(self):
        entries = parse_lora_keys(
            [
                "diffusion_model.transformer_blocks.3.img_mlp.gate_up.lora_A.weight",
                "diffusion_model.transformer_blocks.3.img_mlp.gate_up.lora_B.weight",
            ]
        )
        self.assertEqual(
            sorted(entries),
            [
                "transformer_blocks.3.img_mlp.gate_layer",
                "transformer_blocks.3.img_mlp.proj",
            ],
        )
        self.assertEqual(entries["transformer_blocks.3.img_mlp.gate_layer"].half, "gate")
        self.assertEqual(entries["transformer_blocks.3.img_mlp.proj"].half, "up")

    def test_names_every_unsupported_key(self):
        with self.assertRaises(ValueError) as caught:
            parse_lora_keys(
                [
                    "diffusion_model.transformer_blocks.0.txt_in.lora_A.weight",
                    "diffusion_model.transformer_blocks.0.txt_in.lora_B.weight",
                ]
            )
        self.assertIn("txt_in", str(caught.exception))

    def test_rejects_a_module_without_its_up_factor(self):
        with self.assertRaises(ValueError):
            parse_lora_keys(
                ["diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight"]
            )

    def test_expands_only_the_fused_key(self):
        entries = parse_lora_keys(
            [
                "diffusion_model.transformer_blocks.0.img_mlp.out.lora_A.weight",
                "diffusion_model.transformer_blocks.0.img_mlp.out.lora_B.weight",
            ]
        )
        self.assertEqual(list(entries), ["transformer_blocks.0.img_mlp.out"])
        self.assertIsNone(entries["transformer_blocks.0.img_mlp.out"].half)


class ReadLoraFileTest(unittest.TestCase):
    def test_defaults_alpha_to_the_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.safetensors"
            _write_lora(
                path,
                {
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight": (8, 16),
                    "diffusion_model.transformer_blocks.0.attn.to_q.lora_B.weight": (16, 8),
                },
            )
            lora = read_lora_file(path)
        self.assertEqual(lora.rank, 8)
        self.assertEqual(lora.alpha, 8.0)
        self.assertEqual(lora.format, "comfy")

    def test_reads_alpha_from_the_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.safetensors"
            _write_lora(
                path,
                {
                    "transformer_blocks.0.attn.to_q.lora_A.weight": (8, 16),
                    "transformer_blocks.0.attn.to_q.lora_B.weight": (16, 8),
                },
                metadata={"ss_network_alpha": "4"},
            )
            lora = read_lora_file(path)
        self.assertEqual(lora.alpha, 4.0)
        self.assertEqual(lora.format, "native")
