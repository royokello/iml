import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.numpy import save_file

from qwen_2.lora.inject import apply_loras

_DIM = 8
_MLP = 24


class _StubBlock(nn.Module):
    """One block with the seven LoRA target names and small widths."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(_DIM, _DIM, bias=False)
        self.attn.to_k = nn.Linear(_DIM, _DIM, bias=False)
        self.attn.to_v = nn.Linear(_DIM, _DIM, bias=False)
        self.attn.to_out = nn.ModuleList([nn.Linear(_DIM, _DIM, bias=False)])
        self.img_mlp = nn.Module()
        self.img_mlp.gate_layer = nn.Linear(_DIM, _MLP, bias=False)
        self.img_mlp.proj = nn.Linear(_DIM, _MLP, bias=False)
        self.img_mlp.out = nn.Linear(_MLP, _DIM, bias=False)


class _Stub(nn.Module):
    def __init__(self, blocks: int = 1) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_StubBlock() for _ in range(blocks)])


def _write(path: Path, tensors: dict[str, np.ndarray], metadata: dict | None = None) -> Path:
    save_file(
        {key: value.astype(np.float32) for key, value in tensors.items()},
        str(path),
        metadata=metadata,
    )
    return path


def _pair(prefix: str, rank: int, out_features: int, in_features: int) -> dict[str, np.ndarray]:
    return {
        f"{prefix}.lora_A.weight": np.ones((rank, in_features)),
        f"{prefix}.lora_B.weight": np.ones((out_features, rank)),
    }


class ApplyLorasTest(unittest.TestCase):
    def test_fused_rows_split_gate_first(self):
        model = _Stub()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fused.safetensors"
            prefix = "diffusion_model.transformer_blocks.0.img_mlp.gate_up"
            _write(
                path,
                {
                    f"{prefix}.lora_A.weight": np.ones((2, _DIM)),
                    f"{prefix}.lora_B.weight": np.arange(2 * _MLP * 2).reshape(2 * _MLP, 2),
                },
            )
            injected = apply_loras(model, {str(path): 1.0})

        self.assertEqual(injected, 7)
        gate = model.transformer_blocks[0].img_mlp.gate_layer
        proj = model.transformer_blocks[0].img_mlp.proj
        rows = np.arange(2 * _MLP * 2).reshape(2 * _MLP, 2)
        # The stub is fp32, so no cast happens and the values stay exact.
        expected = torch.tensor(rows, dtype=torch.float32)
        self.assertTrue(torch.equal(gate.lora_B.data, expected[:_MLP]))
        self.assertTrue(torch.equal(proj.lora_B.data, expected[_MLP:]))

    def test_two_ranks_add_without_scaling_the_smaller_file(self):
        model = _Stub()
        prefix = "diffusion_model.transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as folder:
            big = _write(Path(folder) / "big.safetensors", _pair(prefix, 4, _DIM, _DIM))
            small = _write(Path(folder) / "small.safetensors", _pair(prefix, 2, _DIM, _DIM))
            apply_loras(model, {str(big): 0.5, str(small): 1.0})

        module = model.transformer_blocks[0].attn.to_q
        self.assertEqual(module.rank, 4)
        # File alphas default to their own ranks, so the factor is the strength.
        expected_a = torch.zeros(4, _DIM)
        expected_a[:4] += torch.full((4, _DIM), 0.5**0.5)
        expected_a[:2] += torch.full((2, _DIM), 1.0)
        expected_b = torch.zeros(_DIM, 4)
        expected_b[:, :4] += torch.full((_DIM, 4), 0.5**0.5)
        expected_b[:, :2] += torch.full((_DIM, 2), 1.0)
        self.assertTrue(torch.allclose(module.lora_A.data, expected_a))
        self.assertTrue(torch.allclose(module.lora_B.data, expected_b))

    def test_zero_strength_leaves_the_model_unchanged(self):
        model = _Stub()
        inputs = torch.randn(3, _DIM)
        before = model.transformer_blocks[0].attn.to_q(inputs)
        prefix = "diffusion_model.transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder) / "zero.safetensors", _pair(prefix, 2, _DIM, _DIM))
            apply_loras(model, {str(path): 0.0})
        after = model.transformer_blocks[0].attn.to_q(inputs)
        self.assertTrue(torch.equal(before, after))

    def test_rejects_a_module_the_model_does_not_have(self):
        model = _Stub(blocks=1)
        prefix = "diffusion_model.transformer_blocks.9.attn.to_q"
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder) / "deep.safetensors", _pair(prefix, 2, _DIM, _DIM))
            with self.assertRaises(RuntimeError) as caught:
                apply_loras(model, {str(path): 1.0})
        self.assertIn("transformer_blocks.9.attn.to_q", str(caught.exception))

    def test_inference_stores_the_factors_in_the_model_dtype(self):
        model = _Stub().to(torch.bfloat16)
        prefix = "diffusion_model.transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder) / "wide.safetensors", _pair(prefix, 2, _DIM, _DIM))
            apply_loras(model, {str(path): 1.0})
        module = model.transformer_blocks[0].attn.to_q
        self.assertIs(module.lora_A.dtype, torch.bfloat16)
        self.assertIs(module.lora_B.dtype, torch.bfloat16)

    def test_training_keeps_the_factors_in_float32(self):
        model = _Stub().to(torch.bfloat16)
        prefix = "diffusion_model.transformer_blocks.0.attn.to_q"
        with tempfile.TemporaryDirectory() as folder:
            path = _write(Path(folder) / "wide.safetensors", _pair(prefix, 2, _DIM, _DIM))
            apply_loras(model, {str(path): 1.0}, training=True)
        module = model.transformer_blocks[0].attn.to_q
        self.assertIs(module.lora_A.dtype, torch.float32)
        self.assertIs(module.lora_B.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
