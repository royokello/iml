import unittest

import torch

from qwen_2.config import TURBO_SIGMAS
from qwen_2.inputs import (
    build_img_mask,
    build_img_shapes,
    build_sigmas,
    pack_latents,
    unpack_latents,
)


class InputTest(unittest.TestCase):
    def test_img_shapes(self):
        self.assertEqual(build_img_shapes(1024, 1024), [[(1, 64, 64)]])

    def test_turbo_sigmas(self):
        self.assertEqual(len(TURBO_SIGMAS), 8)
        self.assertEqual(len(build_sigmas(8, "turbo")), 8)

    def test_base_sigmas(self):
        sigmas = build_sigmas(40, "base")
        self.assertEqual(sigmas.shape, (40,))
        self.assertAlmostEqual(float(sigmas[0]), 1.0, places=6)
        self.assertAlmostEqual(float(sigmas[-1]), 1.0 / 40, places=6)

    def test_img_mask_marks_only_the_target(self):
        mask = build_img_mask(7, 4096)
        self.assertEqual(mask.shape, (1, 7 + 1024))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(bool(mask[0, :7].any()))
        self.assertTrue(bool(mask[0, 7:].all()))

    def test_img_mask_rejects_partial_slots(self):
        with self.assertRaises(ValueError):
            build_img_mask(7, 4095)

    def test_img_shapes_lists_references_before_the_target(self):
        shapes = build_img_shapes(1024, 1024, [(512, 512), (736, 384)])
        self.assertEqual(
            shapes, [[(1, 32, 32), (1, 24, 46), (1, 64, 64)]]
        )

    def test_img_mask_keeps_the_reference_slots(self):
        pad_mask = torch.zeros(1, 7, dtype=torch.bool)
        pad_mask[0, 2:4] = True
        mask = build_img_mask(7, 4096, pad_mask)
        self.assertEqual(mask.shape, (1, 7 + 1024))
        self.assertTrue(bool(mask[0, 2:4].all()))
        self.assertFalse(bool(mask[0, 4:7].any()))
        self.assertTrue(bool(mask[0, 7:].all()))

    def test_img_mask_rejects_a_pad_mask_of_the_wrong_length(self):
        with self.assertRaises(ValueError):
            build_img_mask(7, 4096, torch.zeros(1, 6, dtype=torch.bool))

    def test_latent_roundtrip(self):
        latents = torch.randn(1, 64, 1, 64, 64)
        packed = pack_latents(latents)
        self.assertEqual(packed.shape, (1, 4096, 64))
        restored = unpack_latents(packed, 1024, 1024)
        self.assertEqual(restored.shape, (1, 64, 1, 64, 64))
        self.assertTrue(torch.equal(restored, latents))
