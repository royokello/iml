import tempfile
import unittest
from pathlib import Path

from qwen_2.files import copy_files


class CopyFilesTest(unittest.TestCase):
    def test_copy_files_copies_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            source.mkdir()
            (source / "config.json").write_bytes(b"x")
            result = copy_files(source, target, ("config.json",))
            self.assertEqual(result, [target / "config.json"])
            self.assertTrue((target / "config.json").is_file())
            self.assertEqual((target / "config.json").read_bytes(), b"x")

    def test_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            source.mkdir()
            with self.assertRaises(FileNotFoundError):
                copy_files(source, target, ("missing.json",))
            self.assertFalse((target / "missing.json").exists())


class CopyModuleTest(unittest.TestCase):
    def test_text_encoder_files(self):
        from qwen_2.text_encoder.copy_files import copy_text_encoder_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = root / "encoder"
            processor = root / "processor"
            encoder.mkdir()
            processor.mkdir()
            for name in ("config.json", "generation_config.json"):
                (encoder / name).write_bytes(b"x")
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "processor_config.json",
            ):
                (processor / name).write_bytes(b"x")

            result = copy_text_encoder_files(root, encoder, processor)

            target = root / "qwen_2" / "model" / "text_encoder"
            self.assertEqual(
                sorted(path.name for path in result),
                sorted(
                    (
                        "config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                        "processor_config.json",
                    )
                ),
            )
            self.assertEqual({path.parent for path in result}, {target})
            for name in ("config.json", "tokenizer.json", "processor_config.json"):
                self.assertTrue((target / name).is_file())

    def test_denoiser_files(self):
        from qwen_2.denoiser.copy_files import copy_denoiser_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "transformer"
            source.mkdir()
            for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
                (source / name).write_bytes(b"x")

            result = copy_denoiser_files(root, source)

            target = root / "qwen_2" / "model" / "denoiser"
            self.assertEqual(
                sorted(path.name for path in result),
                ["config.json", "diffusion_pytorch_model.safetensors.index.json"],
            )
            self.assertTrue((target / "config.json").is_file())

    def test_vae_files(self):
        from qwen_2.vae.copy_files import copy_vae_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "vae_source"
            source.mkdir()
            for name in ("config.json", "diffusion_pytorch_model.safetensors"):
                (source / name).write_bytes(b"x")

            result = copy_vae_files(root, source)

            target = root / "qwen_2" / "model" / "vae"
            self.assertEqual(
                sorted(path.name for path in result),
                ["config.json", "diffusion_pytorch_model.safetensors"],
            )
            self.assertTrue((target / "diffusion_pytorch_model.safetensors").is_file())
