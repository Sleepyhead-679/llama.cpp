import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


VERIFY_MODEL_FILE = Path(__file__).resolve().parents[1] / "verify_model.py"


class VerifyModelTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def load_target(self):
        self.assertTrue(VERIFY_MODEL_FILE.is_file(), "verify_model.py has not been created")
        spec = importlib.util.spec_from_file_location("course_verify_model", VERIFY_MODEL_FILE)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def manifest_for(data, filename):
        return {
            "repo": "Qwen/Qwen3-0.6B-GGUF",
            "revision": "1eaf4d9657fe65ad10a51eab76a8db5b363bddaa",
            "filename": filename,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "license": "apache-2.0",
            "source_url": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
        }

    def test_load_manifest_reads_json(self):
        module = self.load_target()
        expected = self.manifest_for(b"test", "test.gguf")
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(expected), encoding="utf-8")

        self.assertEqual(module.load_manifest(manifest_path), expected)

    def test_missing_model_raises_value_error(self):
        module = self.load_target()
        manifest = self.manifest_for(b"test", "missing.gguf")

        with self.assertRaises(ValueError):
            module.verify_model(self.root / "missing.gguf", manifest)

    def test_wrong_file_size_raises_value_error(self):
        module = self.load_target()
        model_path = self.root / "model.gguf"
        model_path.write_bytes(b"test")
        manifest = self.manifest_for(b"test", model_path.name)
        manifest["size_bytes"] += 1

        with self.assertRaises(ValueError):
            module.verify_model(model_path, manifest)

    def test_wrong_sha256_with_same_size_raises_value_error(self):
        module = self.load_target()
        model_path = self.root / "model.gguf"
        model_path.write_bytes(b"test")
        manifest = self.manifest_for(b"fail", model_path.name)

        with self.assertRaises(ValueError):
            module.verify_model(model_path, manifest)

    def test_valid_model_in_unicode_and_space_path_passes(self):
        module = self.load_target()
        model_directory = self.root / "含 空格的目录"
        model_directory.mkdir()
        model_path = model_directory / "模型 文件.gguf"
        data = b"small test model"
        model_path.write_bytes(data)
        manifest = self.manifest_for(data, model_path.name)

        self.assertIsNone(module.verify_model(model_path, manifest))

    def test_cli_accepts_valid_model(self):
        model_path = self.root / "model.gguf"
        data = b"small test model"
        model_path.write_bytes(data)
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest_for(data, model_path.name)), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(VERIFY_MODEL_FILE), "--manifest", str(manifest_path), "--model", str(model_path)],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("模型校验通过", result.stdout)

    def test_cli_rejects_wrong_sha256(self):
        model_path = self.root / "model.gguf"
        model_path.write_bytes(b"test")
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest_for(b"fail", model_path.name)), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(VERIFY_MODEL_FILE), "--manifest", str(manifest_path), "--model", str(model_path)],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("模型校验失败", result.stderr)
        self.assertIn("SHA-256 不匹配", result.stderr)

    def test_cli_help_is_in_chinese(self):
        result = subprocess.run(
            [sys.executable, str(VERIFY_MODEL_FILE), "--help"],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("用法", result.stdout)
        self.assertIn("模型清单", result.stdout)
        self.assertIn("模型文件", result.stdout)
        self.assertNotIn("usage:", result.stdout)
        self.assertNotIn("show this help message", result.stdout)

    def test_cli_missing_argument_reports_chinese_error(self):
        result = subprocess.run(
            [sys.executable, str(VERIFY_MODEL_FILE), "--model", "model.gguf"],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("参数错误", result.stderr)
        self.assertNotIn("required", result.stderr)


if __name__ == "__main__":
    unittest.main()
