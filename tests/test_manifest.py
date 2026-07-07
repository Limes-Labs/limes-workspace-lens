from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from limes_workspace_lens.manifest import build_manifest, parse_metadata, validate_manifest


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_manifest_validates_and_catches_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            manifest = build_manifest([artifact], root=root, commands=["generate"], metadata={"status": "diagnostic"})
            self.assertEqual(".", manifest["root"])
            self.assertEqual([], validate_manifest(manifest, root=root))

            artifact.write_text('{"ok": false}\n', encoding="utf-8")
            errors = validate_manifest(manifest, root=root)
            self.assertTrue(any("sha256 mismatch" in error for error in errors))

    def test_manifest_catches_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            manifest = build_manifest([artifact], root=root)
            artifact.unlink()
            errors = validate_manifest(manifest, root=root)
            self.assertTrue(any("missing" in error for error in errors))

    def test_manifest_rejects_duplicate_and_escaping_paths(self) -> None:
        manifest = {
            "schema_version": "limes-workspace-lens/artifact-manifest.v0.1",
            "files": [
                {"path": "a.txt", "size_bytes": 1, "sha256": "0" * 64},
                {"path": "a.txt", "size_bytes": 1, "sha256": "0" * 64},
                {"path": "../escape.txt", "size_bytes": 1, "sha256": "0" * 64},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("x", encoding="utf-8")
            errors = validate_manifest(manifest, root=root)
            self.assertTrue(any("duplicates" in error for error in errors))
            self.assertTrue(any("safe relative path" in error for error in errors))

    def test_validate_manifest_rejects_absolute_file_path_even_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            manifest = build_manifest([artifact], root=root)
            manifest["files"][0]["path"] = str(artifact)
            errors = validate_manifest(manifest, root=root)
            self.assertTrue(any("safe relative path" in error for error in errors))

    def test_build_manifest_rejects_inputs_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as other_tmp:
            outside = Path(other_tmp) / "artifact.json"
            outside.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_manifest([outside], root=root_tmp)

    def test_manifest_rejects_public_command_and_metadata_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.json"
            artifact.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unredacted secret-like value"):
                build_manifest(
                    [artifact],
                    root=root,
                    commands=["python run.py --token hf_abcdefghijklmnop"],
                )

            manifest = build_manifest([artifact], root=root, commands=["python run.py"])
            manifest["commands"] = ["python /tmp/private/run.py"]
            manifest["metadata"] = {"HF_TOKEN": "raw-token-value"}
            errors = validate_manifest(manifest, root=root)
            self.assertTrue(any("manifest.commands[0]" in error for error in errors))
            self.assertTrue(any("absolute local path" in error for error in errors))
            self.assertTrue(any("manifest.metadata.HF_TOKEN" in error for error in errors))
            self.assertTrue(any("secret-like field" in error for error in errors))

    def test_parse_metadata_requires_key_value_pairs(self) -> None:
        self.assertEqual({"a": "b"}, parse_metadata(["a=b"]))
        with self.assertRaises(ValueError):
            parse_metadata(["broken"])


class ManifestCliTests(unittest.TestCase):
    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "limes_workspace_lens", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def test_cli_build_and_validate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "report.json"
            manifest = root / "manifest.json"
            artifact.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
            self.run_cli(
                "build-manifest",
                str(artifact),
                "--root",
                str(root),
                "--out",
                str(manifest),
                "--command",
                "unit-test",
                "--metadata",
                "status=diagnostic",
            )
            self.run_cli("validate-manifest", str(manifest), "--root", str(root))
            artifact.write_text(json.dumps({"ok": False}) + "\n", encoding="utf-8")
            failed = self.run_cli("validate-manifest", str(manifest), "--root", str(root), check=False)
            self.assertNotEqual(0, failed.returncode)
            self.assertIn("sha256 mismatch", failed.stderr)

    def test_cli_build_manifest_rejects_unredacted_command_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "report.json"
            manifest = root / "manifest.json"
            artifact.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
            failed = self.run_cli(
                "build-manifest",
                str(artifact),
                "--root",
                str(root),
                "--out",
                str(manifest),
                "--command",
                "python run.py --token hf_abcdefghijklmnop",
                check=False,
            )
            self.assertNotEqual(0, failed.returncode)
            self.assertIn("unredacted secret-like value", failed.stderr)
            self.assertFalse(manifest.exists())

    def test_cli_manifest_validates_after_bundle_move(self) -> None:
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as target_tmp:
            source = Path(source_tmp)
            artifact = source / "artifact.json"
            manifest = source / "manifest.json"
            artifact.write_text('{"portable": true}\n', encoding="utf-8")
            self.run_cli(
                "build-manifest",
                str(artifact),
                "--root",
                str(source),
                "--out",
                str(manifest),
            )
            moved = Path(target_tmp) / "bundle"
            shutil.copytree(source, moved)
            self.run_cli("validate-manifest", str(moved / "manifest.json"))


if __name__ == "__main__":
    unittest.main()
