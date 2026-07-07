from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from limes_workspace_lens.jlens_adapter import (
    AdapterError,
    OptionalDeps,
    build_provenance,
    choose_device,
    load_optional_deps,
    load_prompt_texts,
    parse_positions,
    parse_torch_dtype,
    pretrained_kwargs,
    public_identifier,
    safe_lens_file,
    set_seed,
    sha256_file,
)


class JLensAdapterTests(unittest.TestCase):
    def test_parse_positions_accepts_signed_comma_list(self) -> None:
        self.assertEqual([-1, 0, 4], parse_positions("-1, 0,4"))

    def test_parse_positions_rejects_empty_or_non_integer_values(self) -> None:
        with self.assertRaisesRegex(AdapterError, "empty position"):
            parse_positions("-1,,2")
        with self.assertRaisesRegex(AdapterError, "not an integer"):
            parse_positions("a")

    def test_prompt_loader_rejects_empty_text_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text(
                '{"id": "a", "text": "hello"}\n{"id": "a", "text": "again"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AdapterError, "duplicate prompt id"):
                load_prompt_texts(path)
            path.write_text('{"id": "a", "text": ""}\n', encoding="utf-8")
            with self.assertRaisesRegex(AdapterError, "prompt text"):
                load_prompt_texts(path)

    def test_prompt_loader_respects_positive_max_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            path.write_text(
                '{"id": "a", "text": "one"}\n{"id": "b", "text": "two"}\n',
                encoding="utf-8",
            )
            self.assertEqual(["one"], load_prompt_texts(path, max_prompts=1))
            with self.assertRaisesRegex(AdapterError, "--max-prompts"):
                load_prompt_texts(path, max_prompts=0)

    def test_choose_device_preflights_unavailable_accelerators(self) -> None:
        torch = fake_torch(cuda=False, mps=False)
        self.assertEqual("cpu", choose_device("auto", torch))
        with self.assertRaisesRegex(AdapterError, "CUDA is not available"):
            choose_device("cuda", torch)
        with self.assertRaisesRegex(AdapterError, "MPS is not available"):
            choose_device("mps", torch)

    def test_choose_device_prefers_cuda_then_mps_for_auto(self) -> None:
        self.assertEqual("cuda", choose_device("auto", fake_torch(cuda=True, mps=True)))
        self.assertEqual("mps", choose_device("auto", fake_torch(cuda=False, mps=True)))

    def test_parse_torch_dtype_and_pretrained_kwargs(self) -> None:
        torch = fake_torch(cuda=False, mps=False)
        self.assertEqual("float16-dtype", parse_torch_dtype("float16", torch))
        self.assertEqual("auto", parse_torch_dtype("auto", torch))
        with self.assertRaisesRegex(AdapterError, "--torch-dtype"):
            parse_torch_dtype("int8", torch)
        kwargs = pretrained_kwargs(
            revision="abc123",
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype="float16-dtype",
        )
        self.assertEqual(
            {
                "revision": "abc123",
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": "float16-dtype",
            },
            kwargs,
        )

    def test_missing_optional_dependencies_are_reported_together(self) -> None:
        def fake_import(name: str):
            if name in {"torch", "jlens"}:
                raise ImportError(name)
            return SimpleNamespace(__version__="ok")

        with mock.patch("importlib.import_module", side_effect=fake_import):
            with self.assertRaisesRegex(AdapterError, "torch, jlens"):
                load_optional_deps()

    def test_set_seed_accepts_zero_and_rejects_negative(self) -> None:
        seen: list[int] = []
        torch = SimpleNamespace(
            manual_seed=seen.append,
            cuda=SimpleNamespace(manual_seed_all=seen.append),
        )
        set_seed(torch, 0)
        self.assertEqual([0, 0], seen)
        with self.assertRaisesRegex(AdapterError, "non-negative"):
            set_seed(torch, -1)

    def test_provenance_redacts_absolute_paths_and_hashes_lens_and_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "spec.json"
            spec.write_text("{}", encoding="utf-8")
            lens_dir = root / "lens-dir"
            lens_dir.mkdir()
            lens_file = lens_dir / "lens.pt"
            lens_file.write_bytes(b"lens")
            expected_lens_sha = sha256_file(lens_file)
            expected_spec_sha = sha256_file(spec)
            deps = OptionalDeps(
                torch=SimpleNamespace(__version__="2.x"),
                transformers=SimpleNamespace(__version__="4.x"),
                jlens=SimpleNamespace(__version__="0.x"),
            )
            provenance = build_provenance(
                model=str(root / "model"),
                model_revision="model-rev",
                tokenizer_revision=None,
                lens_repo=str(lens_dir),
                lens_file="lens.pt",
                lens_revision="lens-rev",
                spec_path=spec,
                prompt_count=3,
                positions=[-1],
                top_k=10,
                device="cpu",
                torch_dtype="float16",
                local_files_only=True,
                trust_remote_code=False,
                seed=7,
                deps=deps,
            )
            self.assertEqual("<local:model>", provenance["model"]["requested"])
            self.assertEqual("<local:lens-dir>", provenance["lens"]["repo"])
            self.assertEqual(expected_lens_sha, provenance["lens"]["sha256"])
            self.assertEqual(expected_spec_sha, provenance["spec_sha256"])
            self.assertEqual("4.x", provenance["versions"]["transformers"])

    def test_public_identifier_leaves_remote_ids_intact(self) -> None:
        self.assertEqual("Qwen/Qwen3-0.6B", public_identifier("Qwen/Qwen3-0.6B"))

    def test_lens_file_must_be_safe_relative_path(self) -> None:
        self.assertEqual("nested/lens.pt", safe_lens_file("nested/lens.pt"))
        for value in [
            "",
            "/tmp/lens.pt",
            "../lens.pt",
            "nested\\lens.pt",
            "C:/Users/me/lens.pt",
            "file:///Users/me/lens.pt",
            "https://example.com/lens.pt",
            "~/lens.pt",
        ]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(AdapterError, "--lens-file"):
                    safe_lens_file(value)


def fake_torch(*, cuda: bool, mps: bool) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="fake",
        float16="float16-dtype",
        bfloat16="bfloat16-dtype",
        float32="float32-dtype",
        float64="float64-dtype",
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


if __name__ == "__main__":
    unittest.main()
