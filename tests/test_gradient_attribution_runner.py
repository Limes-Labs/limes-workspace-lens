from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from limes_workspace_lens.gradient_attribution_runner import (
    build_attribution_row,
    build_gradient_attribution_artifact,
    compute_gradient_x_activation,
    parse_prompt_ids,
    prompts_by_id,
    resolve_target_token_id,
    select_readout_targets,
    validate_built_artifact,
)
from limes_workspace_lens.jlens_adapter import AdapterError
from limes_workspace_lens.schema import load_json, validate_readouts


ROOT = Path(__file__).resolve().parents[1]


class GradientAttributionRunnerTests(unittest.TestCase):
    def test_select_readout_targets_filters_rank_prompt_and_max_rows(self) -> None:
        readouts = readout_fixture()
        targets = select_readout_targets(
            readouts,
            readout_rank=2,
            prompt_ids={"math-copy"},
            max_rows=1,
        )
        self.assertEqual(1, len(targets))
        self.assertEqual("math-copy", targets[0].prompt_id)
        self.assertEqual("42", targets[0].token)
        self.assertEqual(42, targets[0].token_id)
        self.assertEqual(2, targets[0].rank)

    def test_select_readout_targets_rejects_empty_selection(self) -> None:
        with self.assertRaisesRegex(AdapterError, "no readout rows matched"):
            select_readout_targets(readout_fixture(), readout_rank=3, prompt_ids={"math-copy"})

    def test_prompt_id_parser_rejects_empty_values(self) -> None:
        self.assertEqual({"a", "b"}, parse_prompt_ids("a,b"))
        with self.assertRaisesRegex(AdapterError, "prompt id"):
            parse_prompt_ids(" , ")

    def test_resolve_target_token_id_prefers_artifact_id_and_fails_closed(self) -> None:
        target = select_readout_targets(readout_fixture(), readout_rank=1)[0]
        tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [999])
        self.assertEqual(49, resolve_target_token_id(tokenizer, target, allow_token_reencode=False))

        legacy = copy.copy(target)
        object.__setattr__(legacy, "token_id", None)
        with self.assertRaisesRegex(AdapterError, "missing token_id"):
            resolve_target_token_id(tokenizer, legacy, allow_token_reencode=False)
        self.assertEqual(999, resolve_target_token_id(tokenizer, legacy, allow_token_reencode=True))

        bad_tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2])
        with self.assertRaisesRegex(AdapterError, "exactly one token id"):
            resolve_target_token_id(bad_tokenizer, legacy, allow_token_reencode=True)

    def test_build_artifact_from_rows_validates_against_schema(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        target = select_readout_targets(readout_fixture(), readout_rank=1)[0]
        row = build_attribution_row(
            target=target,
            target_token_id=49,
            readout_artifact_id="readouts",
            attributions=[
                {
                    "rank": 1,
                    "feature_type": "input_token",
                    "feature_id": "input_token:0:101",
                    "feature_position": 0,
                    "feature_token_id": 101,
                    "feature_text_sha256": "1" * 64,
                    "signed_score": -0.5,
                    "abs_score": 0.5,
                    "normalized_abs": 1.0,
                    "direction": "negative",
                    "token": "calc",
                }
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            readouts_path = Path(tmp) / "readouts.json"
            readouts_path.write_text(json.dumps(readout_fixture()), encoding="utf-8")
            artifact = build_gradient_attribution_artifact(
                spec=spec,
                readouts_path=readouts_path,
                readouts_artifact_path="readouts.json",
                readout_artifact_id="readouts",
                rows=[row],
                model="Qwen/Qwen3-0.6B",
                model_checkpoint="model-rev",
                tokenizer_revision="tokenizer-rev",
                lens_revision="lens-rev",
                fit_procedure="fixture fit",
                position_policy="positions=-1",
                layer_policy=None,
                prompt_suite_hash=None,
                attribution_top_k=4,
                seed=7,
                device="cpu",
                torch_dtype="float32",
                model_revision="model-rev",
                local_files_only=True,
                trust_remote_code=False,
                allow_token_reencode=False,
                readout_rank=1,
            )
            validate_built_artifact(artifact, spec)
            self.assertEqual("model-rev", artifact["model"]["checkpoint"])
            self.assertEqual("readouts.json", artifact["input_artifacts"][0]["path"])
            self.assertEqual(
                "selected_readout_token_rank_1",
                artifact["attribution_compatibility"]["target_policy"],
            )

    def test_attribution_row_records_omitted_mass_from_full_l1_normalization(self) -> None:
        target = select_readout_targets(readout_fixture(), readout_rank=1)[0]
        row = build_attribution_row(
            target=target,
            target_token_id=49,
            readout_artifact_id="readouts",
            attributions=[
                {
                    "rank": 1,
                    "feature_type": "input_token",
                    "feature_id": "input_token:0:101",
                    "signed_score": 0.5,
                    "abs_score": 0.5,
                    "normalized_abs": 0.25,
                    "direction": "positive",
                },
                {
                    "rank": 2,
                    "feature_type": "input_token",
                    "feature_id": "input_token:1:102",
                    "signed_score": 0.5,
                    "abs_score": 0.5,
                    "normalized_abs": 0.25,
                    "direction": "positive",
                },
            ],
        )
        self.assertEqual(0.5, row["quality"]["completeness_delta"])

    def test_readout_schema_accepts_non_negative_token_id_and_rejects_bad_ids(self) -> None:
        self.assertEqual([], validate_readouts(readout_fixture()))
        bad = readout_fixture()
        bad["readouts"][0]["top_tokens"][0]["token_id"] = -1
        self.assertTrue(any("token_id" in error for error in validate_readouts(bad)))
        bad_bool = readout_fixture()
        bad_bool["readouts"][0]["top_tokens"][0]["token_id"] = True
        self.assertTrue(any("token_id" in error for error in validate_readouts(bad_bool)))

    def test_prompt_mapping_uses_spec_text(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        prompts = prompts_by_id(spec)
        self.assertIn("math-copy", prompts)
        self.assertIn("old painting", prompts["math-copy"])

    def test_compute_gradient_x_activation_with_tiny_torch_model_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class TinyTokenizer:
            def __call__(self, text: str, return_tensors: str) -> dict[str, object]:
                self.seen_text = text
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }

            def decode(self, ids: list[int]) -> str:
                return {1: "calc", 2: "+", 3: "answer"}.get(ids[0], "?")

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 3)
                self.proj = torch.nn.Linear(3, 5, bias=False)
                with torch.no_grad():
                    self.embedding.weight.copy_(
                        torch.tensor(
                            [
                                [0.0, 0.0, 0.0],
                                [1.0, 0.5, 0.0],
                                [0.0, 1.0, 0.5],
                                [0.5, 0.0, 1.0],
                                [1.0, 1.0, 1.0],
                            ]
                        )
                    )
                    self.proj.weight.copy_(
                        torch.tensor(
                            [
                                [0.1, 0.0, 0.0],
                                [0.0, 0.1, 0.0],
                                [0.0, 0.0, 0.1],
                                [0.4, 0.2, 0.8],
                                [0.0, 0.0, 0.0],
                            ]
                        )
                    )

            def get_input_embeddings(self) -> torch.nn.Embedding:
                return self.embedding

            def forward(self, *, inputs_embeds: object, attention_mask: object | None = None) -> object:
                return SimpleNamespace(logits=self.proj(inputs_embeds))

        attributions = compute_gradient_x_activation(
            torch_module=torch,
            model=TinyModel(),
            tokenizer=TinyTokenizer(),
            prompt_text="calc",
            target_token_id=3,
            target_position=-1,
            attribution_top_k=2,
            device="cpu",
        )
        self.assertEqual(2, len(attributions))
        self.assertEqual([1, 2], [item["rank"] for item in attributions])
        self.assertEqual("input_token", attributions[0]["feature_type"])
        self.assertGreaterEqual(attributions[0]["abs_score"], attributions[1]["abs_score"])
        normalized_total = sum(item["normalized_abs"] for item in attributions)
        self.assertLess(normalized_total, 1.0)
        self.assertAlmostEqual(1.6 / 2.1, normalized_total)

    def test_cli_reports_missing_model_deps_without_requiring_jlens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            readouts_path = Path(tmp) / "readouts.json"
            readouts_path.write_text(
                json.dumps(readout_fixture(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "scripts/run_gradient_attribution.py",
                    "--model",
                    "fixture-model",
                    "--spec",
                    "examples/workspace_audit_spec.json",
                    "--readouts",
                    str(readouts_path),
                    "--out",
                    str(Path(tmp) / "gradient.json"),
                    "--lens-revision",
                    "fixture-lens",
                    "--fit-procedure",
                    "fixture fit",
                    "--position-policy",
                    "positions=-1",
                    "--max-rows",
                    "1",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("missing optional real-model dependencies", result.stderr)
        self.assertNotIn("jlens", result.stderr.lower())


def readout_fixture() -> dict[str, object]:
    return {
        "schema_version": "limes-workspace-lens/readouts.v0.1",
        "source": "fixture",
        "synthetic": False,
        "readouts": [
            {
                "prompt_id": "math-copy",
                "position": -1,
                "layer": 12,
                "top_tokens": [
                    {"token": "49", "token_id": 49, "rank": 1, "score": 4.2},
                    {"token": "42", "token_id": 42, "rank": 2, "score": 2.1},
                ],
            },
            {
                "prompt_id": "prompt-injection-check",
                "position": -1,
                "layer": 12,
                "top_tokens": [
                    {"token": "fake", "token_id": 200, "rank": 1, "score": 3.0}
                ],
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
