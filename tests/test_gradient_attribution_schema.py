from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from limes_workspace_lens.schema import (
    GRADIENT_ATTRIBUTION_SCHEMA,
    load_json,
    validate_gradient_attribution,
)


ROOT = Path(__file__).resolve().parents[1]


class GradientAttributionSchemaTests(unittest.TestCase):
    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "limes_workspace_lens", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def test_valid_subset_artifact_passes_with_spec_and_cli(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        artifact = valid_gradient_artifact()
        self.assertEqual([], validate_gradient_attribution(artifact, spec))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = tmp_path / "gradient-attribution.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            self.run_cli(
                "validate-gradient-attribution",
                str(artifact_path),
                "--spec",
                "examples/workspace_audit_spec.json",
            )

    def test_rejects_unknown_prompt_duplicate_rows_and_bad_positions(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        artifact = valid_gradient_artifact()
        artifact["rows"].append(copy.deepcopy(artifact["rows"][0]))
        artifact["rows"][1]["prompt_id"] = "not-in-spec"
        artifact["rows"][1]["position"] = True
        artifact["rows"][1]["layer"] = False
        errors = validate_gradient_attribution(artifact, spec)
        self.assertTrue(any("duplicate row id" in error for error in errors))
        self.assertTrue(any("unknown prompt id" in error for error in errors))
        self.assertTrue(any(".position" in error for error in errors))
        self.assertTrue(any(".layer" in error for error in errors))

    def test_rejects_bad_attribution_numbers_and_duplicate_ranks(self) -> None:
        artifact = valid_gradient_artifact()
        row = artifact["rows"][0]
        row["attributions"].append(copy.deepcopy(row["attributions"][0]))
        row["attributions"][1]["signed_score"] = float("nan")
        row["attributions"][1]["abs_score"] = -0.1
        row["attributions"][1]["normalized_abs"] = 1.5
        errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("duplicate rank" in error for error in errors))
        self.assertTrue(any("signed_score" in error for error in errors))
        self.assertTrue(any("abs_score" in error for error in errors))
        self.assertTrue(any("normalized_abs" in error for error in errors))

    def test_rejects_score_direction_semantic_mismatches(self) -> None:
        artifact = valid_gradient_artifact()
        attribution = artifact["rows"][0]["attributions"][0]
        attribution["signed_score"] = -0.72
        attribution["abs_score"] = 0.72
        attribution["direction"] = "positive"
        errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("must be consistent" in error for error in errors))

        mixed = valid_gradient_artifact()
        attribution = mixed["rows"][0]["attributions"][0]
        attribution["signed_score"] = 0.1
        attribution["abs_score"] = 0.8
        attribution["direction"] = "mixed"
        self.assertEqual([], validate_gradient_attribution(mixed))

    def test_rejects_undeclared_feature_type_and_rows_over_top_k(self) -> None:
        artifact = valid_gradient_artifact()
        artifact["attribution_compatibility"]["feature_types"] = ["input_token"]
        errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("must be declared" in error for error in errors))

        too_many = valid_gradient_artifact()
        too_many["attribution_compatibility"]["attribution_top_k"] = 1
        extra = copy.deepcopy(too_many["rows"][0]["attributions"][0])
        extra["rank"] = 2
        too_many["rows"][0]["attributions"].append(extra)
        errors = validate_gradient_attribution(too_many)
        self.assertTrue(any("must not exceed attribution_top_k" in error for error in errors))

    def test_rejects_invalid_operator_and_integrated_gradient_metadata(self) -> None:
        artifact = valid_gradient_artifact()
        artifact["attribution_compatibility"]["operator"] = "made_up_gradient"
        errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("operator" in error for error in errors))

        integrated_gradients = valid_gradient_artifact()
        integrated_gradients["attribution_compatibility"]["operator"] = "integrated_gradients"
        integrated_gradients["attribution_compatibility"]["baseline_policy"] = "none"
        errors = validate_gradient_attribution(integrated_gradients)
        self.assertTrue(any("baseline_policy" in error for error in errors))
        self.assertTrue(any("requires a positive integer" in error for error in errors))

    def test_rejects_quality_and_control_condition_inconsistencies(self) -> None:
        artifact = valid_gradient_artifact()
        artifact["rows"][0]["quality"]["autograd_enabled"] = False
        artifact["rows"][0]["quality"]["nonzero_total_attribution"] = False
        artifact["rows"][0]["condition"] = {"kind": "prompt_variant"}
        errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("autograd_enabled" in error for error in errors))
        self.assertTrue(any("nonzero_total_attribution" in error for error in errors))
        self.assertTrue(any("control_id" in error for error in errors))

    def test_allows_zero_gradient_negative_evidence_when_declared(self) -> None:
        artifact = valid_gradient_artifact()
        row = artifact["rows"][0]
        row["attributions"][0]["signed_score"] = 0
        row["attributions"][0]["abs_score"] = 0
        row["attributions"][0]["normalized_abs"] = 0
        row["attributions"][0]["direction"] = "zero"
        row["quality"]["nonzero_total_attribution"] = False
        self.assertEqual([], validate_gradient_attribution(artifact))

    def test_feature_labels_do_not_trigger_secret_false_positives(self) -> None:
        artifact = valid_gradient_artifact()
        attribution = artifact["rows"][0]["attributions"][0]
        attribution["feature_id"] = "feature mentions Bearer abcdefgh"
        attribution["token"] = "secret"
        attribution["feature_text_sha256"] = "a" * 64
        self.assertEqual([], validate_gradient_attribution(artifact))

    def test_public_generation_and_input_metadata_still_reject_leaks(self) -> None:
        artifact = valid_gradient_artifact()
        artifact["generation"]["command"] = "python run.py --token sk-abcdefghijklmnop"
        command_errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("generation.command" in error for error in command_errors))

        path_leak = valid_gradient_artifact()
        path_leak["input_artifacts"][0]["path"] = "/tmp/private/readouts.json"
        path_errors = validate_gradient_attribution(path_leak)
        self.assertTrue(any("input_artifacts[0].path" in error for error in path_errors))
        self.assertTrue(any("absolute local path" in error for error in path_errors))

        source_leak = valid_gradient_artifact()
        source_leak["source"] = "/Users/private/run sk-abcdefghijklmnop"
        source_errors = validate_gradient_attribution(source_leak)
        self.assertTrue(any("gradient_attribution.source" in error for error in source_errors))

    def test_model_checkpoint_must_match_compatibility_checkpoint(self) -> None:
        artifact = valid_gradient_artifact()
        artifact["model"]["checkpoint"] = "different-checkpoint"
        errors = validate_gradient_attribution(artifact)
        self.assertTrue(any("must match" in error for error in errors))


def valid_gradient_artifact() -> dict[str, Any]:
    return {
        "schema_version": GRADIENT_ATTRIBUTION_SCHEMA,
        "generated_utc": "2026-07-07T00:00:00Z",
        "source": "gradient-attribution-fixture",
        "synthetic": False,
        "model": {
            "id": "fixture-model",
            "checkpoint": "fixture-checkpoint",
        },
        "compatibility": compatibility(),
        "attribution_compatibility": {
            "operator": "gradient_x_activation",
            "target_policy": "selected_readout_token",
            "feature_types": ["residual_stream", "input_token"],
            "attribution_top_k": 8,
            "rank_by": "abs_score",
            "normalization": "l1_abs",
            "baseline_policy": "not_applicable",
            "hook_policy": "residual_stream_pre_layer_norm",
            "autograd_backend": "torch.func.grad",
            "dtype": "float32",
        },
        "generation": {
            "mode": "offline_autograd",
            "command": "python -m workspace_lens_attribution run --config runs/config.json",
            "dependency_profile": "torch-autograd",
            "seed": 7,
            "config": {
                "batch_size": 1,
                "gradient_checkpointing": False,
            },
        },
        "input_artifacts": [
            {
                "kind": "readouts",
                "path": "readouts.json",
                "sha256": "0" * 64,
            }
        ],
        "rows": [
            {
                "row_id": "math-copy:token-49:layer-32",
                "prompt_id": "math-copy",
                "position": -1,
                "layer": 32,
                "target": {
                    "kind": "readout_token",
                    "token": "49",
                    "rank": 1,
                    "score": 4.2,
                    "description": "Top readout token attribution target.",
                    "artifact_ref": "readouts",
                },
                "condition": {
                    "kind": "observed",
                    "control_id": None,
                    "alignment_policy": "same_prompt_position",
                },
                "attributions": [
                    {
                        "rank": 1,
                        "feature_type": "residual_stream",
                        "feature_id": "layer32/residual:174",
                        "feature_position": -1,
                        "feature_token_id": 905,
                        "feature_text_sha256": "1" * 64,
                        "signed_score": 0.72,
                        "abs_score": 0.72,
                        "normalized_abs": 1.0,
                        "direction": "positive",
                        "layer": 32,
                        "token": "49",
                    }
                ],
                "quality": {
                    "finite_values": True,
                    "target_found_in_readouts": True,
                    "autograd_enabled": True,
                    "nonzero_total_attribution": True,
                    "completeness_delta": 0.01,
                },
            }
        ],
    }


def compatibility() -> dict[str, Any]:
    return {
        "model_checkpoint": "fixture-checkpoint",
        "tokenizer_revision": "fixture-tokenizer",
        "lens_source": "anthropics/jacobian-lens",
        "lens_revision": "fixture-lens-revision",
        "prompt_suite_hash": "fixture-prompt-suite-sha256",
        "top_k": 8,
        "layer_policy": "workspace_layer_range=24-40",
        "position_policy": "positions=-1",
        "fit_procedure": "fixture-fit-procedure",
    }


if __name__ == "__main__":
    unittest.main()
