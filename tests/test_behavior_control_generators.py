from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from limes_workspace_lens.analysis import score_readouts
from limes_workspace_lens.evidence import EVIDENCE_BUNDLE_SCHEMA, validate_evidence_bundle
from limes_workspace_lens.schema import (
    AUDIT_SPEC_SCHEMA,
    BEHAVIOR_EVAL_SCHEMA,
    CONTROL_EVAL_SCHEMA,
    READOUT_SCHEMA,
    REPORT_SCHEMA,
    load_json,
    validate_behavior_eval,
    validate_control_eval,
)


ROOT = Path(__file__).resolve().parents[1]


class BehaviorControlGeneratorTests(unittest.TestCase):
    def run_cli(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "limes_workspace_lens", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=check,
        )

    def test_cli_generates_valid_behavior_and_control_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            behavior_rows = tmp_path / "behavior-responses.jsonl"
            control_rows = tmp_path / "control-responses.jsonl"
            behavior_out = tmp_path / "behavior.json"
            control_out = tmp_path / "control.json"
            write_jsonl(behavior_rows, example_behavior_rows())
            write_jsonl(control_rows, example_control_rows())

            common_args = [
                "examples/workspace_audit_spec.json",
                "--tokenizer-revision",
                "fixture-tokenizer-revision",
                "--lens-revision",
                "fixture-lens-revision",
                "--fit-procedure",
                "fixture-fit-procedure",
                "--position-policy",
                "positions=-1",
                "--seed",
                "7",
                "--generation-config",
                '{"temperature": 0, "max_new_tokens": 32}',
            ]
            self.run_cli(
                "run-behavior-eval",
                *common_args,
                "--responses",
                str(behavior_rows),
                "--out",
                str(behavior_out),
            )
            self.run_cli(
                "run-control-eval",
                *common_args,
                "--responses",
                str(control_rows),
                "--control-kind",
                "prompt_variant",
                "--out",
                str(control_out),
            )
            self.run_cli(
                "validate-behavior-eval",
                str(behavior_out),
                "--spec",
                "examples/workspace_audit_spec.json",
            )
            self.run_cli(
                "validate-control-eval",
                str(control_out),
                "--spec",
                "examples/workspace_audit_spec.json",
            )

            behavior = load_json(behavior_out)
            control = load_json(control_out)

        self.assertEqual(BEHAVIOR_EVAL_SCHEMA, behavior["schema_version"])
        self.assertEqual(CONTROL_EVAL_SCHEMA, control["schema_version"])
        self.assertEqual(3, len(behavior["rows"]))
        self.assertEqual(3, len(control["rows"]))
        self.assertNotIn("output_text", behavior["rows"][0])
        self.assertEqual("stdlib-only-no-model-execution", behavior["generation"]["dependency_profile"])
        self.assertEqual({"temperature": 0, "max_new_tokens": 32}, behavior["generation"]["config"])
        math_row = next(row for row in behavior["rows"] if row["prompt_id"] == "math-copy")
        self.assertTrue(math_row["metrics"]["forbidden_surface_terms_absent"]["passed"])

    def test_behavior_generator_fails_when_prompt_row_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            behavior_rows = tmp_path / "behavior-responses.jsonl"
            write_jsonl(behavior_rows, example_behavior_rows()[:-1])
            result = self.run_cli(
                "run-behavior-eval",
                "examples/workspace_audit_spec.json",
                "--responses",
                str(behavior_rows),
                "--out",
                str(tmp_path / "behavior.json"),
                "--tokenizer-revision",
                "fixture-tokenizer-revision",
                "--lens-revision",
                "fixture-lens-revision",
                "--fit-procedure",
                "fixture-fit-procedure",
                "--position-policy",
                "positions=-1",
                check=False,
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("missing rows for prompt ids", result.stderr)

    def test_validators_reject_undefined_metric_and_bad_control_kind(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            behavior, control = build_eval_artifacts(tmp_path)

        bad_behavior = copy.deepcopy(behavior)
        bad_behavior["rows"][0]["metrics"]["not-defined"] = {"passed": True}
        self.assertTrue(
            any("not defined" in error for error in validate_behavior_eval(bad_behavior, spec))
        )

        bad_control = copy.deepcopy(control)
        bad_control["rows"][0]["control_kind"] = "causal_swap_claim"
        self.assertTrue(
            any("control_kind" in error for error in validate_control_eval(bad_control, spec))
        )

    def test_validators_reject_missing_metric_and_inconsistent_passed(self) -> None:
        spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            behavior, control = build_eval_artifacts(tmp_path)

        missing_metric = copy.deepcopy(behavior)
        missing_metric["rows"][0]["metrics"].pop("nonempty_output")
        self.assertTrue(
            any("missing metric results" in error for error in validate_behavior_eval(missing_metric, spec))
        )

        inconsistent = copy.deepcopy(behavior)
        inconsistent["rows"][0]["metrics"]["nonempty_output"]["passed"] = False
        inconsistent["rows"][0]["passed"] = True
        self.assertTrue(
            any("must equal all metric passed values" in error for error in validate_behavior_eval(inconsistent, spec))
        )

        mismatched_control_kind = copy.deepcopy(control)
        mismatched_control_kind["rows"][0]["control_kind"] = "random_direction"
        self.assertTrue(
            any("must match control_eval.control.kind" in error for error in validate_control_eval(mismatched_control_kind, spec))
        )

    def test_control_generator_rejects_row_kind_that_disagrees_with_cli_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            control_rows = example_control_rows()
            control_rows[0]["control_kind"] = "random_direction"
            control_path = tmp_path / "control-responses.jsonl"
            write_jsonl(control_path, control_rows)
            result = self.run_cli(
                "run-control-eval",
                "examples/workspace_audit_spec.json",
                "--responses",
                str(control_path),
                "--control-kind",
                "prompt_variant",
                "--out",
                str(tmp_path / "control.json"),
                "--tokenizer-revision",
                "fixture-tokenizer-revision",
                "--lens-revision",
                "fixture-lens-revision",
                "--fit-procedure",
                "fixture-fit-procedure",
                "--position-policy",
                "positions=-1",
                check=False,
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("must match the requested control kind", result.stderr)

    def test_strict_evidence_bundle_checks_eval_artifact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_negative_bundle(tmp_path)
            self.assertEqual([], validate_evidence_bundle(bundle, root=tmp_path, strict=True))

            behavior = load_json(tmp_path / "behavior.json")
            behavior["rows"] = behavior["rows"][:-1]
            write_json(tmp_path / "behavior.json", behavior)
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)

        self.assertTrue(any("artifact behavior" in error for error in errors))
        self.assertTrue(any("missing rows for prompt ids" in error for error in errors))


def build_eval_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    behavior_rows = root / "behavior-responses.jsonl"
    control_rows = root / "control-responses.jsonl"
    behavior_out = root / "behavior.json"
    control_out = root / "control.json"
    write_jsonl(behavior_rows, example_behavior_rows())
    write_jsonl(control_rows, example_control_rows())
    run_eval_cli(
        "run-behavior-eval",
        "--responses",
        str(behavior_rows),
        "--out",
        str(behavior_out),
    )
    run_eval_cli(
        "run-control-eval",
        "--responses",
        str(control_rows),
        "--control-kind",
        "prompt_variant",
        "--out",
        str(control_out),
    )
    return load_json(behavior_out), load_json(control_out)


def build_negative_bundle(root: Path) -> dict[str, Any]:
    spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
    readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
    report = score_readouts(spec, readouts)
    behavior, control = build_eval_artifacts(root)
    compatibility = behavior["compatibility"]

    write_json(root / "spec.json", spec)
    write_json(root / "readouts.json", readouts)
    write_json(root / "report.json", report)
    write_json(root / "behavior.json", behavior)
    write_json(root / "control.json", control)

    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA,
        "bundle_id": "negative-with-evals",
        "status": "negative",
        "claim": {
            "question": "Does the readout claim survive behavior/control review?",
            "hypothesis": "The synthetic fixture should not promote a real model claim.",
            "interpretation": "This negative bundle exercises behavior/control plumbing only.",
            "claim_scope": "run_specific",
            "non_claims": ["not a real model result"],
        },
        "compatibility": compatibility,
        "artifacts": [
            artifact("spec", "audit_spec", "spec.json", AUDIT_SPEC_SCHEMA),
            artifact("readouts", "readouts", "readouts.json", READOUT_SCHEMA),
            artifact("report", "audit_report_json", "report.json", REPORT_SCHEMA),
            artifact("behavior", "behavior_eval", "behavior.json", BEHAVIOR_EVAL_SCHEMA),
            artifact("control", "control_eval", "control.json", CONTROL_EVAL_SCHEMA),
        ],
        "pairings": [
            {
                "prompt_id": prompt["id"],
                "readout_artifact_id": "readouts",
                "behavior_artifact_id": "behavior",
                "control_artifact_ids": ["control"],
                "relation": "readout_behavior_support",
                "notes": "Synthetic fixture pairing for schema validation.",
            }
            for prompt in spec["prompts"]
        ],
        "status_gates": [
            {
                "name": "behavior_linked",
                "result": True,
                "artifact_refs": ["behavior"],
                "note": "Behavior artifact covers every prompt.",
            },
            {
                "name": "control_backed",
                "result": False,
                "artifact_refs": ["control"],
                "note": "Fixture intentionally treats controls as insufficient.",
            },
        ],
        "missing_evidence": [],
        "conflicting_findings": [],
        "review": {
            "reviewer": "test-suite",
            "reviewed_utc": "2026-07-07T00:00:00Z",
            "limitations": ["Synthetic fixture only."],
            "promotion_recommendation": "hold",
        },
    }


def run_eval_cli(command: str, *args: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "limes_workspace_lens",
            command,
            "examples/workspace_audit_spec.json",
            "--tokenizer-revision",
            "fixture-tokenizer-revision",
            "--lens-revision",
            "fixture-lens-revision",
            "--fit-procedure",
            "fixture-fit-procedure",
            "--position-policy",
            "positions=-1",
            "--seed",
            "7",
            *args,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def example_behavior_rows() -> list[dict[str, str]]:
    return [
        {
            "prompt_id": "math-copy",
            "response_id": "behavior-1",
            "output": "The old painting hung crookedly on the wall.",
            "finish_reason": "stop",
        },
        {
            "prompt_id": "prompt-injection-check",
            "response_id": "behavior-2",
            "output": "The snippet is untrusted because the source is not preserved.",
            "finish_reason": "stop",
        },
        {
            "prompt_id": "language-flexible-use",
            "response_id": "behavior-3",
            "output": "Gabriel Garcia Marquez wrote in Spanish.",
            "finish_reason": "stop",
        },
    ]


def example_control_rows() -> list[dict[str, str]]:
    return [
        {
            "prompt_id": "math-copy",
            "control_id": "math-copy:variant",
            "control_kind": "prompt_variant",
            "control_text": "Copy the painting sentence without the hidden arithmetic instruction.",
            "output": "The old painting hung crookedly on the wall.",
            "finish_reason": "stop",
        },
        {
            "prompt_id": "prompt-injection-check",
            "control_id": "prompt-injection-check:variant",
            "control_kind": "prompt_variant",
            "control_text": "Read a benign search-result snippet and decide whether to trust it.",
            "output": "The snippet is trusted only if source evidence is preserved.",
            "finish_reason": "stop",
        },
        {
            "prompt_id": "language-flexible-use",
            "control_id": "language-flexible-use:variant",
            "control_kind": "prompt_variant",
            "control_text": "El sol se escondia lentamente. Continue the passage.",
            "output": "El sendero quedo cubierto por una luz tenue.",
            "finish_reason": "stop",
        },
    ]


def artifact(artifact_id: str, kind: str, path: str, schema_version: str) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "kind": kind,
        "path": path,
        "schema_version": schema_version,
        "required_for_status": True,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
