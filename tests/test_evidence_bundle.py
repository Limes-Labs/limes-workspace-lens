from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from limes_workspace_lens.analysis import score_readouts
from limes_workspace_lens.evidence import EVIDENCE_BUNDLE_SCHEMA, validate_evidence_bundle
from limes_workspace_lens.eval_artifacts import build_behavior_eval, build_control_eval
from limes_workspace_lens.schema import AUDIT_SPEC_SCHEMA, READOUT_SCHEMA, REPORT_SCHEMA, load_json


ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_SCHEMA = "limes-workspace-lens/behavior-eval.v0.1"
CONTROL_SCHEMA = "limes-workspace-lens/control-eval.v0.1"


class EvidenceBundleTests(unittest.TestCase):
    def test_valid_diagnostic_bundle_with_explicit_missing_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="diagnostic", synthetic=True)
            self.assertEqual([], validate_evidence_bundle(bundle, root=tmp_path, strict=True))

    def test_cli_validates_diagnostic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="diagnostic", synthetic=True)
            bundle_path = tmp_path / "bundle.json"
            write_json(bundle_path, bundle)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "limes_workspace_lens",
                    "validate-bundle",
                    str(bundle_path),
                    "--root",
                    str(tmp_path),
                    "--strict",
                    "--expected-status",
                    "diagnostic",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual("", result.stderr)
        self.assertEqual(0, result.returncode)
        self.assertIn("valid:", result.stdout)

    def test_verified_bundle_fails_when_behavior_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=False)
            bundle["artifacts"] = [
                artifact for artifact in bundle["artifacts"] if artifact["id"] != "behavior"
            ]
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("behavior_eval" in error for error in errors))
        self.assertTrue(any("unknown artifact id 'behavior'" in error for error in errors))

    def test_verified_bundle_rejects_synthetic_readouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=True)
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("cannot use synthetic readouts" in error for error in errors))

    def test_mixed_bundle_requires_conflicting_findings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="mixed", synthetic=False)
            bundle["conflicting_findings"] = []
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("mixed bundles must list conflicts" in error for error in errors))

    def test_negative_bundle_requires_failed_behavior_or_control_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="negative", synthetic=False)
            for gate in bundle["status_gates"]:
                if gate["name"] in {"behavior_linked", "control_backed"}:
                    gate["result"] = True
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("failed behavior/control gate" in error for error in errors))

    def test_negative_bundle_requires_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="negative", synthetic=False)
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=False)
        self.assertTrue(any("negative validation requires --strict" in error for error in errors))

    def test_verified_bundle_rejects_failed_behavior_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=False)
            behavior = load_json(tmp_path / "behavior.json")
            behavior["rows"][0]["metrics"]["nonempty_output"]["passed"] = False
            behavior["rows"][0]["passed"] = False
            write_json(tmp_path / "behavior.json", behavior)
            for artifact_row in bundle["artifacts"]:
                if artifact_row["id"] == "behavior":
                    artifact_row["sha256"] = sha256(tmp_path / "behavior.json")
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("require passing behavior/control rows" in error for error in errors))

    def test_pairing_unknown_prompt_and_artifact_refs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="diagnostic", synthetic=True)
            bundle["pairings"][0]["prompt_id"] = "unknown-prompt"
            bundle["pairings"][0]["readout_artifact_id"] = "missing-readout"
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("unknown prompt id" in error for error in errors))
        self.assertTrue(any("unknown artifact id 'missing-readout'" in error for error in errors))

    def test_duplicate_artifact_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="diagnostic", synthetic=True)
            bundle["artifacts"].append(copy.deepcopy(bundle["artifacts"][0]))
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("duplicate artifact id" in error for error in errors))

    def test_strict_mode_fails_on_missing_referenced_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="diagnostic", synthetic=True)
            (tmp_path / "report.json").unlink()
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("referenced path does not exist" in error for error in errors))

    def test_strict_mode_rejects_symlink_escape_from_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            tmp_path = Path(tmp)
            outside_path = Path(outside)
            bundle = build_bundle(tmp_path, status="diagnostic", synthetic=True)
            external_report = outside_path / "external-report.json"
            external_report.write_text(
                (tmp_path / "report.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            escaped_link = tmp_path / "escaped-report.json"
            try:
                escaped_link.symlink_to(external_report)
            except OSError as exc:
                self.skipTest(f"symlinks are not available: {exc}")
            for artifact_row in bundle["artifacts"]:
                if artifact_row["id"] == "report":
                    artifact_row["path"] = "escaped-report.json"
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("escapes artifact root" in error for error in errors))

    def test_verified_bundle_requires_preserved_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=False)
            bundle["artifacts"][0].pop("sha256")
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("verified artifacts need SHA-256" in error for error in errors))

    def test_verified_required_kinds_must_be_marked_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=False)
            bundle["artifacts"][0]["required_for_status"] = False
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("required_for_status=true" in error for error in errors))

    def test_compatibility_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=False)
            bundle["compatibility"]["top_k"] = 99
            bundle["compatibility"]["tokenizer_revision"] = "wrong-tokenizer"
            bundle["compatibility"]["layer_policy"] = "wrong-layer-policy"
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any(".top_k" in error for error in errors))
        self.assertTrue(any(".compatibility.tokenizer_revision" in error for error in errors))
        self.assertTrue(any(".compatibility.layer_policy" in error for error in errors))

    def test_control_artifact_without_claimed_prompt_rows_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bundle = build_bundle(tmp_path, status="verified", synthetic=False)
            control = load_json(tmp_path / "control.json")
            control["rows"] = [
                row for row in control["rows"] if row["prompt_id"] != "math-copy"
            ]
            write_json(tmp_path / "control.json", control)
            errors = validate_evidence_bundle(bundle, root=tmp_path, strict=True)
        self.assertTrue(any("has no row for prompt 'math-copy'" in error for error in errors))
        self.assertTrue(any("missing rows for prompt ids" in error for error in errors))


def build_bundle(
    root: Path,
    *,
    status: str,
    synthetic: bool,
) -> dict[str, Any]:
    spec = load_json(ROOT / "examples" / "workspace_audit_spec.json")
    readouts = load_json(ROOT / "examples" / "synthetic_readouts.json")
    readouts["synthetic"] = synthetic
    readouts["source"] = "readouts.json"
    report = score_readouts(spec, readouts)
    compatibility = compatibility_from_spec(spec)

    behavior_responses = root / "behavior-responses.jsonl"
    control_responses = root / "control-responses.jsonl"
    write_jsonl(behavior_responses, behavior_response_rows())
    write_jsonl(control_responses, control_response_rows())
    behavior_eval = build_behavior_eval(
        spec,
        behavior_response_rows(),
        compatibility=compatibility,
        responses_path=str(behavior_responses),
        model_id=spec["model"]["name"],
        seed=7,
        command="test-suite run-behavior-eval",
    )
    control_eval = build_control_eval(
        spec,
        control_response_rows(),
        compatibility=compatibility,
        responses_path=str(control_responses),
        model_id=spec["model"]["name"],
        control_kind="prompt_variant",
        seed=7,
        command="test-suite run-control-eval",
    )
    command_log = {
        "schema_version": "limes-workspace-lens/command-log.v0.1",
        "compatibility": compatibility,
        "commands": ["python3 -m limes_workspace_lens summarize-readouts ..."],
    }
    compute_manifest = {
        "schema_version": "limes-workspace-lens/compute-manifest.v0.1",
        "compatibility": compatibility,
        "runtime": "test-fixture",
    }
    lens_identity = {
        "schema_version": "limes-workspace-lens/lens-artifact.v0.1",
        "compatibility": compatibility,
        "revision": "fixture-lens-revision",
    }

    write_json(root / "spec.json", spec)
    write_json(root / "readouts.json", readouts)
    write_json(root / "report.json", report)
    write_json(root / "behavior.json", behavior_eval)
    write_json(root / "control.json", control_eval)
    write_json(root / "command-log.json", command_log)
    write_json(root / "compute-manifest.json", compute_manifest)
    write_json(root / "lens-identity.json", lens_identity)

    artifacts = [
        artifact("spec", "audit_spec", "spec.json", AUDIT_SPEC_SCHEMA, status),
        artifact("readouts", "readouts", "readouts.json", READOUT_SCHEMA, status),
        artifact("report", "audit_report_json", "report.json", REPORT_SCHEMA, status),
    ]
    if status in {"mixed", "negative", "verified"}:
        artifacts.extend(
            [
                artifact("behavior", "behavior_eval", "behavior.json", BEHAVIOR_SCHEMA, status),
                artifact("control", "control_eval", "control.json", CONTROL_SCHEMA, status),
            ]
        )
    if status == "verified":
        artifacts.extend(
            [
                artifact(
                    "command-log",
                    "command_log",
                    "command-log.json",
                    "limes-workspace-lens/command-log.v0.1",
                    status,
                ),
                artifact(
                    "compute",
                    "compute_manifest",
                    "compute-manifest.json",
                    "limes-workspace-lens/compute-manifest.v0.1",
                    status,
                ),
                artifact(
                    "lens",
                    "lens_artifact_or_revision",
                    "lens-identity.json",
                    "limes-workspace-lens/lens-artifact.v0.1",
                    status,
                ),
            ]
        )

    if status == "verified":
        for row in artifacts:
            row["sha256"] = sha256(root / row["path"])

    pairings = []
    pairing_prompt_ids = (
        [prompt["id"] for prompt in spec["prompts"]]
        if status in {"mixed", "negative", "verified"}
        else ["math-copy"]
    )
    for prompt_id in pairing_prompt_ids:
        pairing = {
            "prompt_id": prompt_id,
            "readout_artifact_id": "readouts",
            "control_artifact_ids": [],
            "relation": "readout_behavior_support",
            "notes": "Fixture pairing for schema validation.",
        }
        if status in {"mixed", "negative", "verified"}:
            pairing["behavior_artifact_id"] = "behavior"
            pairing["control_artifact_ids"] = ["control"]
        if status == "mixed" and prompt_id == "math-copy":
            pairing["relation"] = "readout_behavior_conflict"
        pairings.append(pairing)

    return {
        "schema_version": EVIDENCE_BUNDLE_SCHEMA,
        "bundle_id": f"bundle-{status}",
        "status": status,
        "claim": {
            "question": "Does the readout signal survive review?",
            "hypothesis": "The fixture readout is useful only as a schema exercise.",
            "interpretation": "No real model claim is made.",
            "claim_scope": "hypothesis_generation" if status == "diagnostic" else "run_specific",
            "non_claims": ["not a model-quality score", "not evidence of consciousness"],
        },
        "compatibility": compatibility,
        "artifacts": artifacts,
        "pairings": pairings,
        "status_gates": gates_for_status(status),
        "missing_evidence": (
            [
                {"kind": "behavior_eval", "reason": "not run for diagnostic fixture"},
                {"kind": "control_eval", "reason": "not run for diagnostic fixture"},
            ]
            if status == "diagnostic"
            else []
        ),
        "conflicting_findings": (
            [{"finding": "Readout and control disagree in fixture.", "artifact_refs": ["control"]}]
            if status == "mixed"
            else []
        ),
        "review": {
            "reviewer": "test-suite",
            "reviewed_utc": "2026-07-07T00:00:00Z",
            "limitations": ["Fixture artifacts only."],
            "promotion_recommendation": "promote" if status == "verified" else "hold",
        },
    }


def compatibility_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_checkpoint": spec["model"]["checkpoint"],
        "tokenizer_revision": "fixture-tokenizer-revision",
        "lens_source": spec["lens"]["source"],
        "lens_revision": "fixture-lens-revision",
        "prompt_suite_hash": "fixture-prompt-suite-sha256",
        "top_k": spec["lens"]["top_k"],
        "layer_policy": "workspace_layer_range=24-40",
        "position_policy": "positions=-1",
        "fit_procedure": "fixture-fit-procedure",
    }


def behavior_response_rows() -> list[dict[str, str]]:
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
            "output": "The snippet is untrusted because the source evidence is missing.",
            "finish_reason": "stop",
        },
        {
            "prompt_id": "language-flexible-use",
            "response_id": "behavior-3",
            "output": "Gabriel Garcia Marquez wrote in Spanish.",
            "finish_reason": "stop",
        },
    ]


def control_response_rows() -> list[dict[str, str]]:
    return [
        {
            "prompt_id": "math-copy",
            "control_id": "math-copy:variant",
            "control_kind": "prompt_variant",
            "control_text": "Copy the painting sentence without hidden arithmetic.",
            "output": "The old painting hung crookedly on the wall.",
            "finish_reason": "stop",
        },
        {
            "prompt_id": "prompt-injection-check",
            "control_id": "prompt-injection-check:variant",
            "control_kind": "prompt_variant",
            "control_text": "Read a benign search result and decide whether to trust it.",
            "output": "The result needs source evidence before trust.",
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


def artifact(
    artifact_id: str,
    kind: str,
    path: str,
    schema_version: str,
    status: str,
) -> dict[str, Any]:
    return {
        "id": artifact_id,
        "kind": kind,
        "path": path,
        "schema_version": schema_version,
        "required_for_status": status in {"mixed", "negative", "verified"}
        or kind in {"audit_spec", "readouts", "audit_report_json"},
    }


def gates_for_status(status: str) -> list[dict[str, Any]]:
    if status == "diagnostic":
        return [
            {
                "name": "replayed",
                "result": False,
                "artifact_refs": ["spec", "readouts", "report"],
                "note": "Diagnostic fixture has not been replayed.",
            }
        ]
    gates = [
        {
            "name": "behavior_linked",
            "result": True,
            "artifact_refs": ["behavior"],
            "note": "Behavior artifact is paired with the prompt.",
        },
        {
            "name": "control_backed",
            "result": status != "negative",
            "artifact_refs": ["control"],
            "note": "Control artifact is paired with the prompt.",
        },
    ]
    if status == "verified":
        gates.extend(
            [
                {
                    "name": "replayed",
                    "result": True,
                    "artifact_refs": ["command-log"],
                    "note": "Commands were replayed.",
                },
                {
                    "name": "preserved",
                    "result": True,
                    "artifact_refs": ["compute", "lens"],
                    "note": "Artifacts carry preserved hashes.",
                },
                {
                    "name": "compatible_settings",
                    "result": True,
                    "artifact_refs": ["spec", "report", "behavior", "control"],
                    "note": "Compatibility objects match.",
                },
            ]
        )
    return gates


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
