from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .comparison import COMPARISON_SCHEMA
from .schema import (
    AUDIT_SPEC_SCHEMA,
    BEHAVIOR_EVAL_SCHEMA,
    COMMAND_LOG_SCHEMA,
    COMPUTE_MANIFEST_SCHEMA,
    CONTROL_EVAL_SCHEMA,
    INTERVENTION_SCHEMA,
    LENS_ARTIFACT_SCHEMA,
    READOUT_SCHEMA,
    REPORT_SCHEMA,
    require_keys,
    validate_behavior_eval,
    validate_audit_spec,
    validate_command_log,
    validate_compute_manifest,
    validate_control_eval,
    validate_lens_artifact,
    validate_readouts,
    validate_report,
)


EVIDENCE_BUNDLE_SCHEMA = "limes-workspace-lens/evidence-bundle.v0.1"

VALID_BUNDLE_STATUSES = {"diagnostic", "mixed", "negative", "verified"}
PROMOTION_RECOMMENDATIONS = {"hold", "review", "promote"}

REQUIRED_COMPATIBILITY_FIELDS = [
    "model_checkpoint",
    "tokenizer_revision",
    "lens_source",
    "lens_revision",
    "prompt_suite_hash",
    "top_k",
    "layer_policy",
    "position_policy",
    "fit_procedure",
]

KNOWN_ARTIFACT_SCHEMAS = {
    "audit_spec": AUDIT_SPEC_SCHEMA,
    "readouts": READOUT_SCHEMA,
    "audit_report_json": REPORT_SCHEMA,
    "comparison_json": COMPARISON_SCHEMA,
    "intervention_plan": INTERVENTION_SCHEMA,
    "behavior_eval": BEHAVIOR_EVAL_SCHEMA,
    "control_eval": CONTROL_EVAL_SCHEMA,
    "command_log": COMMAND_LOG_SCHEMA,
    "compute_manifest": COMPUTE_MANIFEST_SCHEMA,
    "lens_artifact_or_revision": LENS_ARTIFACT_SCHEMA,
}

BASE_REQUIRED_KINDS = {"audit_spec", "readouts", "audit_report_json"}
BEHAVIOR_REQUIRED_KINDS = BASE_REQUIRED_KINDS | {"behavior_eval", "control_eval"}
VERIFIED_REQUIRED_KINDS = BEHAVIOR_REQUIRED_KINDS | {
    "command_log",
    "compute_manifest",
    "lens_artifact_or_revision",
}
VERIFIED_REQUIRED_GATES = {
    "replayed",
    "preserved",
    "behavior_linked",
    "control_backed",
    "compatible_settings",
}
CONFLICT_RELATIONS = {"readout_behavior_conflict", "control_disagreement"}


def validate_evidence_bundle(
    bundle: dict[str, Any],
    *,
    root: str | Path | None = None,
    strict: bool = False,
    expected_status: str | None = None,
) -> list[str]:
    errors = require_keys(
        bundle,
        [
            "schema_version",
            "bundle_id",
            "status",
            "claim",
            "compatibility",
            "artifacts",
            "pairings",
            "status_gates",
            "missing_evidence",
            "conflicting_findings",
            "review",
        ],
        "bundle",
    )
    if bundle.get("schema_version") != EVIDENCE_BUNDLE_SCHEMA:
        errors.append(
            "bundle: schema_version must be "
            f"{EVIDENCE_BUNDLE_SCHEMA!r}, got {bundle.get('schema_version')!r}"
        )

    status = bundle.get("status")
    if status not in VALID_BUNDLE_STATUSES:
        errors.append(f"bundle.status: must be one of {sorted(VALID_BUNDLE_STATUSES)}")
    if expected_status is not None and status != expected_status:
        errors.append(f"bundle.status: expected {expected_status!r}, got {status!r}")
    if not _non_empty_string(bundle.get("bundle_id")):
        errors.append("bundle.bundle_id: must be a non-empty string")

    errors.extend(_validate_claim(bundle.get("claim"), status))
    compatibility = bundle.get("compatibility")
    errors.extend(_validate_compatibility(compatibility))

    root_path = Path(root).resolve() if root is not None else None
    artifacts, artifact_kinds, loaded_artifacts = _validate_artifacts(
        bundle.get("artifacts"), root_path, strict, errors
    )

    prompt_ids = _prompt_ids_from_loaded_specs(loaded_artifacts)
    pairings = bundle.get("pairings")
    pairing_prompt_ids = _validate_pairings(pairings, artifacts, prompt_ids, loaded_artifacts, errors)

    gate_results = _validate_status_gates(bundle.get("status_gates"), artifacts, errors)
    errors.extend(_validate_missing_evidence(bundle.get("missing_evidence"), artifacts))
    errors.extend(_validate_conflicting_findings(bundle.get("conflicting_findings"), artifacts))
    errors.extend(_validate_review(bundle.get("review")))
    errors.extend(_validate_loaded_eval_artifacts(loaded_artifacts))

    if status in VALID_BUNDLE_STATUSES:
        errors.extend(
            _validate_status_rules(
                status,
                bundle,
                artifact_kinds,
                artifacts,
                gate_results,
                loaded_artifacts,
                prompt_ids,
                pairing_prompt_ids,
                strict,
            )
        )
    if isinstance(compatibility, dict):
        errors.extend(_validate_loaded_compatibility(compatibility, loaded_artifacts))

    return errors


def _validate_claim(claim: Any, status: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(claim, dict):
        return ["bundle.claim: must be an object"]
    errors.extend(
        require_keys(
            claim,
            ["question", "hypothesis", "interpretation", "claim_scope", "non_claims"],
            "bundle.claim",
        )
    )
    for key in ["question", "hypothesis", "interpretation", "claim_scope"]:
        if not _non_empty_string(claim.get(key)):
            errors.append(f"bundle.claim.{key}: must be a non-empty string")
    if not _string_list(claim.get("non_claims")):
        errors.append("bundle.claim.non_claims: must be a list of strings")
    if status == "diagnostic" and claim.get("claim_scope") != "hypothesis_generation":
        errors.append("bundle.claim.claim_scope: diagnostic bundles must use hypothesis_generation")
    return errors


def _validate_compatibility(compatibility: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(compatibility, dict):
        return ["bundle.compatibility: must be an object"]
    errors.extend(require_keys(compatibility, REQUIRED_COMPATIBILITY_FIELDS, "bundle.compatibility"))
    for key in REQUIRED_COMPATIBILITY_FIELDS:
        value = compatibility.get(key)
        if key == "top_k":
            if not _positive_int(value):
                errors.append("bundle.compatibility.top_k: must be a positive integer")
        elif not _non_empty_string(value):
            errors.append(f"bundle.compatibility.{key}: must be a non-empty string")
    return errors


def _validate_artifacts(
    artifact_rows: Any,
    root: Path | None,
    strict: bool,
    errors: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_kinds: dict[str, set[str]] = {}
    loaded_artifacts: dict[str, dict[str, Any]] = {}

    if not isinstance(artifact_rows, list) or not artifact_rows:
        errors.append("bundle.artifacts: must be a non-empty list")
        return artifacts, artifact_kinds, loaded_artifacts

    seen: set[str] = set()
    for index, artifact in enumerate(artifact_rows):
        where = f"bundle.artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(
            require_keys(
                artifact,
                ["id", "kind", "path", "schema_version", "required_for_status"],
                where,
            )
        )
        artifact_id = artifact.get("id")
        kind = artifact.get("kind")
        if not _non_empty_string(artifact_id):
            errors.append(f"{where}.id: must be a non-empty string")
            continue
        if artifact_id in seen:
            errors.append(f"{where}.id: duplicate artifact id {artifact_id!r}")
        seen.add(artifact_id)

        if not _non_empty_string(kind):
            errors.append(f"{where}.kind: must be a non-empty string")
        if not _safe_relative_path(artifact.get("path")):
            errors.append(f"{where}.path: must be a safe relative path")
        if not _non_empty_string(artifact.get("schema_version")):
            errors.append(f"{where}.schema_version: must be a non-empty string")
        expected_schema = KNOWN_ARTIFACT_SCHEMAS.get(kind)
        if expected_schema is not None and artifact.get("schema_version") != expected_schema:
            errors.append(f"{where}.schema_version: must be {expected_schema!r} for {kind}")
        if not isinstance(artifact.get("required_for_status"), bool):
            errors.append(f"{where}.required_for_status: must be a boolean")

        artifacts[artifact_id] = artifact
        artifact_kinds.setdefault(str(kind), set()).add(artifact_id)

        try:
            artifact_path = _resolve_artifact_path(root, artifact)
        except ValueError as exc:
            errors.append(f"{where}.path: {exc}")
            continue
        if artifact_path is None:
            continue
        if not artifact_path.exists():
            if strict:
                errors.append(f"{where}.path: referenced path does not exist: {artifact.get('path')}")
            continue
        if artifact.get("sha256") is not None:
            if not _valid_sha256(artifact.get("sha256")):
                errors.append(f"{where}.sha256: must be a lowercase SHA-256 hex digest")
            elif _sha256(artifact_path) != artifact["sha256"]:
                errors.append(f"{where}.sha256: digest does not match referenced file")
        if _json_like_artifact(str(kind), artifact_path):
            try:
                loaded_artifacts[artifact_id] = _load_json_object(artifact_path)
            except ValueError as exc:
                errors.append(f"{where}.path: {exc}")

    return artifacts, artifact_kinds, loaded_artifacts


def _validate_pairings(
    pairings: Any,
    artifacts: dict[str, dict[str, Any]],
    known_prompt_ids: set[str],
    loaded_artifacts: dict[str, dict[str, Any]],
    errors: list[str],
) -> set[str]:
    pairing_prompt_ids: set[str] = set()
    if not isinstance(pairings, list) or not pairings:
        errors.append("bundle.pairings: must be a non-empty list")
        return pairing_prompt_ids

    for index, pairing in enumerate(pairings):
        where = f"bundle.pairings[{index}]"
        if not isinstance(pairing, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(
            require_keys(
                pairing,
                ["prompt_id", "readout_artifact_id", "control_artifact_ids", "relation", "notes"],
                where,
            )
        )
        prompt_id = pairing.get("prompt_id")
        if not _non_empty_string(prompt_id):
            errors.append(f"{where}.prompt_id: must be a non-empty string")
        else:
            pairing_prompt_ids.add(prompt_id)
            if known_prompt_ids and prompt_id not in known_prompt_ids:
                errors.append(f"{where}.prompt_id: unknown prompt id {prompt_id!r}")

        readout_artifact_id = pairing.get("readout_artifact_id")
        if not _artifact_ref(readout_artifact_id, artifacts, where, "readout_artifact_id", errors):
            pass
        elif artifacts[readout_artifact_id].get("kind") != "readouts":
            errors.append(f"{where}.readout_artifact_id: must reference a readouts artifact")

        behavior_artifact_id = pairing.get("behavior_artifact_id")
        if behavior_artifact_id is not None:
            if _artifact_ref(
                behavior_artifact_id, artifacts, where, "behavior_artifact_id", errors
            ) and artifacts[behavior_artifact_id].get("kind") != "behavior_eval":
                errors.append(f"{where}.behavior_artifact_id: must reference a behavior_eval artifact")
            _validate_prompt_coverage(
                loaded_artifacts.get(str(behavior_artifact_id)),
                prompt_id,
                where,
                "behavior_artifact_id",
                errors,
            )

        control_artifact_ids = pairing.get("control_artifact_ids")
        if not isinstance(control_artifact_ids, list):
            errors.append(f"{where}.control_artifact_ids: must be a list")
        else:
            for control_id in control_artifact_ids:
                if _artifact_ref(control_id, artifacts, where, "control_artifact_ids", errors):
                    if artifacts[control_id].get("kind") != "control_eval":
                        errors.append(
                            f"{where}.control_artifact_ids: must reference control_eval artifacts"
                        )
                    _validate_prompt_coverage(
                        loaded_artifacts.get(str(control_id)),
                        prompt_id,
                        where,
                        "control_artifact_ids",
                        errors,
                    )
        if not _non_empty_string(pairing.get("relation")):
            errors.append(f"{where}.relation: must be a non-empty string")
        if not _non_empty_string(pairing.get("notes")):
            errors.append(f"{where}.notes: must be a non-empty string")

    return pairing_prompt_ids


def _validate_status_gates(
    gates: Any,
    artifacts: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, bool]:
    gate_results: dict[str, bool] = {}
    if not isinstance(gates, list) or not gates:
        errors.append("bundle.status_gates: must be a non-empty list")
        return gate_results
    for index, gate in enumerate(gates):
        where = f"bundle.status_gates[{index}]"
        if not isinstance(gate, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(require_keys(gate, ["name", "result", "artifact_refs", "note"], where))
        name = gate.get("name")
        if not _non_empty_string(name):
            errors.append(f"{where}.name: must be a non-empty string")
        result = gate.get("result")
        if not isinstance(result, bool):
            errors.append(f"{where}.result: must be a boolean")
        elif isinstance(name, str):
            gate_results[name] = result
        artifact_refs = gate.get("artifact_refs")
        if not isinstance(artifact_refs, list):
            errors.append(f"{where}.artifact_refs: must be a list")
        else:
            for artifact_ref in artifact_refs:
                _artifact_ref(artifact_ref, artifacts, where, "artifact_refs", errors)
        if not _non_empty_string(gate.get("note")):
            errors.append(f"{where}.note: must be a non-empty string")
    return gate_results


def _validate_missing_evidence(
    missing_evidence: Any, artifacts: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(missing_evidence, list):
        return ["bundle.missing_evidence: must be a list"]
    for index, item in enumerate(missing_evidence):
        where = f"bundle.missing_evidence[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(require_keys(item, ["kind", "reason"], where))
        if not _non_empty_string(item.get("kind")):
            errors.append(f"{where}.kind: must be a non-empty string")
        if not _non_empty_string(item.get("reason")):
            errors.append(f"{where}.reason: must be a non-empty string")
        for artifact_ref in item.get("artifact_refs", []):
            _artifact_ref(artifact_ref, artifacts, where, "artifact_refs", errors)
    return errors


def _validate_conflicting_findings(
    conflicting_findings: Any, artifacts: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(conflicting_findings, list):
        return ["bundle.conflicting_findings: must be a list"]
    for index, item in enumerate(conflicting_findings):
        where = f"bundle.conflicting_findings[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(require_keys(item, ["finding", "artifact_refs"], where))
        if not _non_empty_string(item.get("finding")):
            errors.append(f"{where}.finding: must be a non-empty string")
        artifact_refs = item.get("artifact_refs")
        if not isinstance(artifact_refs, list) or not artifact_refs:
            errors.append(f"{where}.artifact_refs: must be a non-empty list")
        else:
            for artifact_ref in artifact_refs:
                _artifact_ref(artifact_ref, artifacts, where, "artifact_refs", errors)
    return errors


def _validate_review(review: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(review, dict):
        return ["bundle.review: must be an object"]
    errors.extend(
        require_keys(
            review,
            ["reviewer", "reviewed_utc", "limitations", "promotion_recommendation"],
            "bundle.review",
        )
    )
    if not _non_empty_string(review.get("reviewer")):
        errors.append("bundle.review.reviewer: must be a non-empty string")
    reviewed_utc = review.get("reviewed_utc")
    if not _non_empty_string(reviewed_utc) or not _parse_datetime(str(reviewed_utc)):
        errors.append("bundle.review.reviewed_utc: must be an ISO-8601 datetime string")
    if not _string_list(review.get("limitations")):
        errors.append("bundle.review.limitations: must be a list of strings")
    recommendation = review.get("promotion_recommendation")
    if recommendation not in PROMOTION_RECOMMENDATIONS:
        errors.append(
            "bundle.review.promotion_recommendation: must be one of "
            f"{sorted(PROMOTION_RECOMMENDATIONS)}"
        )
    return errors


def _validate_status_rules(
    status: str,
    bundle: dict[str, Any],
    artifact_kinds: dict[str, set[str]],
    artifacts: dict[str, dict[str, Any]],
    gate_results: dict[str, bool],
    loaded_artifacts: dict[str, dict[str, Any]],
    prompt_ids: set[str],
    pairing_prompt_ids: set[str],
    strict: bool,
) -> list[str]:
    errors: list[str] = []
    review = bundle.get("review", {})
    recommendation = review.get("promotion_recommendation") if isinstance(review, dict) else None
    missing_evidence = bundle.get("missing_evidence", [])
    conflicting_findings = bundle.get("conflicting_findings", [])
    pairings = bundle.get("pairings", [])

    if status == "diagnostic":
        errors.extend(_require_artifact_kinds(artifact_kinds, artifacts, BASE_REQUIRED_KINDS, status))
        if not missing_evidence:
            errors.append("bundle.missing_evidence: diagnostic bundles must list missing evidence")
        if recommendation == "promote":
            errors.append("bundle.review.promotion_recommendation: diagnostic bundles cannot promote")
        return errors

    if status == "mixed":
        if not strict:
            errors.append("bundle.status: mixed validation requires --strict")
        errors.extend(
            _require_artifact_kinds(artifact_kinds, artifacts, BEHAVIOR_REQUIRED_KINDS, status)
        )
        if not conflicting_findings:
            errors.append("bundle.conflicting_findings: mixed bundles must list conflicts")
        if recommendation not in {"hold", "review"}:
            errors.append("bundle.review.promotion_recommendation: mixed bundles require hold or review")
        if not _has_conflict_relation(pairings):
            errors.append("bundle.pairings: mixed bundles need a conflict relation")
        errors.extend(_require_behavior_and_controls(pairings, status))
        errors.extend(_require_pairing_prompt_coverage(prompt_ids, pairing_prompt_ids, status))
        return errors

    if status == "negative":
        if not strict:
            errors.append("bundle.status: negative validation requires --strict")
        errors.extend(
            _require_artifact_kinds(artifact_kinds, artifacts, BEHAVIOR_REQUIRED_KINDS, status)
        )
        if recommendation not in {"hold", "review"}:
            errors.append("bundle.review.promotion_recommendation: negative bundles require hold or review")
        if not _has_failed_behavior_or_control_gate(gate_results):
            errors.append("bundle.status_gates: negative bundles need a failed behavior/control gate")
        errors.extend(_require_behavior_and_controls(pairings, status))
        errors.extend(_require_pairing_prompt_coverage(prompt_ids, pairing_prompt_ids, status))
        return errors

    if status == "verified":
        if not strict:
            errors.append("bundle.status: verified validation requires --strict")
        errors.extend(
            _require_artifact_kinds(artifact_kinds, artifacts, VERIFIED_REQUIRED_KINDS, status)
        )
        missing_gates = sorted(VERIFIED_REQUIRED_GATES - set(gate_results))
        if missing_gates:
            errors.append(f"bundle.status_gates: verified bundles are missing gates {missing_gates}")
        failed_gates = sorted(name for name in VERIFIED_REQUIRED_GATES if gate_results.get(name) is False)
        if failed_gates:
            errors.append(f"bundle.status_gates: verified gates failed {failed_gates}")
        errors.extend(_require_behavior_and_controls(pairings, status))
        errors.extend(_require_pairing_prompt_coverage(prompt_ids, pairing_prompt_ids, status))
        errors.extend(_require_eval_rows_passed(loaded_artifacts, status))
        errors.extend(_require_command_logs_succeeded(loaded_artifacts, status))
        errors.extend(_require_preserved_artifact_hashes(artifacts))
        errors.extend(_reject_synthetic_verified_readouts(loaded_artifacts))
        return errors

    return errors


def _validate_loaded_compatibility(
    compatibility: dict[str, Any], loaded_artifacts: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    for artifact_id, artifact in loaded_artifacts.items():
        artifact_compatibility = artifact.get("compatibility")
        if isinstance(artifact_compatibility, dict):
            for key in REQUIRED_COMPATIBILITY_FIELDS:
                _expect_equal(
                    artifact_compatibility.get(key),
                    compatibility.get(key),
                    f"artifact {artifact_id}.compatibility.{key}",
                    errors,
                )
        if artifact.get("schema_version") == AUDIT_SPEC_SCHEMA:
            model = artifact.get("model", {})
            lens = artifact.get("lens", {})
            _expect_equal(
                model.get("checkpoint"),
                compatibility.get("model_checkpoint"),
                f"artifact {artifact_id}.model.checkpoint",
                errors,
            )
            _expect_equal(
                lens.get("source"),
                compatibility.get("lens_source"),
                f"artifact {artifact_id}.lens.source",
                errors,
            )
            if "top_k" in lens:
                _expect_equal(
                    lens.get("top_k"),
                    compatibility.get("top_k"),
                    f"artifact {artifact_id}.lens.top_k",
                    errors,
                )
        if artifact.get("schema_version") == REPORT_SCHEMA:
            model = artifact.get("model", {})
            lens = artifact.get("lens", {})
            _expect_equal(
                model.get("checkpoint"),
                compatibility.get("model_checkpoint"),
                f"artifact {artifact_id}.model.checkpoint",
                errors,
            )
            _expect_equal(
                lens.get("source"),
                compatibility.get("lens_source"),
                f"artifact {artifact_id}.lens.source",
                errors,
            )
            _expect_equal(
                artifact.get("top_k"),
                compatibility.get("top_k"),
                f"artifact {artifact_id}.top_k",
                errors,
            )
        if artifact.get("schema_version") == READOUT_SCHEMA and "top_k" in artifact:
            _expect_equal(
                artifact.get("top_k"),
                compatibility.get("top_k"),
                f"artifact {artifact_id}.top_k",
                errors,
            )
    return errors


def _validate_loaded_eval_artifacts(loaded_artifacts: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    spec = _loaded_audit_spec(loaded_artifacts)
    for artifact_id, artifact in loaded_artifacts.items():
        schema_version = artifact.get("schema_version")
        if schema_version == AUDIT_SPEC_SCHEMA:
            errors.extend(f"artifact {artifact_id}: {error}" for error in validate_audit_spec(artifact))
        if schema_version == READOUT_SCHEMA:
            errors.extend(
                f"artifact {artifact_id}: {error}" for error in validate_readouts(artifact, spec)
            )
        if schema_version == REPORT_SCHEMA:
            errors.extend(f"artifact {artifact_id}: {error}" for error in validate_report(artifact))
        if schema_version == BEHAVIOR_EVAL_SCHEMA:
            errors.extend(
                f"artifact {artifact_id}: {error}" for error in validate_behavior_eval(artifact, spec)
            )
        if schema_version == CONTROL_EVAL_SCHEMA:
            errors.extend(
                f"artifact {artifact_id}: {error}" for error in validate_control_eval(artifact, spec)
            )
        if schema_version == COMMAND_LOG_SCHEMA:
            errors.extend(f"artifact {artifact_id}: {error}" for error in validate_command_log(artifact))
        if schema_version == COMPUTE_MANIFEST_SCHEMA:
            errors.extend(
                f"artifact {artifact_id}: {error}" for error in validate_compute_manifest(artifact)
            )
        if schema_version == LENS_ARTIFACT_SCHEMA:
            errors.extend(f"artifact {artifact_id}: {error}" for error in validate_lens_artifact(artifact))
    return errors


def _require_eval_rows_passed(
    loaded_artifacts: dict[str, dict[str, Any]],
    status: str,
) -> list[str]:
    errors: list[str] = []
    for artifact_id, artifact in loaded_artifacts.items():
        if artifact.get("schema_version") not in {BEHAVIOR_EVAL_SCHEMA, CONTROL_EVAL_SCHEMA}:
            continue
        rows = artifact.get("rows")
        if not isinstance(rows, list):
            continue
        failed_prompts = sorted(
            str(row.get("prompt_id"))
            for row in rows
            if isinstance(row, dict) and row.get("passed") is False
        )
        if failed_prompts:
            errors.append(
                f"bundle.artifacts[{artifact_id}]: {status} bundles require passing "
                f"behavior/control rows, failed prompts {failed_prompts}"
            )
    return errors


def _require_command_logs_succeeded(
    loaded_artifacts: dict[str, dict[str, Any]],
    status: str,
) -> list[str]:
    errors: list[str] = []
    for artifact_id, artifact in loaded_artifacts.items():
        if artifact.get("schema_version") != COMMAND_LOG_SCHEMA:
            continue
        commands = artifact.get("commands")
        if not isinstance(commands, list):
            continue
        failed_command_ids = sorted(
            str(command.get("id", index))
            for index, command in enumerate(commands)
            if isinstance(command, dict) and command.get("exit_code") != 0
        )
        if failed_command_ids:
            errors.append(
                f"bundle.artifacts[{artifact_id}]: {status} bundles require successful "
                f"command logs, failed commands {failed_command_ids}"
            )
    return errors


def _require_artifact_kinds(
    artifact_kinds: dict[str, set[str]],
    artifacts: dict[str, dict[str, Any]],
    required_kinds: set[str],
    status: str,
) -> list[str]:
    missing = sorted(kind for kind in required_kinds if not artifact_kinds.get(kind))
    errors = []
    if missing:
        errors.append(f"bundle.artifacts: {status} bundles require artifact kinds {missing}")
    not_required = []
    for kind in required_kinds - set(missing):
        ids = artifact_kinds.get(kind, set())
        if not any(artifacts[artifact_id].get("required_for_status") is True for artifact_id in ids):
            not_required.append(kind)
    if not_required:
        errors.append(
            f"bundle.artifacts: {status} bundles must mark required kinds {sorted(not_required)} "
            "with required_for_status=true"
        )
    return errors


def _require_behavior_and_controls(pairings: Any, status: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(pairings, list):
        return errors
    for index, pairing in enumerate(pairings):
        if not isinstance(pairing, dict):
            continue
        if not _non_empty_string(pairing.get("behavior_artifact_id")):
            errors.append(
                f"bundle.pairings[{index}].behavior_artifact_id: {status} bundles require behavior"
            )
        controls = pairing.get("control_artifact_ids")
        if not isinstance(controls, list) or not controls:
            errors.append(
                f"bundle.pairings[{index}].control_artifact_ids: "
                f"{status} bundles require controls"
            )
    return errors


def _require_pairing_prompt_coverage(
    prompt_ids: set[str],
    pairing_prompt_ids: set[str],
    status: str,
) -> list[str]:
    if not prompt_ids:
        return []
    missing = sorted(prompt_ids - pairing_prompt_ids)
    if not missing:
        return []
    return [f"bundle.pairings: {status} bundles are missing prompt pairings {missing}"]


def _require_preserved_artifact_hashes(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for artifact_id, artifact in artifacts.items():
        if artifact.get("required_for_status") and not _valid_sha256(artifact.get("sha256")):
            errors.append(f"bundle.artifacts[{artifact_id}].sha256: verified artifacts need SHA-256")
    return errors


def _reject_synthetic_verified_readouts(loaded_artifacts: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for artifact_id, artifact in loaded_artifacts.items():
        if artifact.get("schema_version") == READOUT_SCHEMA and artifact.get("synthetic") is True:
            errors.append(f"bundle.artifacts[{artifact_id}]: verified bundles cannot use synthetic readouts")
    return errors


def _has_conflict_relation(pairings: Any) -> bool:
    if not isinstance(pairings, list):
        return False
    return any(
        isinstance(pairing, dict) and pairing.get("relation") in CONFLICT_RELATIONS
        for pairing in pairings
    )


def _has_failed_behavior_or_control_gate(gate_results: dict[str, bool]) -> bool:
    return gate_results.get("behavior_linked") is False or gate_results.get("control_backed") is False


def _validate_prompt_coverage(
    artifact: dict[str, Any] | None,
    prompt_id: Any,
    where: str,
    field: str,
    errors: list[str],
) -> None:
    if artifact is None or not _non_empty_string(prompt_id):
        return
    covered = _prompt_ids_from_eval_artifact(artifact)
    if not covered:
        errors.append(f"{where}.{field}: referenced artifact has no prompt rows")
    elif prompt_id not in covered:
        errors.append(f"{where}.{field}: referenced artifact has no row for prompt {prompt_id!r}")


def _prompt_ids_from_loaded_specs(loaded_artifacts: dict[str, dict[str, Any]]) -> set[str]:
    prompt_ids: set[str] = set()
    for artifact in loaded_artifacts.values():
        if artifact.get("schema_version") != AUDIT_SPEC_SCHEMA:
            continue
        prompts = artifact.get("prompts", [])
        if isinstance(prompts, list):
            prompt_ids.update(
                prompt["id"]
                for prompt in prompts
                if isinstance(prompt, dict) and _non_empty_string(prompt.get("id"))
            )
    return prompt_ids


def _loaded_audit_spec(loaded_artifacts: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for artifact in loaded_artifacts.values():
        if artifact.get("schema_version") == AUDIT_SPEC_SCHEMA:
            return artifact
    return None


def _prompt_ids_from_eval_artifact(artifact: dict[str, Any]) -> set[str]:
    rows = artifact.get("rows", artifact.get("results", []))
    if not isinstance(rows, list):
        return set()
    return {
        row["prompt_id"]
        for row in rows
        if isinstance(row, dict) and _non_empty_string(row.get("prompt_id"))
    }


def _artifact_ref(
    value: Any,
    artifacts: dict[str, dict[str, Any]],
    where: str,
    field: str,
    errors: list[str],
) -> bool:
    if not _non_empty_string(value):
        errors.append(f"{where}.{field}: must be a non-empty artifact id")
        return False
    if value not in artifacts:
        errors.append(f"{where}.{field}: unknown artifact id {value!r}")
        return False
    return True


def _resolve_artifact_path(root: Path | None, artifact: dict[str, Any]) -> Path | None:
    path = artifact.get("path")
    if root is None or not _safe_relative_path(path):
        return None
    candidate = (root / str(path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{path!r} escapes artifact root") from exc
    return candidate


def _json_like_artifact(kind: str, path: Path) -> bool:
    return kind.endswith("_json") or kind in {
        "audit_spec",
        "readouts",
        "behavior_eval",
        "control_eval",
        "compute_manifest",
    } or path.suffix == ".json"


def _safe_relative_path(path: Any) -> bool:
    if not _non_empty_string(path):
        return False
    pure = PurePosixPath(str(path))
    return not pure.is_absolute() and ".." not in pure.parts


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("referenced JSON artifact must contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _expect_equal(actual: Any, expected: Any, where: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{where}: expected {expected!r}, got {actual!r}")


def _parse_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
