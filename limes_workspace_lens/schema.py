from __future__ import annotations

import json
from pathlib import Path
from typing import Any


AUDIT_SPEC_SCHEMA = "limes-workspace-lens/audit-spec.v0.1"
READOUT_SCHEMA = "limes-workspace-lens/readouts.v0.1"
REPORT_SCHEMA = "limes-workspace-lens/report.v0.1"
REFLECTION_SCHEMA = "limes-workspace-lens/reflection-jsonl.v0.1"
INTERVENTION_SCHEMA = "limes-workspace-lens/intervention-plan.v0.1"


class ValidationError(ValueError):
    """Raised when a public artifact does not satisfy the repo schema."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def require_keys(data: dict[str, Any], keys: list[str], where: str) -> list[str]:
    errors: list[str] = []
    for key in keys:
        if key not in data:
            errors.append(f"{where}: missing required key '{key}'")
    return errors


def validate_audit_spec(spec: dict[str, Any]) -> list[str]:
    errors = require_keys(
        spec,
        ["schema_version", "project", "model", "lens", "prompts", "audit_terms"],
        "spec",
    )
    if spec.get("schema_version") != AUDIT_SPEC_SCHEMA:
        errors.append(
            f"spec: schema_version must be {AUDIT_SPEC_SCHEMA!r}, got {spec.get('schema_version')!r}"
        )

    for key in ["project", "model", "lens", "audit_terms"]:
        if key in spec and not isinstance(spec[key], dict):
            errors.append(f"spec.{key}: must be an object")

    prompts = spec.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        errors.append("spec.prompts: must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, prompt in enumerate(prompts):
            where = f"spec.prompts[{index}]"
            if not isinstance(prompt, dict):
                errors.append(f"{where}: must be an object")
                continue
            errors.extend(require_keys(prompt, ["id", "kind", "text"], where))
            prompt_id = prompt.get("id")
            if isinstance(prompt_id, str):
                if prompt_id in seen:
                    errors.append(f"{where}.id: duplicate prompt id {prompt_id!r}")
                seen.add(prompt_id)
            else:
                errors.append(f"{where}.id: must be a string")
            if "expected_workspace_terms" in prompt and not _is_string_list(
                prompt["expected_workspace_terms"]
            ):
                errors.append(f"{where}.expected_workspace_terms: must be a list of strings")

    audit_terms = spec.get("audit_terms")
    if isinstance(audit_terms, dict):
        for name, terms in audit_terms.items():
            if not isinstance(name, str) or not name:
                errors.append("spec.audit_terms: category names must be non-empty strings")
            if not _is_string_list(terms):
                errors.append(f"spec.audit_terms.{name}: must be a list of strings")

    reflection = spec.get("reflection_training")
    if reflection is not None:
        if not isinstance(reflection, dict):
            errors.append("spec.reflection_training: must be an object")
        elif not _is_string_list(reflection.get("principles", [])):
            errors.append("spec.reflection_training.principles: must be a list of strings")

    interventions = spec.get("interventions", [])
    if interventions is not None:
        if not isinstance(interventions, list):
            errors.append("spec.interventions: must be a list")
        else:
            for index, intervention in enumerate(interventions):
                where = f"spec.interventions[{index}]"
                if not isinstance(intervention, dict):
                    errors.append(f"{where}: must be an object")
                    continue
                errors.extend(require_keys(intervention, ["id", "prompt_id", "kind"], where))

    return errors


def validate_readouts(readouts: dict[str, Any]) -> list[str]:
    errors = require_keys(readouts, ["schema_version", "readouts"], "readouts")
    if readouts.get("schema_version") != READOUT_SCHEMA:
        errors.append(
            f"readouts: schema_version must be {READOUT_SCHEMA!r}, got {readouts.get('schema_version')!r}"
        )
    rows = readouts.get("readouts")
    if not isinstance(rows, list) or not rows:
        errors.append("readouts.readouts: must be a non-empty list")
        return errors
    for index, row in enumerate(rows):
        where = f"readouts.readouts[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: must be an object")
            continue
        errors.extend(require_keys(row, ["prompt_id", "position", "layer", "top_tokens"], where))
        if not isinstance(row.get("layer"), int):
            errors.append(f"{where}.layer: must be an integer")
        top_tokens = row.get("top_tokens")
        if not isinstance(top_tokens, list) or not top_tokens:
            errors.append(f"{where}.top_tokens: must be a non-empty list")
            continue
        for token_index, token in enumerate(top_tokens):
            token_where = f"{where}.top_tokens[{token_index}]"
            if not isinstance(token, dict):
                errors.append(f"{token_where}: must be an object")
                continue
            errors.extend(require_keys(token, ["token", "rank"], token_where))
            if not isinstance(token.get("token"), str):
                errors.append(f"{token_where}.token: must be a string")
            if not isinstance(token.get("rank"), int):
                errors.append(f"{token_where}.rank: must be an integer")
    return errors


def validate_report(report: dict[str, Any]) -> list[str]:
    errors = require_keys(
        report,
        ["schema_version", "project", "model", "lens", "input_readouts", "top_k", "prompt_summaries"],
        "report",
    )
    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append(
            f"report: schema_version must be {REPORT_SCHEMA!r}, got {report.get('schema_version')!r}"
        )
    for key in ["project", "model", "lens", "input_readouts"]:
        if key in report and not isinstance(report[key], dict):
            errors.append(f"report.{key}: must be an object")
    top_k = report.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        errors.append("report.top_k: must be a positive integer")
    category_counts = report.get("category_counts", {})
    if category_counts is not None:
        if not isinstance(category_counts, dict):
            errors.append("report.category_counts: must be an object")
        else:
            for category, count in category_counts.items():
                if not isinstance(category, str):
                    errors.append("report.category_counts: category names must be strings")
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    errors.append(f"report.category_counts.{category}: must be a non-negative integer")
    prompt_summaries = report.get("prompt_summaries")
    if not isinstance(prompt_summaries, list) or not prompt_summaries:
        errors.append("report.prompt_summaries: must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, row in enumerate(prompt_summaries):
            where = f"report.prompt_summaries[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{where}: must be an object")
                continue
            errors.extend(require_keys(row, ["prompt_id", "status"], where))
            prompt_id = row.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                errors.append(f"{where}.prompt_id: must be a non-empty string")
            elif prompt_id in seen:
                errors.append(f"{where}.prompt_id: duplicate prompt id {prompt_id!r}")
            else:
                seen.add(prompt_id)
            for count_key in ["expected_workspace_term_hits", "audit_term_hits"]:
                if count_key in row and (
                    not isinstance(row[count_key], int)
                    or isinstance(row[count_key], bool)
                    or row[count_key] < 0
                ):
                    errors.append(f"{where}.{count_key}: must be a non-negative integer")
    return errors


def ensure_valid(errors: list[str]) -> None:
    if errors:
        raise ValidationError("\n".join(errors))


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)
